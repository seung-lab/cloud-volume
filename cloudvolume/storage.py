import six
from six.moves import queue as Queue
from collections import defaultdict
import json
import os.path
import re
from functools import partial

import boto3
import botocore
from glob import glob
import google.cloud.exceptions
from google.cloud.storage import Client
import requests
import tenacity
from tqdm import tqdm

from . import compression
from .lib import mkdir, extract_path
from .threaded_queue import ThreadedQueue
from .connectionpools import S3ConnectionPool, GCloudBucketPool

# This is just to support pooling by bucket
class keydefaultdict(defaultdict):
  def __missing__(self, key):
    if self.default_factory is None:
      raise KeyError( key )
    else:
      ret = self[key] = self.default_factory(key)
      return ret

S3_POOL = None
GC_POOL = None
def reset_connection_pools():
  global S3_POOL
  global GC_POOL
  S3_POOL = keydefaultdict(lambda service: keydefaultdict(lambda bucket_name: S3ConnectionPool(service, bucket_name)))
  GC_POOL = keydefaultdict(lambda bucket_name: GCloudBucketPool(bucket_name))

reset_connection_pools()

retry = tenacity.retry(
  reraise=True, 
  stop=tenacity.stop_after_attempt(7), 
  wait=tenacity.wait_full_jitter(0.5, 60.0),
)

DEFAULT_THREADS = 20

class UnsupportedProtocol(Exception):
  pass

class SimpleStorage(object):
  """
  Access files stored in Google Storage (gs), Amazon S3 (s3), 
  or the local Filesystem (file).

  e.g. with Storage('gs://bucket/dataset/layer') as stor:
      files = stor.get_file('filename')

  Required:
    layer_path (str): A protocol prefixed path of the above format.
      Accepts s3:// gs:// and file://. File paths are absolute.

  Optional:
    n_threads (int:20): number of threads to use downloading and uplaoding.
      If 0, execution will be on the main python thread.
    progress (bool:false): Show a tqdm progress bar for multiple 
      uploads and downloads.
  """
  def __init__(self, layer_path, progress=False):

    self.progress = progress

    self._layer_path = layer_path
    self._path = extract_path(layer_path)
    
    if self._path.protocol == 'file':
      self._interface_cls = FileInterface
    elif self._path.protocol == 'gs':
      self._interface_cls = GoogleCloudStorageInterface
    elif self._path.protocol in ('s3', 'matrix'):
      self._interface_cls = S3Interface
    elif self._path.protocol in ('http', 'https'):
      self._interface_cls = HttpInterface
    else:
      raise UnsupportedProtocol(str(self._path))

    self._interface = self._interface_cls(self._path)

  @property
  def layer_path(self):
    return self._layer_path

  def get_path_to_file(self, file_path):
    return os.path.join(self._layer_path, file_path)

  def put_json(self, file_path, content, content_type='application/json', *args, **kwargs):
    if type(content) != str:
      content = json.dumps(content)
    return self.put_file(file_path, content, content_type=content_type, *args, **kwargs)
    
  def put_file(self, file_path, content, content_type=None, compress=None, cache_control=None):
    """ 
    Args:
      filename (string): it can contains folders
      content (string): binary data to save
    """
    return self.put_files([ (file_path, content) ], 
      content_type=content_type, 
      compress=compress, 
      cache_control=cache_control, 
      block=False
    )

  def put_files(self, files, content_type=None, compress=None, cache_control=None, block=True):
    """
    Put lots of files at once and get a nice progress bar. It'll also wait
    for the upload to complete, just like get_files.

    Required:
      files: [ (filepath, content), .... ]
    """
    for path, content in files:
      content = compression.compress(content, method=compress)
      self._interface.put_file(path, content, content_type, compress, cache_control=cache_control)
    return self

  def exists(self, file_path):
    """Test if a single file exists. Returns boolean."""
    return self._interface.exists(file_path)

  def files_exist(self, file_paths):
    """
    Threaded exists for all file paths. 

    file_paths: (list) file paths to test for existence

    Returns: { filepath: bool }
    """
    results = {}
    for path in file_paths:
      results[path] = self._interface.exists(path)
    return results

  def get_json(self, file_path):
    content = self.get_file(file_path)
    if content is None:
      return None
    return json.loads(content.decode('utf8'))

  def get_file(self, file_path):
    content, encoding = self._interface.get_file(file_path)
    content = compression.decompress(content, encoding, filename=file_path)
    return content

  def delete_file(self, file_path):
    self._interface.delete_file(file_path)

  def delete_files(self, file_paths):
    for path in file_paths:
      self._interface.delete_file(path)
    return self

  def list_files(self, prefix="", flat=False):
    """
    List the files in the layer with the given prefix. 

    flat means only generate one level of a directory,
    while non-flat means generate all file paths with that 
    prefix.

    Here's how flat=True handles different senarios:
      1. partial directory name prefix = 'bigarr'
        - lists the '' directory and filters on key 'bigarr'
      2. full directory name prefix = 'bigarray'
        - Same as (1), but using key 'bigarray'
      3. full directory name + "/" prefix = 'bigarray/'
        - Lists the 'bigarray' directory
      4. partial file name prefix = 'bigarray/chunk_'
        - Lists the 'bigarray/' directory and filters on 'chunk_'
    
    Return: generated sequence of file paths relative to layer_path
    """

    for f in self._interface.list_files(prefix, flat):
      yield f

  def __del__(self):
    self._interface.release_connection()

  def __enter__(self):
    return self

  def __exit__(self, exception_type, exception_value, traceback):
    self._interface.release_connection()

class Storage(ThreadedQueue):
  """
  Access files stored in Google Storage (gs), Amazon S3 (s3), 
  or the local Filesystem (file).

  e.g. with Storage('gs://bucket/dataset/layer') as stor:
      files = stor.get_file('filename')

  Required:
    layer_path (str): A protocol prefixed path of the above format.
      Accepts s3:// gs:// and file://. File paths are absolute.

  Optional:
    n_threads (int:20): number of threads to use downloading and uplaoding.
      If 0, execution will be on the main python thread.
    progress (bool:false): Show a tqdm progress bar for multiple 
      uploads and downloads.
  """
  def __init__(self, layer_path, n_threads=20, progress=False):

    self.progress = progress

    self._layer_path = layer_path
    self._path = extract_path(layer_path)
    
    if self._path.protocol == 'file':
      self._interface_cls = FileInterface
    elif self._path.protocol == 'gs':
      self._interface_cls = GoogleCloudStorageInterface
    elif self._path.protocol in ('s3', 'matrix'):
      self._interface_cls = S3Interface
    elif self._path.protocol in ('http', 'https'):
      self._interface_cls = HttpInterface
    else:
      raise UnsupportedProtocol(str(self._path))

    self._interface = self._interface_cls(self._path)

    super(Storage, self).__init__(n_threads)

  def _initialize_interface(self):
    return self._interface_cls(self._path)

  def _close_interface(self, interface):
    interface.release_connection()

  def _consume_queue(self, terminate_evt):
    super(Storage, self)._consume_queue(terminate_evt)
    self._interface.release_connection()

  @property
  def layer_path(self):
    return self._layer_path

  def get_path_to_file(self, file_path):
    return os.path.join(self._layer_path, file_path)

  def put_json(self, file_path, content, content_type='application/json', *args, **kwargs):
    if type(content) != str:
      content = json.dumps(content)
    return self.put_file(file_path, content, content_type=content_type, *args, **kwargs)
  
  def put_file(self, file_path, content, content_type=None, compress=None, cache_control=None):
    """ 
    Args:
      filename (string): it can contains folders
      content (string): binary data to save
    """
    return self.put_files([ (file_path, content) ], 
      content_type=content_type, 
      compress=compress, 
      cache_control=cache_control, 
      block=False
    )

  def put_files(self, files, content_type=None, compress=None, cache_control=None, block=True):
    """
    Put lots of files at once and get a nice progress bar. It'll also wait
    for the upload to complete, just like get_files.

    Required:
      files: [ (filepath, content), .... ]
    """
    def base_uploadfn(path, content, interface):
      interface.put_file(path, content, content_type, compress, cache_control=cache_control)

    for path, content in files:
      content = compression.compress(content, method=compress)
      uploadfn = partial(base_uploadfn, path, content)

      if len(self._threads):
        self.put(uploadfn)
      else:
        uploadfn(self._interface)

    if block:
      desc = 'Uploading' if self.progress else None
      self.wait(desc)

    return self

  def exists(self, file_path):
    """Test if a single file exists. Returns boolean."""
    return self._interface.exists(file_path)

  def files_exist(self, file_paths):
    """
    Threaded exists for all file paths. 

    file_paths: (list) file paths to test for existence

    Returns: { filepath: bool }
    """
    results = {}

    def exist_thunk(path, interface):
      results[path] = interface.exists(path)

    for path in file_paths:
      if len(self._threads):
        self.put(partial(exist_thunk, path))
      else:
        exist_thunk(path, self._interface)

    desc = 'Existence Testing' if self.progress else None
    self.wait(desc)

    return results

  def get_json(self, file_path):
    content = self.get_file(file_path)
    if content is None:
      return None
    return json.loads(content.decode('utf8'))

  def get_file(self, file_path):
    content, encoding = self._interface.get_file(file_path)
    content = compression.decompress(content, encoding, filename=file_path)
    return content

  def get_files(self, file_paths):
    """
    returns a list of files faster by using threads
    """

    results = []

    def get_file_thunk(path, interface):
      result = error = None 

      try:
        result = interface.get_file(path)
      except Exception as err:
        error = err
        # important to print immediately because 
        # errors are collected at the end
        print(err) 
      
      content, encoding = result
      content = compression.decompress(content, encoding)

      results.append({
        "filename": path,
        "content": content,
        "error": error,
      })

    for path in file_paths:
      if len(self._threads):
        self.put(partial(get_file_thunk, path))
      else:
        get_file_thunk(path, self._interface)

    desc = 'Downloading' if self.progress else None
    self.wait(desc)

    return results

  def delete_file(self, file_path):

    def thunk_delete(interface):
      interface.delete_file(file_path)

    if len(self._threads):
      self.put(thunk_delete)
    else:
      thunk_delete(self._interface)

    return self

  def delete_files(self, file_paths):

    def thunk_delete(path, interface):
      interface.delete_file(path)

    for path in file_paths:
      if len(self._threads):
        self.put(partial(thunk_delete, path))
      else:
        thunk_delete(path, self._interface)

    desc = 'Deleting' if self.progress else None
    self.wait(desc)

    return self

  def list_files(self, prefix="", flat=False):
    """
    List the files in the layer with the given prefix. 

    flat means only generate one level of a directory,
    while non-flat means generate all file paths with that 
    prefix.

    Here's how flat=True handles different senarios:
      1. partial directory name prefix = 'bigarr'
        - lists the '' directory and filters on key 'bigarr'
      2. full directory name prefix = 'bigarray'
        - Same as (1), but using key 'bigarray'
      3. full directory name + "/" prefix = 'bigarray/'
        - Lists the 'bigarray' directory
      4. partial file name prefix = 'bigarray/chunk_'
        - Lists the 'bigarray/' directory and filters on 'chunk_'
    
    Return: generated sequence of file paths relative to layer_path
    """

    for f in self._interface.list_files(prefix, flat):
      yield f

  def __del__(self):
    super(Storage, self).__del__()
    self._interface.release_connection()

  def __exit__(self, exception_type, exception_value, traceback):
    super(Storage, self).__exit__(exception_type, exception_value, traceback)
    self._interface.release_connection()

class FileInterface(object):
  def __init__(self, path):
    self._path = path

  def get_path_to_file(self, file_path):
    
    clean = filter(None,[
      self._path.bucket, 
      self._path.intermediate_path,                             
      self._path.dataset,
      self._path.layer,
      file_path
    ])

    return os.path.join(*clean)

  def put_file(self, file_path, content, content_type, compress, cache_control=None):
    path = self.get_path_to_file(file_path)
    mkdir(os.path.dirname(path))

    if compress:
      path += '.gz'

    if content \
      and content_type \
      and re.search('json|te?xt', content_type) \
      and type(content) is str:

      content = content.encode('utf-8')

    try:
      with open(path, 'wb') as f:
        f.write(content)
    except IOError as err:
      with open(path, 'wb') as f:
        f.write(content)

  def get_file(self, file_path):
    path = self.get_path_to_file(file_path)

    compressed = os.path.exists(path + '.gz')
      
    if compressed:
      path += '.gz'

    encoding = 'gzip' if compressed else None
    
    try:
      with open(path, 'rb') as f:
        data = f.read()
      return data, encoding
    except IOError:
      return None, encoding

  def exists(self, file_path):
    path = self.get_path_to_file(file_path)
    return os.path.exists(path) or os.path.exists(path + '.gz')

  def delete_file(self, file_path):
    path = self.get_path_to_file(file_path)
    if os.path.exists(path):
      os.remove(path)
    elif os.path.exists(path + '.gz'):
      os.remove(path + '.gz')

  def list_files(self, prefix, flat):
    """
    List the files in the layer with the given prefix. 

    flat means only generate one level of a directory,
    while non-flat means generate all file paths with that 
    prefix.
    """

    layer_path = self.get_path_to_file("")        
    path = os.path.join(layer_path, prefix) + '*'

    filenames = []
    remove = layer_path + '/'

    if flat:
      for file_path in glob(path):
        if not os.path.isfile(file_path):
          continue
        filename = file_path.replace(remove, '')
        filenames.append(filename)
    else:
      subdir = os.path.join(layer_path, os.path.dirname(prefix))
      for root, dirs, files in os.walk(subdir):
        files = [ os.path.join(root, f) for f in files ]
        files = [ f.replace(remove, '') for f in files ]
        files = [ f for f in files if f[:len(prefix)] == prefix ]
        
        for filename in files:
          filenames.append(filename)
    
    def stripgz(fname):
      (base, ext) = os.path.splitext(fname)
      if ext == '.gz':
        return base
      else:
        return fname

    filenames = list(map(stripgz, filenames))
    return _radix_sort(filenames).__iter__()

  def release_connection(self):
    pass


class GoogleCloudStorageInterface(object):
  def __init__(self, path):
    global GC_POOL
    self._path = path
    self._bucket = GC_POOL[path.bucket].get_connection()

  def get_path_to_file(self, file_path):
    clean = filter(None,[
      self._path.intermediate_path,
      self._path.dataset,
      self._path.layer,
      file_path
    ])
    return os.path.join(*clean)

  @retry
  def put_file(self, file_path, content, content_type, compress, cache_control=None):
    key = self.get_path_to_file(file_path)
    blob = self._bucket.blob( key )
    if compress:
      blob.content_encoding = "gzip"
    if cache_control:
      blob.cache_control = cache_control
    blob.upload_from_string(content, content_type)

  @retry
  def get_file(self, file_path):
    key = self.get_path_to_file(file_path)
    blob = self._bucket.get_blob( key )
    if not blob:
      return None, None # content, encoding

    # blob handles the decompression so the encoding is None
    return blob.download_as_string(), None # content, encoding

  def exists(self, file_path):
    key = self.get_path_to_file(file_path)
    blob = self._bucket.get_blob(key)
    return blob is not None

  @retry
  def delete_file(self, file_path):
    key = self.get_path_to_file(file_path)
    
    try:
      self._bucket.delete_blob( key )
    except google.cloud.exceptions.NotFound:
      pass

  def list_files(self, prefix, flat=False):
    """
    List the files in the layer with the given prefix. 

    flat means only generate one level of a directory,
    while non-flat means generate all file paths with that 
    prefix.
    """
    layer_path = self.get_path_to_file("")        
    path = os.path.join(layer_path, prefix)
    for blob in self._bucket.list_blobs(prefix=path):
      filename = blob.name.replace(layer_path + '/', '')
      if not flat and filename[-1] != '/':
        yield filename
      elif flat and '/' not in blob.name.replace(path, ''):
        yield filename

  def release_connection(self):
    global GC_POOL
    GC_POOL[self._path.bucket].release_connection(self._bucket)

class HttpInterface(object):
  def __init__(self, path):
    self._path = path

  def get_path_to_file(self, file_path):
    clean = filter(None, [
      self._path.bucket,
      self._path.intermediate_path,
      self._path.dataset,
      self._path.layer,
      file_path
    ])
    clean = os.path.join(*clean)
    return self._path.protocol + '://' + clean

  # @retry
  def delete_file(self, file_path):
    raise NotImplementedError()

  # @retry
  def put_file(self, file_path, content, content_type, compress, cache_control=None):
    raise NotImplementedError()

  @retry
  def get_file(self, file_path):
    key = self.get_path_to_file(file_path)
    resp = requests.get(key)
    return resp.content, resp.encoding

  def exists(self, file_path):
    key = self.get_path_to_file(file_path)
    resp = requests.get(key, stream=True)
    return resp.ok

  def list_files(self, prefix, flat=False):
    raise NotImplementedError()

  def release_connection(self):
    pass

class S3Interface(object):
  def __init__(self, path):
    global S3_POOL
    self._path = path
    self._conn = S3_POOL[path.protocol][path.bucket].get_connection()

  def get_path_to_file(self, file_path):
    clean = filter(None,[self._path.intermediate_path,
               self._path.dataset,
               self._path.layer,
               file_path])
    return os.path.join(*clean)

  @retry
  def put_file(self, file_path, content, content_type, compress, cache_control=None):
    key = self.get_path_to_file(file_path)

    attrs = {
      'Bucket': self._path.bucket,
      'Body': content,
      'Key': key,
      'ContentType': (content_type or 'application/octet-stream'),
    }

    if compress:
      attrs['ContentEncoding'] = 'gzip'
    if cache_control:
      attrs['CacheControl'] = cache_control

    self._conn.put_object(**attrs)

  @retry
  def get_file(self, file_path):
    """
      There are many types of execptions which can get raised
      from this method. We want to make sure we only return
      None when the file doesn't exist.
    """

    try:
      resp = self._conn.get_object(
        Bucket=self._path.bucket,
        Key=self.get_path_to_file(file_path),
      )

      encoding = ''
      if 'ContentEncoding' in resp:
        encoding = resp['ContentEncoding']

      return resp['Body'].read(), encoding
    except botocore.exceptions.ClientError as err: 
      if err.response['Error']['Code'] == 'NoSuchKey':
        return None, None
      else:
        raise

  def exists(self, file_path):
    exists = True
    try:
      self._conn.head_object(
        Bucket=self._path.bucket,
        Key=self.get_path_to_file(file_path),
      )
    except botocore.exceptions.ClientError as e:
      if e.response['Error']['Code'] == "404":
        exists = False
      else:
        raise
    
    return exists

  @retry
  def delete_file(self, file_path):
    self._conn.delete_object(
      Bucket=self._path.bucket,
      Key=self.get_path_to_file(file_path),
    )

  def list_files(self, prefix, flat=False):
    """
    List the files in the layer with the given prefix. 

    flat means only generate one level of a directory,
    while non-flat means generate all file paths with that 
    prefix.
    """

    layer_path = self.get_path_to_file("")        
    path = os.path.join(layer_path, prefix)

    resp = self._conn.list_objects_v2(
      Bucket=self._path.bucket,
      Prefix=path,
    )

    def iterate(resp):
      if 'Contents' not in resp.keys():
        resp['Contents'] = []

      for item in resp['Contents']:
        key = item['Key']
        filename = key.replace(layer_path + '/', '')
        if not flat and filename[-1] != '/':
          yield filename
        elif flat and '/' not in key.replace(path, ''):
          yield filename


    for filename in iterate(resp):
      yield filename

    while resp['IsTruncated'] and resp['NextContinuationToken']:
      resp = self._conn.list_objects_v2(
        Bucket=self._path.bucket,
        Prefix=path,
        ContinuationToken=resp['NextContinuationToken'],
      )

      for filename in iterate(resp):
        yield filename

  def release_connection(self):
    global S3_POOL
    S3_POOL[self._path.protocol][self._path.bucket].release_connection(self._conn)

def _radix_sort(L, i=0):
  """
  Most significant char radix sort
  """
  if len(L) <= 1: 
    return L
  done_bucket = []
  buckets = [ [] for x in range(255) ]
  for s in L:
    if i >= len(s):
      done_bucket.append(s)
    else:
      buckets[ ord(s[i]) ].append(s)
  buckets = [ _radix_sort(b, i + 1) for b in buckets ]
  return done_bucket + [ b for blist in buckets for b in blist ]
