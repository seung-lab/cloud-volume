import six
from collections import defaultdict
import json
import os.path
import posixpath
import re

import boto3
import botocore
from glob import glob
import google.cloud.exceptions
from google.cloud.storage import Batch, Client
import requests
import tenacity

from cloudvolume.connectionpools import S3ConnectionPool, GCloudBucketPool
from cloudvolume.lib import mkdir
from cloudvolume.exceptions import UnsupportedCompressionType

COMPRESSION_EXTENSIONS = ('.gz', '.br')

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
  wait=tenacity.wait_random_exponential(0.5, 60.0),
)

class StorageInterface(object):
  def release_connection(self):
    pass
  def __enter__(self):
    return self
  def __exit__(self, exception_type, exception_value, traceback):
    self.release_connection()

class FileInterface(StorageInterface):
  def __init__(self, path):
    super(StorageInterface, self).__init__()
    self._path = path

  def get_path_to_file(self, file_path):
    return os.path.join(
      self._path.basepath, self._path.layer, file_path
    )

  def put_file(
    self, file_path, content, 
    content_type, compress, 
    cache_control=None
  ):
    path = self.get_path_to_file(file_path)
    mkdir(os.path.dirname(path))

    # keep default as gzip
    if compress == "br":
      path += ".br"
    elif compress:
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

  def get_file(self, file_path, start=None, end=None):
    path = self.get_path_to_file(file_path)

    if os.path.exists(path + '.gz'):
      encoding = "gzip"
      path += '.gz'
    elif os.path.exists(path + '.br'):
      encoding = "br"
      path += ".br"
    else:
      encoding = None

    try:
      with open(path, 'rb') as f:
        if start is not None:
          f.seek(start)
        if end is not None:
          start = start if start is not None else 0
          num_bytes = end - start
          data = f.read(num_bytes)
        else:
          data = f.read()
      return data, encoding
    except IOError:
      return None, encoding

  def exists(self, file_path):
    path = self.get_path_to_file(file_path)
    return os.path.exists(path) or any(( os.path.exists(path + ext) for ext in COMPRESSION_EXTENSIONS ))

  def files_exist(self, file_paths):
    return { path: self.exists(path) for path in file_paths }

  def delete_file(self, file_path):
    path = self.get_path_to_file(file_path)
    if os.path.exists(path):
      os.remove(path)
    elif os.path.exists(path + '.gz'):
      os.remove(path + '.gz')
    elif os.path.exists(path + ".br"):
      os.remove(path + ".br")

  def delete_files(self, file_paths):
    for path in file_paths:
      self.delete_file(path)

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

    remove = layer_path
    if len(remove) and remove[-1] != '/':
      remove += '/'

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
    
    def stripext(fname):
      (base, ext) = os.path.splitext(fname)
      if ext in COMPRESSION_EXTENSIONS:
        return base
      else:
        return fname

    filenames = list(map(stripext, filenames))
    return _radix_sort(filenames).__iter__()

class GoogleCloudStorageInterface(StorageInterface):
  def __init__(self, path):
    super(StorageInterface, self).__init__()
    global GC_POOL
    self._path = path
    self._bucket = GC_POOL[path.bucket].get_connection()

  def get_path_to_file(self, file_path):
    return posixpath.join(self._path.no_bucket_basepath, self._path.layer, file_path)

  @retry
  def put_file(self, file_path, content, content_type, compress, cache_control=None):
    key = self.get_path_to_file(file_path)
    blob = self._bucket.blob( key )

    # gcloud disable brotli until content-encoding works
    if compress == "br":
      raise UnsupportedCompressionType("Brotli unsupported on google cloud storage")
    elif compress:
      blob.content_encoding = "gzip"
    if cache_control:
      blob.cache_control = cache_control
    blob.upload_from_string(content, content_type)

  @retry
  def get_file(self, file_path, start=None, end=None):
    key = self.get_path_to_file(file_path)
    blob = self._bucket.blob( key )

    if start is not None:
      start = int(start)
    if end is not None:
      end = int(end - 1)      

    try:
      # blob handles the decompression so the encoding is None
      return blob.download_as_string(start=start, end=end), None # content, encoding
    except google.cloud.exceptions.NotFound as err:
      return None, None

  @retry
  def exists(self, file_path):
    key = self.get_path_to_file(file_path)
    blob = self._bucket.blob(key)
    return blob.exists()

  def files_exist(self, file_paths):
    result = {path: None for path in file_paths}
    MAX_BATCH_SIZE = Batch._MAX_BATCH_SIZE

    for i in range(0, len(file_paths), MAX_BATCH_SIZE):
      # Retrieve current batch of blobs. On Batch __exit__ it will populate all
      # future responses before raising errors about the (likely) missing keys.
      try:
        with self._bucket.client.batch():
          for file_path in file_paths[i:i+MAX_BATCH_SIZE]:
            key = self.get_path_to_file(file_path)
            result[file_path] = self._bucket.get_blob(key)
      except google.cloud.exceptions.NotFound as err:
        pass  # Missing keys are expected

    for file_path, blob in result.items():
      # Blob exists if ``dict``, missing if ``_FutureDict``
      result[file_path] = isinstance(blob._properties, dict)

    return result

  @retry
  def delete_file(self, file_path):
    key = self.get_path_to_file(file_path)
    
    try:
      self._bucket.delete_blob( key )
    except google.cloud.exceptions.NotFound:
      pass

  def delete_files(self, file_paths):
    MAX_BATCH_SIZE = Batch._MAX_BATCH_SIZE

    for i in range(0, len(file_paths), MAX_BATCH_SIZE):
      try:
        with self._bucket.client.batch():
          for file_path in file_paths[i : i + MAX_BATCH_SIZE]:
            key = self.get_path_to_file(file_path)
            self._bucket.delete_blob(key)
      except google.cloud.exceptions.NotFound:
        pass

  @retry
  def list_files(self, prefix, flat=False):
    """
    List the files in the layer with the given prefix. 

    flat means only generate one level of a directory,
    while non-flat means generate all file paths with that 
    prefix.
    """
    layer_path = self.get_path_to_file("")        
    path = posixpath.join(layer_path, prefix)
    for blob in self._bucket.list_blobs(prefix=path):
      filename = blob.name.replace(layer_path, '')
      if not filename:
        continue
      elif not flat and filename[-1] != '/':
        yield filename
      elif flat and '/' not in blob.name.replace(path, ''):
        yield filename

  def release_connection(self):
    global GC_POOL
    GC_POOL[self._path.bucket].release_connection(self._bucket)

class HttpInterface(StorageInterface):
  def __init__(self, path):
    super(StorageInterface, self).__init__()
    self._path = path

  def get_path_to_file(self, file_path):
    path = posixpath.join(
      self._path.basepath, self._path.layer, file_path
    )
    return self._path.protocol + '://' + path

  # @retry
  def delete_file(self, file_path):
    raise NotImplementedError()

  def delete_files(self, file_paths):
    raise NotImplementedError()

  # @retry
  def put_file(self, file_path, content, content_type, compress, cache_control=None):
    raise NotImplementedError()

  @retry
  def get_file(self, file_path, start=None, end=None):
    key = self.get_path_to_file(file_path)

    if start is not None or end is not None:
      start = int(start) if start is not None else ''
      end = int(end - 1) if end is not None else ''
      headers = { "Range": "bytes={}-{}".format(start, end) }
      resp = requests.get(key, headers=headers)
    else:
      resp = requests.get(key)
    if resp.status_code in (404, 403):
      return None, None
    resp.raise_for_status()

    if 'Content-Encoding' not in resp.headers:
      return resp.content, None
    # requests automatically decodes these
    elif resp.headers['Content-Encoding'] in ('', 'gzip', 'deflate', 'br'):
      return resp.content, None
    else:
      return resp.content, resp.headers['Content-Encoding']

  @retry
  def exists(self, file_path):
    key = self.get_path_to_file(file_path)
    resp = requests.get(key, stream=True)
    resp.close()
    return resp.ok

  def files_exist(self, file_paths):
    return {path: self.exists(path) for path in file_paths}

  def list_files(self, prefix, flat=False):
    raise NotImplementedError()

class S3Interface(StorageInterface):
  def __init__(self, path):
    super(StorageInterface, self).__init__()
    global S3_POOL
    self._path = path
    self._conn = S3_POOL[path.protocol][path.bucket].get_connection()

  def get_path_to_file(self, file_path):
    return posixpath.join(self._path.no_bucket_basepath, self._path.layer, file_path)

  @retry
  def put_file(self, file_path, content, content_type, compress, cache_control=None, ACL="bucket-owner-full-control"):
    key = self.get_path_to_file(file_path)

    attrs = {
      'Bucket': self._path.bucket,
      'Body': content,
      'Key': key,
      'ContentType': (content_type or 'application/octet-stream'),
      'ACL': ACL,
    }

    # keep gzip as default
    if compress == "br":
      attrs['ContentEncoding'] = 'br'
    elif compress:
      attrs['ContentEncoding'] = 'gzip'
    if cache_control:
      attrs['CacheControl'] = cache_control

    self._conn.put_object(**attrs)

  @retry
  def get_file(self, file_path, start=None, end=None):
    """
    There are many types of execptions which can get raised
    from this method. We want to make sure we only return
    None when the file doesn't exist.
    """

    kwargs = {}
    if start is not None or end is not None:
      start = int(start) if start is not None else ''
      end = int(end - 1) if end is not None else ''
      kwargs['Range'] = "bytes={}-{}".format(start, end)

    try:
      resp = self._conn.get_object(
        Bucket=self._path.bucket,
        Key=self.get_path_to_file(file_path),
        **kwargs
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

  def files_exist(self, file_paths):
    return {path: self.exists(path) for path in file_paths}

  @retry
  def delete_file(self, file_path):

    # Not necessary to handle 404s here.
    # From the boto3 documentation:

    # delete_object(**kwargs)
    # Removes the null version (if there is one) of an object and inserts a delete marker, 
    # which becomes the latest version of the object. If there isn't a null version, 
    # Amazon S3 does not remove any objects.

    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.delete_object

    self._conn.delete_object(
      Bucket=self._path.bucket,
      Key=self.get_path_to_file(file_path),
    )

  def delete_files(self, file_paths):
    for path in file_paths:
      self.delete_file(path)

  def list_files(self, prefix, flat=False):
    """
    List the files in the layer with the given prefix. 

    flat means only generate one level of a directory,
    while non-flat means generate all file paths with that 
    prefix.
    """

    layer_path = self.get_path_to_file("")        
    path = posixpath.join(layer_path, prefix)

    @retry
    def s3lst(continuation_token=None):
      kwargs = {
        'Bucket': self._path.bucket,
        'Prefix': path,
      }

      if continuation_token:
        kwargs['ContinuationToken'] = continuation_token

      return self._conn.list_objects_v2(**kwargs)

    resp = s3lst()

    def iterate(resp):
      if 'Contents' not in resp.keys():
        resp['Contents'] = []

      for item in resp['Contents']:
        key = item['Key']
        filename = key.replace(layer_path, '')
        if not flat and filename[-1] != '/':
          yield filename
        elif flat and '/' not in key.replace(path, ''):
          yield filename


    for filename in iterate(resp):
      yield filename

    while resp['IsTruncated'] and resp['NextContinuationToken']:
      resp = s3lst(resp['NextContinuationToken'])

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