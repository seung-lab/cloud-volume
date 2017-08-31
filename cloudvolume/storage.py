from collections import namedtuple
import six
from six import StringIO, BytesIO
from six.moves import queue as Queue
import os.path
import re
from functools import partial

import boto3
import botocore
from glob import glob
import google.cloud.exceptions
from google.cloud.storage import Client
import gzip
import tenacity

from lib import mkdir
from threaded_queue import ThreadedQueue
from connectionpools import S3ConnectionPool, GCloudConnectionPool

S3_POOL = S3ConnectionPool()
GC_POOL = GCloudConnectionPool()

retry = tenacity.retry(
    reraise=True, 
    stop=tenacity.stop_after_attempt(7), 
    wait=tenacity.wait_full_jitter(0.5, 60.0),
)

class Storage(ThreadedQueue):
    """
    Probably rather sooner that later we will have to store datasets in S3.
    The idea is to modify this class constructor to probably take a path of 
    the problem protocol://bucket_name/dataset_name/layer_name where protocol
    can be s3, gs or file.

    file:// would be useful for when the in-memory python datasource uses too much RAM,
    or possible for writing unit tests.

    This should be the only way to interact with files for any of the protocols.
    """
    gzip_magic_numbers = [ 0x1f, 0x8b ]
    path_regex = re.compile(r'^(gs|file|s3)://(/?.*?)/(.*/)?([^//]+)/([^//]+)/?$')
    ExtractedPath = namedtuple('ExtractedPath',
        ['protocol','bucket_name','dataset_path','dataset_name','layer_name'])

    def __init__(self, layer_path='', n_threads=20):
        self._layer_path = layer_path
        self._path = self.extract_path(layer_path)
        
        if self._path.protocol == 'file':
            self._interface_cls = FileInterface
        elif self._path.protocol == 'gs':
            self._interface_cls = GoogleCloudStorageInterface
        elif self._path.protocol == 's3':
            self._interface_cls = S3Interface

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

    @classmethod
    def extract_path(cls, layer_path):
        match = cls.path_regex.match(layer_path)
        if not match:
            return None
        else:
            return cls.ExtractedPath(*match.groups())

    def put_file(self, file_path, content, content_type=None, compress=False):
        """ 
        Args:
            filename (string): it can contains folders
            content (string): binary data to save
        """
        return self.put_files([ (file_path, content) ], content_type, compress, block=False)

    def put_files(self, files, content_type=None, compress=False, block=True):
        """
        Put lots of files at once and get a nice progress bar. It'll also wait
        for the upload to complete, just like get_files.

        Required:
            files: [ (filepath, content), .... ]
        """
        def base_uploadfn(path, content, interface):
            interface.put_file(path, content, content_type, compress)

        for path, content in files:
            if compress:
                content = self._compress(content)

            uploadfn = partial(base_uploadfn, path, content)

            if len(self._threads):
                self.put(uploadfn)
            else:
                uploadfn(self._interface)

        if block:
            self.wait()

        return self

    def exists(self, file_path):
        return self._interface.exists(file_path)

    def get_file(self, file_path):
        # Create get_files does uses threading to speed up downloading

        content, decompress = self._interface.get_file(file_path)
        if content and decompress != False:
            content = self._maybe_uncompress(content)
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
                print(err)
            
            content, decompress = result
            if content and decompress:
                content = self._maybe_uncompress(content)

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

        self.wait()

        return results

    def delete_file(self, file_path):

        def thunk_delete(interface):
            interface.delete_file(file_path)

        if len(self._threads):
            self.put(thunk_delete)
        else:
            thunk_delete(self._interface)

        return self

    def _maybe_uncompress(self, content):
        """ Uncompression is applied if the first to bytes matches with
            the gzip magic numbers. 
            There is once chance in 65536 that a file that is not gzipped will
            be ungzipped. That's why is better to set uncompress to False in
            get file.
        """
        first_two_bytes = [ byte for byte in bytearray(content)[:2] ]
        if first_two_bytes == self.gzip_magic_numbers:
            return self._uncompress(content)
        return content

    @staticmethod
    def _compress(content):
        stringio = BytesIO()
        gzip_obj = gzip.GzipFile(mode='wb', fileobj=stringio)
        gzip_obj.write(content)
        gzip_obj.close()
        return stringio.getvalue()

    @staticmethod
    def _uncompress(content):
        stringio = BytesIO(content)
        with gzip.GzipFile(mode='rb', fileobj=stringio) as gfile:
            return gfile.read()

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
        
        clean = filter(None,[self._path.bucket_name,
                             self._path.dataset_path,
                             self._path.dataset_name,
                             self._path.layer_name,
                             file_path])
        return  os.path.join(*clean)

    def put_file(self, file_path, content, content_type, compress):
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

        try:
            with open(path, 'rb') as f:
                data = f.read()
            return data, compressed
        except IOError:
            return None, False

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
        self._client = GC_POOL.get_connection()
        self._bucket = self._client.get_bucket(self._path.bucket_name)

    def get_path_to_file(self, file_path):
        clean = filter(None,[self._path.dataset_path,
                             self._path.dataset_name,
                             self._path.layer_name,
                             file_path])
        return  os.path.join(*clean)

    @retry
    def put_file(self, file_path, content, content_type, compress):
        key = self.get_path_to_file(file_path)
        blob = self._bucket.blob( key )
        if compress:
            blob.content_encoding = "gzip"
        blob.upload_from_string(content, content_type)

    @retry
    def get_file(self, file_path):
        key = self.get_path_to_file(file_path)
        blob = self._bucket.get_blob( key )
        if not blob:
            return None, False
        # blob handles the decompression in the case
        # it is necessary
        return blob.download_as_string(), False

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
        GC_POOL.release_connection(self._client)

class S3Interface(object):

    def __init__(self, path):
        global S3_POOL
        self._path = path
        self._conn = S3_POOL.get_connection()

    def get_path_to_file(self, file_path):
        clean = filter(None,[self._path.dataset_path,
                             self._path.dataset_name,
                             self._path.layer_name,
                             file_path])
        return  os.path.join(*clean)

    @retry
    def put_file(self, file_path, content, content_type, compress):
        key = self.get_path_to_file(file_path)
        if compress:
            self._conn.put_object(
                Bucket=self._path.bucket_name,
                Body=content,
                Key=key,
                ContentType=(content_type or 'application/octet-stream'),
                ContentEncoding='gzip',
            )
        else:
            self._conn.put_object(
                Bucket=self._path.bucket_name,
                Body=content,
                Key=key,
                ContentType=(content_type or 'application/octet-stream'),
            )
            
    @retry
    def get_file(self, file_path):
        """
            There are many types of execptions which can get raised
            from this method. We want to make sure we only return
            None when the file doesn't exist.
        """

        try:
            resp = self._conn.get_object(
                Bucket=self._path.bucket_name,
                Key=self.get_path_to_file(file_path),
            )

            encoding = ''
            if 'ContentEncoding' in resp:
                encoding = resp['ContentEncoding']

            return resp['Body'].read(), encoding == "gzip"
        except botocore.exceptions.ClientError as err: 
            if err.response['Error']['Code'] == 'NoSuchKey':
                return None, False
            else:
                raise

    def exists(self, file_path):
        exists = True
        try:
            self._conn.head_object(
                Bucket=self._path.bucket_name,
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
            Bucket=self._path.bucket_name,
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
            Bucket=self._path.bucket_name,
            Prefix=path,
        )

        if 'Contents' not in resp.keys():
            resp['Contents'] = []

        for item in resp['Contents']:
            key = item['Key']
            filename = key.replace(layer_path + '/', '')
            if not flat and filename[-1] != '/':
                yield filename
            elif flat and '/' not in key.replace(path, ''):
                yield filename

    def release_connection(self):
        global S3_POOL
        S3_POOL.release_connection(self._conn)

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
