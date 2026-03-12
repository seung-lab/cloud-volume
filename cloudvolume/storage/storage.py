from __future__ import annotations

import queue as Queue
from collections import defaultdict
import itertools
import json
import os.path
import posixpath
import re
from functools import partial
import types
from typing import Any, Iterator, Iterable

from tqdm import tqdm

from cloudvolume import compression
from cloudvolume.exceptions import UnsupportedProtocolError
from cloudvolume.lib import mkdir, scatter, jsonify, duplicates, yellow, red
from cloudvolume.threaded_queue import ThreadedQueue, DEFAULT_THREADS
from cloudvolume.scheduler import schedule_green_jobs

import cloudvolume.paths

from .storage_interfaces import (
  FileInterface, HttpInterface,
  S3Interface, GoogleCloudStorageInterface,
  StorageInterface,
)

def get_interface_class(protocol: str) -> type[StorageInterface]:
  if protocol == 'file':
    return FileInterface
  elif protocol == 'gs':
    return GoogleCloudStorageInterface
  elif protocol in ('s3', 'matrix'):
    return S3Interface
  elif protocol in ('http', 'https'):
    return HttpInterface
  else:
    raise UnsupportedProtocolError(str(self._path))

def default_byte_iterator(starts: Iterable[int | None] | None, ends: Iterable[int | None] | None) -> tuple[Iterator[int | None], Iterator[int | None]]:
  if starts is None:
    starts = itertools.repeat(None)
  if ends is None:
    ends = itertools.repeat(None)
  return iter(starts), iter(ends)

WARNING_PRINTED = False

class StorageBase(object):
  """Abastract base class of Storage with some implementation details."""
  def __init__(self, layer_path: str, progress: bool | str = False) -> None:
    global WARNING_PRINTED
    self.progress: bool | str = progress

    self._layer_path: str = layer_path
    self._path: Any = cloudvolume.paths.extract(layer_path)
    self._interface_cls: type[StorageInterface] = get_interface_class(self._path.protocol)

    if not WARNING_PRINTED:
      print(red(
        "Storage is obsolete and will be removed soon! "
        "Update your code to use CloudFiles instead. "
        "See https://github.com/seung-lab/cloud-files"
      ))
      WARNING_PRINTED = True

  @property
  def layer_path(self) -> str:
    return self._layer_path

  def progress_description(self, prefix: str) -> str | None:
    if isinstance(self.progress, str):
      return prefix + ' ' + self.progress
    else:
      return prefix if self.progress else None

  def get_connection(self) -> StorageInterface:
    return self._interface_cls(self._path)

  def get_path_to_file(self, file_path: str) -> str:
    return posixpath.join(self._layer_path, file_path)

  def put_json(self, file_path: str, content: Any, content_type: str = 'application/json', *args: Any, **kwargs: Any) -> Any:
    if type(content) != str:
      content = jsonify(content)
    return self.put_file(file_path, content, content_type=content_type, *args, **kwargs)

  def get_json(self, file_path: str) -> Any:
    content = self.get_file(file_path)
    if content is None:
      return None
    return json.loads(content.decode('utf8'))

  def put_file(self, file_path: str, content: bytes | str, content_type: str | None = None, compress: str | bool | None = None, compress_level: int | None = None, cache_control: str | None = None) -> Any:
    """
    Args:
      filename (string): it can contains folders
      content (string): binary data to save
    """
    return self.put_files([ (file_path, content) ],
      content_type=content_type,
      compress=compress,
      compress_level=compress_level,
      cache_control=cache_control,
      block=False
    )

  def exists(self, file_path: str) -> bool:
    raise NotImplementedError()

  def files_exist(self, file_paths: list[str]) -> dict[str, bool]:
    raise NotImplementedError()

  def get_file(self, file_path: str) -> bytes | None:
    raise NotImplementedError()

  def get_files(self, file_paths: list[str], starts: Iterable[int | None] | None = None, ends: Iterable[int | None] | None = None) -> list[dict[str, Any]]:
    raise NotImplementedError()

  def delete_file(self, file_path: str) -> Any:
    raise NotImplementedError()

  def delete_files(self, file_paths: list[str]) -> Any:
    raise NotImplementedError()

  def list_files(self, prefix: str = "", flat: bool = False) -> Iterator[str]:
    raise NotImplementedError()

  def __del__(self) -> None:
    pass

  def __enter__(self) -> StorageBase:
    return self

  def __exit__(self, exception_type: Any, exception_value: Any, traceback: Any) -> None:
    pass

class SimpleStorage(StorageBase):
  """
  Access files stored in Google Storage (gs), Amazon S3 (s3),
  or the local Filesystem (file).

  e.g. with Storage('gs://bucket/dataset/layer') as stor:
      files = stor.get_file('filename')

  Required:
    layer_path (str): A protocol prefixed path of the above format.
      Accepts s3:// gs:// and file://. File paths are absolute.

  Optional:
    progress (bool:false): Show a tqdm progress bar for multiple
      uploads and downloads.
  """
  def __init__(self, layer_path: str, progress: bool | str = False) -> None:
    super(SimpleStorage, self).__init__(layer_path, progress)
    self._interface: StorageInterface = self.get_connection()

  def put_files(self, files: Iterable[tuple[str, bytes | str]], content_type: str | None = None, compress: str | bool | None = None, compress_level: int | None = None, cache_control: str | None = None, block: bool = True) -> SimpleStorage:
    """
    Put lots of files at once and get a nice progress bar. It'll also wait
    for the upload to complete, just like get_files.

    Required:
      files: [ (filepath, content), .... ]
    """
    desc = self.progress_description('Uploading')
    for path, content in tqdm(files, disable=(not self.progress), desc=desc):
      content = compression.compress(content, method=compress, compress_level=compress_level)
      self._interface.put_file(path, content, content_type, compress, cache_control=cache_control)
    return self

  def exists(self, file_path: str) -> bool:
    """Test if a single file exists. Returns boolean."""
    return self._interface.exists(file_path)

  def files_exist(self, file_paths: list[str]) -> dict[str, bool]:
    """
    Threaded exists for all file paths.

    file_paths: (list) file paths to test for existence

    Returns: { filepath: bool }
    """
    return self._interface.files_exist(file_paths)

  def get_file(self, file_path: str, start: int | None = None, end: int | None = None) -> bytes | None:
    """
    Get the binary contents of a file. Optionally, specify
    the inclusive byte range to request.
    """
    content, encoding = self._interface.get_file(file_path, start=start, end=end)
    content = compression.decompress(content, encoding, filename=file_path)
    return content

  def get_files(self, file_paths: list[str], starts: Iterable[int | None] | None = None, ends: Iterable[int | None] | None = None) -> list[dict[str, Any]]:
    starts, ends = default_byte_iterator(starts, ends)

    iterator = tqdm(
      zip(file_paths, starts, ends),
      disable=(not self.progress),
      desc=self.progress_description("Downloading")
    )

    results: list[dict[str, Any]] = []
    for path, start, end in iterator:
      error = None

      try:
        content = self.get_file(path, start, end)
      except Exception as err:
        error = err
        content = None

      results.append({
        'filename': path,
        'byte_range': (start, end),
        'content': content,
        'error': error,
      })

    return results

  def delete_file(self, file_path: str) -> None:
    self._interface.delete_file(file_path)

  def delete_files(self, file_paths: list[str]) -> SimpleStorage:
    for path in file_paths:
      self._interface.delete_file(path)
    return self

  def list_files(self, prefix: str = "", flat: bool = False) -> Iterator[str]:
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

  def __del__(self) -> None:
    self._interface.release_connection()

  def __enter__(self) -> SimpleStorage:
    return self

  def __exit__(self, exception_type: Any, exception_value: Any, traceback: Any) -> None:
    self._interface.release_connection()

  def __getitem__(self, key: str) -> bytes | None:
    return self.get_file(key)

  def __setitem__(self, key: str | tuple[str, dict[str, Any]], value: Any) -> None:
    if type(key) == tuple:
      key, kwargs = key
    else:
      kwargs = {}

    self.put_item(key, value, **kwargs)

class GreenStorage(StorageBase):
  def __init__(
    self, layer_path: str, progress: bool | str = False,
    n_threads: int = DEFAULT_THREADS,
  ) -> None:
    import gevent.pool
    super(GreenStorage, self).__init__(layer_path, progress)
    self.concurrency: int = n_threads if n_threads > 0 else 1
    self.pool: Any = gevent.pool.Pool(self.concurrency)

  def exists(self, file_path: str) -> bool:
    """Test if a single file exists. Returns boolean."""
    with self.get_connection() as conn:
      return conn.exists(file_path)

  def files_exist(self, file_paths: list[str]) -> dict[str, bool]:
    """
    Threaded exists for all file paths.

    file_paths: (list) file paths to test for existence

    Returns: { filepath: bool }
    """
    results: dict[str, bool] = {}

    def exist_thunk(paths: list[str]) -> None:
      with self.get_connection() as conn:
        results.update(conn.files_exist(paths))

    desc = self.progress_description('Existence Testing')
    schedule_green_jobs(
      fns=( partial(exist_thunk, paths) for paths in scatter(file_paths, self.concurrency) ),
      progress=(desc if self.progress else None),
      concurrency=self.concurrency,
      total=len(file_paths),
    )

    return results

  def get_file(self, file_path: str, start: int | None = None, end: int | None = None) -> bytes | None:
    """
    Get the binary contents of a file. Optionally, specify
    the inclusive byte range to request.
    """
    with self.get_connection() as conn:
      content, encoding = conn.get_file(file_path, start=start, end=end)
    return compression.decompress(content, encoding, filename=file_path)

  def get_files(self, file_paths: list[str], starts: Iterable[int | None] | None = None, ends: Iterable[int | None] | None = None) -> list[dict[str, Any]]:
    """
    Returns: [
      { "filename": ..., "content": bytes, "error": exception or None },
      ...
    ]
    """
    starts, ends = default_byte_iterator(starts, ends)

    def getfn(path: str, start: int | None, end: int | None) -> dict[str, Any]:
      result = error = None

      conn = self.get_connection()
      try:
        result = conn.get_file(path, start=start, end=end)
      except Exception as err:
        error = err
        # important to print immediately because
        # errors are collected at the end
        print(err)
        del conn
      else:
        conn.release_connection()

      content, encoding = result
      content = compression.decompress(content, encoding)

      return {
        "filename": path,
        "byte_range": (start, end),
        "content": content,
        "error": error,
      }

    desc = self.progress_description('Downloading')

    return schedule_green_jobs(
      fns=(
        partial(getfn, path, start, end)
        for path, start, end in zip(file_paths, starts, ends)
      ),
      progress=(desc if self.progress else None),
      concurrency=self.concurrency,
      total=len(file_paths),
    )

  def put_files(
    self, files: Iterable[tuple[str, bytes | str]],
    content_type: str | None = None, compress: str | bool | None = None, compress_level: int | None = None,
    cache_control: str | None = None, block: bool = True
  ) -> GreenStorage:
    """
    Put lots of files at once and get a nice progress bar. It'll also wait
    for the upload to complete, just like get_files.

    Required:
      files: [ (filepath, content), .... ]
    """
    if compress not in compression.COMPRESSION_TYPES:
      raise NotImplementedError()

    def uploadfn(path: str, content: bytes | str) -> None:
      with self.get_connection() as conn:
        content = compression.compress(content, method=compress, compress_level=compress_level)
        conn.put_file(
          file_path=path,
          content=content,
          content_type=content_type,
          compress=compress,
          cache_control=cache_control,
        )

    if not isinstance(gen, types.GeneratorType):
      dupes = duplicates([ path for path, content in files ])
      if dupes:
        raise ValueError("Cannot write the same file multiple times in one pass. This causes a race condition. Files: " + ", ".join(dupes))

    fns = ( partial(uploadfn, path, content) for path, content in files )

    if block:
      desc = desc = self.progress_description('Uploading')
      schedule_green_jobs(
        fns=fns,
        progress=(desc if self.progress else None),
        concurrency=self.concurrency,
        total=len(files),
      )
    else:
      for fn in fns:
        self.pool.spawn(fn)

    return self

  def wait(self, desc: str | None = None) -> None:
    self.pool.join()

  def start_threads(self) -> None:
    import gevent.pool
    self.pool.kill()
    self.pool = gevent.pool.Pool(self.concurrency)

  def kill_threads(self) -> None:
    self.pool.kill()

  def delete_file(self, file_path: str) -> GreenStorage:
    with self.get_connection() as conn:
      conn.delete_file(file_path)
    return self

  def delete_files(self, file_paths: list[str]) -> GreenStorage:
    def thunk_delete(path: str) -> None:
      with self.get_connection() as conn:
        conn.delete_file(path)

    desc = self.progress_description('Deleting')

    schedule_green_jobs(
      fns=( partial(thunk_delete, path) for path in file_paths ),
      progress=(desc if self.progress else None),
      concurrency=self.concurrency,
      total=len(file_paths),
    )

    return self

  def list_files(self, prefix: str = "", flat: bool = False) -> Iterator[str]:
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
    with self.get_connection() as conn:
      for f in conn.list_files(prefix, flat):
        yield f

  def __enter__(self) -> GreenStorage:
    StorageBase.__enter__(self)
    self.start_threads()
    return self

  def __exit__(self, exception_type: Any, exception_value: Any, traceback: Any) -> None:
    StorageBase.__exit__(self, exception_type, exception_value, traceback)
    self.pool.join()
    self.kill_threads()

class ThreadedStorage(StorageBase, ThreadedQueue):
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
  def __init__(self, layer_path: str, n_threads: int = 20, progress: bool | str = False) -> None:
    StorageBase.__init__(self, layer_path, progress)
    ThreadedQueue.__init__(self, n_threads)
    self._interface: StorageInterface = self.get_connection()

  def _initialize_interface(self) -> StorageInterface:
    return self._interface_cls(self._path)

  def _close_interface(self, interface: StorageInterface) -> None:
    interface.release_connection()

  def _consume_queue(self, terminate_evt: Any) -> None:
    ThreadedQueue._consume_queue(self, terminate_evt)
    self._interface.release_connection()

  @property
  def layer_path(self) -> str:
    return self._layer_path

  def get_path_to_file(self, file_path: str) -> str:
    return posixpath.join(self._layer_path, file_path)

  def put_json(self, file_path: str, content: Any, content_type: str = 'application/json', *args: Any, **kwargs: Any) -> Any:
    if type(content) != str:
      content = jsonify(content)
    return self.put_file(file_path, content, content_type=content_type, *args, **kwargs)

  def put_file(self, file_path: str, content: bytes | str, content_type: str | None = None, compress: str | bool | None = None, compress_level: int | None = None, cache_control: str | None = None) -> Any:
    """
    Args:
      filename (string): it can contains folders
      content (string): binary data to save
    """
    return self.put_files([ (file_path, content) ],
      content_type=content_type,
      compress=compress,
      compress_level=compress_level,
      cache_control=cache_control,
      block=False
    )

  def put_files(self, files: Iterable[tuple[str, bytes | str]], content_type: str | None = None, compress: str | bool | None = None, compress_level: int | None = None, cache_control: str | None = None, block: bool = True) -> ThreadedStorage:
    """
    Put lots of files at once and get a nice progress bar. It'll also wait
    for the upload to complete, just like get_files.

    Required:
      files: [ (filepath, content), .... ]
    """
    def base_uploadfn(path: str, content: bytes | str, interface: StorageInterface) -> None:
      interface.put_file(path, content, content_type, compress, cache_control=cache_control)

    seen: set[str] = set()
    for path, content in files:
      if path in seen:
        raise ValueError("Cannot write the same file multiple times in one pass. This causes a race condition. File: " + path)
      seen.add(path)

      content = compression.compress(content, method=compress, compress_level=compress_level)
      uploadfn = partial(base_uploadfn, path, content)

      if len(self._threads):
        self.put(uploadfn)
      else:
        uploadfn(self._interface)

    if block:
      desc = self.progress_description('Uploading')
      self.wait(desc)

    return self

  def exists(self, file_path: str) -> bool:
    """Test if a single file exists. Returns boolean."""
    return self._interface.exists(file_path)

  def files_exist(self, file_paths: list[str]) -> dict[str, bool]:
    """
    Threaded exists for all file paths.

    file_paths: (list) file paths to test for existence

    Returns: { filepath: bool }
    """
    results: dict[str, bool] = {}

    def exist_thunk(paths: list[str], interface: StorageInterface) -> None:
      results.update(interface.files_exist(paths))

    if len(self._threads):
      for block in scatter(file_paths, len(self._threads)):
        self.put(partial(exist_thunk, block))
    else:
      exist_thunk(file_paths, self._interface)

    desc = self.progress_description('Existence Testing')
    self.wait(desc)

    return results

  def get_file(self, file_path: str, start: int | None = None, end: int | None = None) -> bytes | None:
    """
    Get the binary contents of a file. Optionally, specify
    the inclusive byte range to request.
    """
    content, encoding = self._interface.get_file(file_path, start=start, end=end)
    content = compression.decompress(content, encoding, filename=file_path)
    return content

  def get_files(self, file_paths: list[str], starts: Iterable[int | None] | None = None, ends: Iterable[int | None] | None = None) -> list[dict[str, Any]]:
    """
    returns a list of files faster by using threads
    """
    results: list[dict[str, Any]] = []
    starts, ends = default_byte_iterator(starts, ends)

    def get_file_thunk(path: str, start: int | None, end: int | None, interface: StorageInterface) -> None:
      result = error = None

      try:
        result = interface.get_file(path, start=start, end=end)
      except Exception as err:
        error = err
        # important to print immediately because
        # errors are collected at the end
        print(err)

      content, encoding = result
      content = compression.decompress(content, encoding)

      results.append({
        "filename": path,
        "byte_range": (start, end),
        "content": content,
        "error": error,
      })

    for path, start, end in zip(file_paths, starts, ends):
      if len(self._threads):
        self.put(partial(get_file_thunk, path, start, end))
      else:
        get_file_thunk(path, start, end, self._interface)

    desc = self.progress_description('Downloading')
    self.wait(desc)

    return results

  def delete_file(self, file_path: str) -> ThreadedStorage:

    def thunk_delete(interface: StorageInterface) -> None:
      interface.delete_file(file_path)

    if len(self._threads):
      self.put(thunk_delete)
    else:
      thunk_delete(self._interface)

    return self

  def delete_files(self, file_paths: list[str]) -> ThreadedStorage:

    def thunk_delete(path: str, interface: StorageInterface) -> None:
      interface.delete_file(path)

    for path in file_paths:
      if len(self._threads):
        self.put(partial(thunk_delete, path))
      else:
        thunk_delete(path, self._interface)

    desc = self.progress_description('Deleting')
    self.wait(desc)

    return self

  def list_files(self, prefix: str = "", flat: bool = False) -> Iterator[str]:
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

  def __del__(self) -> None:
    ThreadedQueue.__del__(self)
    self._interface.release_connection()

  def __exit__(self, exception_type: Any, exception_value: Any, traceback: Any) -> None:
    ThreadedQueue.__exit__(self, exception_type, exception_value, traceback)
    self._interface.release_connection()
