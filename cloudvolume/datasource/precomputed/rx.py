from functools import partial
import itertools
import math
import os

import numpy as np
from six.moves import range
from tqdm import tqdm

from ...exceptions import EmptyVolumeException
from ...lib import (  
  mkdir, clamp, xyzrange, Vec, 
  Bbox, min2, max2, check_bounds, 
  jsonify
)
from ... import chunks

from cloudvolume.scheduler import schedule_jobs
from cloudvolume.storage import SimpleStorage, reset_connection_pools
from cloudvolume.threaded_queue import DEFAULT_THREADS
from cloudvolume.volumecutout import VolumeCutout

import cloudvolume.sharedmemory as shm

from .common import (
  fs_lock, parallel_execution, chunknames, shade,
  should_compress, content_type
)

def download(
    requested_bbox, mip, 
    meta, cache,
    fill_missing, progress,
    parallel, location, 
    retain, use_shared_memory, 
    use_file, compress, order='F',
    green=False
  ):
  """Cutout a requested bounding box from storage and return it as a numpy array."""
  
  full_bbox = requested_bbox.expand_to_chunk_size(
    meta.chunk_size(mip), offset=meta.voxel_offset(mip)
  )
  full_bbox = Bbox.clamp(full_bbox, meta.bounds(mip))
  cloudpaths = list(chunknames(
    full_bbox, meta.bounds(mip), 
    meta.key(mip), meta.chunk_size(mip), 
    protocol=meta.path.protocol
  ))
  shape = list(requested_bbox.size3()) + [ meta.num_channels ]

  compress_cache = should_compress(meta.encoding(mip), compress, cache, iscache=True)

  handle = None

  if use_shared_memory and use_file:
    raise ValueError("use_shared_memory and use_file are mutually exclusive arguments.")

  if parallel == 1:
    if use_shared_memory: # write to shared memory
      handle, renderbuffer = shm.ndarray(
        shape, dtype=meta.dtype, order=order,
        location=location, lock=fs_lock
      )
      if not retain:
        shm.unlink(location)
    elif use_file: # write to ordinary file
      handle, renderbuffer = shm.ndarray_fs(
        shape, dtype=meta.dtype, order=order,
        location=location, lock=fs_lock,
        emulate_shm=False
      )
      if not retain:
        os.unlink(location)
    else:
      renderbuffer = np.zeros(shape=shape, dtype=meta.dtype, order=order)

    def process(img3d, bbox):
      shade(renderbuffer, requested_bbox, img3d, bbox)

    download_chunks_threaded(
      meta, cache, mip, cloudpaths, 
      fn=process, fill_missing=fill_missing,
      progress=progress, compress_cache=compress_cache, 
      green=green 
    )
  else:
    handle, renderbuffer = multiprocess_download(
      requested_bbox, mip, cloudpaths,
      meta, cache, compress_cache,
      fill_missing, progress,
      parallel, location, retain, 
      use_shared_memory=(use_file == False),
      order=order,
      green=green,
    )
  
  return VolumeCutout.from_volume(
    meta, mip, renderbuffer, 
    requested_bbox, handle=handle
  )

def multiprocess_download(
    requested_bbox, mip, cloudpaths,
    meta, cache, compress_cache,
    fill_missing, progress,
    parallel, location, 
    retain, use_shared_memory, order,
    green
  ):

  cloudpaths_by_process = []
  length = int(math.ceil(len(cloudpaths) / float(parallel)) or 1)
  for i in range(0, len(cloudpaths), length):
    cloudpaths_by_process.append(
      cloudpaths[i:i+length]
    )

  cpd = partial(child_process_download, 
    meta, cache, mip, compress_cache, 
    requested_bbox, 
    fill_missing, progress,
    location, use_shared_memory,
    green
  )
  parallel_execution(cpd, cloudpaths_by_process, parallel, cleanup_shm=location)

  shape = list(requested_bbox.size3()) + [ meta.num_channels ]

  if use_shared_memory:
    mmap_handle, renderbuffer = shm.ndarray(
      shape, dtype=meta.dtype, order=order, 
      location=location, lock=fs_lock
    )
  else:
    handle, renderbuffer = shm.ndarray_fs(
      shape, dtype=meta.dtype, order=order,
      location=location, lock=fs_lock,
      emulate_shm=False
    )    

  if not retain:
    if use_shared_memory:
      shm.unlink(location)
    else:
      os.unlink(location)

  return mmap_handle, renderbuffer

def child_process_download(
    meta, cache, mip, compress_cache, 
    dest_bbox, 
    fill_missing, progress,
    location, use_shared_memory, green,
    cloudpaths
  ):
  reset_connection_pools() # otherwise multi-process hangs

  shape = list(dest_bbox.size3()) + [ meta.num_channels ]

  if use_shared_memory:
    array_like, dest_img = shm.ndarray(
      shape, dtype=meta.dtype, 
      location=location, lock=fs_lock
    )
  else:
    array_like, dest_img = shm.ndarray_fs(
      shape, dtype=meta.dtype, 
      location=location, emulate_shm=False, 
      lock=fs_lock
    )

  def process(src_img, src_bbox):
    shade(dest_img, dest_bbox, src_img, src_bbox)

  download_chunks_threaded(
    meta, cache, mip, cloudpaths,
    fn=process, fill_missing=fill_missing,
    progress=progress, compress_cache=compress_cache,
    green=green
  )

  array_like.close()

def download_chunk(
    meta, cache, 
    cloudpath, mip,
    filename, fill_missing,
    enable_cache, compress_cache
  ):
  with SimpleStorage(cloudpath) as stor:
    content = stor.get_file(filename)

  if enable_cache:
    with SimpleStorage('file://' + cache.path) as stor:
      stor.put_file(
        file_path=filename, 
        content=(content or b''), 
        content_type=content_type(meta.encoding(mip)), 
        compress=compress_cache,
      )

  bbox = Bbox.from_filename(filename) # possible off by one error w/ exclusive bounds
  img3d = decode(meta, filename, content, fill_missing, mip)
  return img3d, bbox

def download_chunks_threaded(
    meta, cache, mip, cloudpaths, fn, 
    fill_missing, progress, compress_cache,
    green=False
  ):
  locations = cache.compute_data_locations(cloudpaths)
  cachedir = 'file://' + os.path.join(cache.path, meta.key(mip))

  def process(cloudpath, filename, enable_cache):
    img3d, bbox = download_chunk(
      meta, cache, cloudpath, mip,
      filename, fill_missing,
      enable_cache, compress_cache
    )
    fn(img3d, bbox)

  local_downloads = ( 
    partial(process, cachedir, os.path.basename(filename), False) for filename in locations['local'] 
  )
  remote_downloads = ( 
    partial(process, meta.cloudpath, filename, cache.enabled) for filename in locations['remote'] 
  )

  downloads = itertools.chain( local_downloads, remote_downloads )

  schedule_jobs(
    fns=downloads, 
    concurrency=DEFAULT_THREADS, 
    progress=('Downloading' if progress else None),
    total=len(cloudpaths),
    green=green,
  )

def decode(meta, filename, content, fill_missing, mip):
  """
  Decode content from bytes into a numpy array using the 
  dataset metadata.

  If fill_missing is True, return a zeroed array if 
  content is empty. Otherwise, raise an EmptyVolumeException
  in that case.

  Returns: ndarray
  """
  bbox = Bbox.from_filename(filename)
  content_len = len(content) if content is not None else 0

  if not content:
    if fill_missing:
      content = ''
    else:
      raise EmptyVolumeException(filename)

  shape = list(bbox.size3()) + [ meta.num_channels ]

  try:
    return chunks.decode(
      content, 
      encoding=meta.encoding(mip), 
      shape=shape, 
      dtype=meta.dtype, 
      block_size=meta.compressed_segmentation_block_size(mip),
    )
  except Exception as error:
    print(red('File Read Error: {} bytes, {}, {}, errors: {}'.format(
        content_len, bbox, filename, error)))
    raise

