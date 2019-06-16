from functools import partial
import itertools
import os

import numpy as np
from six.moves import range
from tqdm import tqdm

from cloudvolume import lib, chunks
from cloudvolume.exceptions import EmptyVolumeException
from cloudvolume.cacheservice import CacheService
from cloudvlume.lib import (  
  mkdir, clamp, xyzrange, Vec, 
  Bbox, min2, max2, check_bounds, 
  jsonify
)

from cloudvolume.scheduler import schedule_jobs
from cloudvolume.storage import SimpleStorage, reset_connection_pools
from cloudvolume.threaded_queue import DEFAULT_THREADS
from cloudvolume.volumecutout import VolumeCutout

import cloudvolume.sharedmemory as shm

from .common import fs_lock, parallel_execution, chunknames, shade

def download(
    vol, requested_bbox, steps, channel_slice=slice(None), 
    parallel=1, 
    shared_memory_location=None, output_to_shared_memory=False
  ):
  """Cutout a requested bounding box from storage and return it as a numpy array."""
  cloudpath_bbox = requested_bbox.expand_to_chunk_size(vol.chunk_size, offset=vol.voxel_offset)
  cloudpath_bbox = Bbox.clamp(cloudpath_bbox, vol.bounds)
  cloudpaths = list(chunknames(cloudpath_bbox, vol.bounds, vol.key, vol.chunk_size))
  shape = list(requested_bbox.size3()) + [ vol.num_channels ]

  handle = None

  if parallel == 1:
    if output_to_shared_memory:
      array_like, renderbuffer = shm.bbox2array(vol, requested_bbox, 
        location=shared_memory_location, lock=fs_lock)
      shm.track_mmap(array_like)
    else:
      renderbuffer = np.zeros(shape=shape, dtype=vol.dtype, order='F')

    def process(img3d, bbox):
      shade(renderbuffer, requested_bbox, img3d, bbox)
    download_chunks_threaded(vol, cloudpaths, fn=process)
  else:
    handle, renderbuffer = multiprocess_download(vol, requested_bbox, cloudpaths, parallel, 
      shared_memory_location, output_to_shared_memory)
  
  renderbuffer = renderbuffer[ ::steps.x, ::steps.y, ::steps.z, channel_slice ]
  return VolumeCutout.from_volume(vol, renderbuffer, requested_bbox, handle=handle)

def multiprocess_download(
    vol, requested_bbox, 
    cloudpaths, parallel, 
    shared_memory_location, output_to_shared_memory
  ):

  cloudpaths_by_process = []
  length = int(math.ceil(len(cloudpaths) / float(parallel)) or 1)
  for i in range(0, len(cloudpaths), length):
    cloudpaths_by_process.append(
      cloudpaths[i:i+length]
    )

  spd = partial(child_process_download, vol, requested_bbox, vol.cache.enabled)
  parallel_execution(spd, cloudpaths_by_process, parallel, cleanup_shm=shared_memory_location)

  mmap_handle, renderbuffer = shm.bbox2array(vol, requested_bbox, lock=fs_lock, location=shared_memory_location)
  if not output_to_shared_memory:
    shm.unlink(vol.shared_memory_id)
  shm.track_mmap(mmap_handle)

  return mmap_handle, renderbuffer

def child_process_download(cv, bufferbbox, caching, cloudpaths):
  reset_connection_pools() # otherwise multi-process hangs
  cv.init_submodules(caching)

  array_like, renderbuffer = shm.bbox2array(cv, bufferbbox, lock=fs_lock)
  def process(img3d, bbox):
    shade(renderbuffer, bufferbbox, img3d, bbox)
  download_chunks_threaded(cv, cloudpaths, fn=process)
  array_like.close()

def download_chunk(vol, cloudpath, filename, cache):
  with SimpleStorage(cloudpath) as stor:
    content = stor.get_file(filename)

  if cache:
    with SimpleStorage('file://' + vol.cache.path) as stor:
      stor.put_file(
        file_path=filename, 
        content=(content or b''), 
        content_type=content_type(vol), 
        compress=should_compress(vol, iscache=True),
      )

  bbox = Bbox.from_filename(filename) # possible off by one error w/ exclusive bounds
  img3d = decode(vol, filename, content)
  return img3d, bbox

def download_chunks_threaded(vol, cloudpaths, fn):
  locations = vol.cache.compute_data_locations(cloudpaths)
  cachedir = 'file://' + os.path.join(vol.cache.path, vol.key)

  def process(cloudpath, filename, cache):
    img3d, bbox = download_chunk(vol, cloudpath, filename, cache)
    fn(img3d, bbox)

  local_downloads = ( partial(process, cachedir, filename, False) for filename in locations['local'] )
  remote_downloads = ( partial(process, vol.layer_cloudpath, filename, vol.cache.enabled) for filename in locations['remote'] )

  downloads = itertools.chain( local_downloads, remote_downloads )

  schedule_jobs(
    fns=downloads, 
    concurrency=DEFAULT_THREADS, 
    progress=('Downloading' if vol.progress else None),
    total=len(cloudpaths),
    green=False # not ready for this yet
  )

def decode(vol, filename, content):
  """Decode content according to settings in a cloudvolume instance."""
  bbox = Bbox.from_filename(filename)
  content_len = len(content) if content is not None else 0

  if not content:
    if vol.fill_missing:
      content = ''
    else:
      raise EmptyVolumeException(filename)

  shape = list(bbox.size3()) + [ vol.num_channels ]

  try:
    return chunks.decode(
      content, 
      encoding=vol.encoding, 
      shape=shape, 
      dtype=vol.dtype, 
      block_size=vol.compressed_segmentation_block_size,
    )
  except Exception as error:
    print(red('File Read Error: {} bytes, {}, {}, errors: {}'.format(
        content_len, bbox, filename, error)))
    raise

