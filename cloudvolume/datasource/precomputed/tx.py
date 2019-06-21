from functools import partial
import os

import numpy as np
from six.moves import range
from tqdm import tqdm

from cloudvolume import lib, chunks
from cloudvolume.exceptions import AlignmentError
from cloudvolume.cacheservice import CacheService
from cloudvolume.lib import ( 
  mkdir, clamp, xyzrange, Vec, 
  Bbox, min2, max2,
)
from cloudvolume.scheduler import schedule_jobs
from cloudvolume.storage import Storage, SimpleStorage, reset_connection_pools
from cloudvolume.threaded_queue import ThreadedQueue, DEFAULT_THREADS
from cloudvolume.volumecutout import VolumeCutout

import cloudvolume.sharedmemory as shm

from .common import fs_lock, parallel_execution, chunknames, shade

NON_ALIGNED_WRITE = yellow(
  """
  Non-Aligned writes are disabled by default. There are several good reasons 
  not to use them. 

  1) Memory and Network Inefficiency
    Producing non-aligned writes requires downloading the chunks that overlap
    with the write area but are not wholly contained. They are then painted
    in the overlap region and added to the upload queue. This requires
    the outer shell of the aligned image to be downloaded painted, and uploaded. 
  2) Race Conditions
    If you are using multiple instances of CloudVolume, the partially overlapped 
    chunks will be downloaded, partially painted, and uploaded. If this procedure
    occurs close in time between two processes, the final chunk may be partially
    painted. If you are sure only one CloudVolume instance will be accessing an
    area at any one time, this is not a problem.

  If after reading this you are still sure you want non-aligned writes, you can
  set non_aligned_writes=True.

  Alignment Check: 
    Mip:             {mip}
    Chunk Size:      {chunk_size}
    Volume Offset:   {offset}
    Received:        {got} 
    Nearest Aligned: {check}
""")

def upload(
    vol, img, offset, 
    parallel=1, 
    manual_shared_memory_id=None, 
    manual_shared_memory_bbox=None, 
    manual_shared_memory_order='F', 
    delete_black_uploads=False
  ):
  """Upload img to vol with offset. This is the primary entry point for uploads."""
  global NON_ALIGNED_WRITE

  if not np.issubdtype(img.dtype, np.dtype(vol.dtype).type):
    raise ValueError('The uploaded image data type must match the volume data type. volume: {}, image: {}'.format(vol.dtype, img.dtype))

  (is_aligned, bounds, expanded) = check_grid_aligned(vol, img, offset)

  if is_aligned:
    upload_aligned(
      vol, img, offset, 
      parallel=parallel, 
      manual_shared_memory_id=manual_shared_memory_id, 
      manual_shared_memory_bbox=manual_shared_memory_bbox,
      manual_shared_memory_order=manual_shared_memory_order,
      delete_black_uploads=delete_black_uploads
    )
    return
  elif vol.non_aligned_writes == False:
    msg = NON_ALIGNED_WRITE.format(mip=vol.mip, chunk_size=vol.chunk_size, offset=vol.voxel_offset, got=bounds, check=expanded)
    raise AlignmentError(msg)

  # Upload the aligned core
  retracted = bounds.shrink_to_chunk_size(vol.underlying, vol.voxel_offset)
  core_bbox = retracted.clone() - bounds.minpt

  if not core_bbox.subvoxel():
    core_img = img[ core_bbox.to_slices() ] 
    upload_aligned(
      vol, core_img, retracted.minpt, 
      parallel=parallel, 
      manual_shared_memory_id=manual_shared_memory_id, 
      manual_shared_memory_bbox=manual_shared_memory_bbox,
      manual_shared_memory_order=manual_shared_memory_order, 
      delete_black_uploads=delete_black_uploads,
    )

  # Download the shell, paint, and upload
  all_chunks = set(chunknames(expanded, vol.bounds, vol.key, vol.underlying))
  core_chunks = set(chunknames(retracted, vol.bounds, vol.key, vol.underlying))
  shell_chunks = all_chunks.difference(core_chunks)

  def shade_and_upload(img3d, bbox):
    # decode is returning non-writable chunk
    # we're throwing them away so safe to write
    img3d.setflags(write=1) 
    shade(img3d, bbox, img, bounds)
    threaded_upload_chunks(
      vol, img3d, 
      (( Vec(0,0,0), Vec(*img3d.shape[:3]), bbox.minpt, bbox.maxpt),), 
      n_threads=0,
      delete_black_uploads=delete_black_uploads,
    )

  download_multiple(vol, shell_chunks, fn=shade_and_upload)

def upload_aligned(
    vol, img, offset, 
    parallel=1, 
    manual_shared_memory_id=None, 
    manual_shared_memory_bbox=None, 
    manual_shared_memory_order='F', 
    delete_black_uploads=False
  ):
  global fs_lock

  chunk_ranges = list(generate_chunks(vol, img, offset))

  if parallel == 1:
    threaded_upload_chunks(
      vol, img, chunk_ranges, 
      delete_black_uploads=delete_black_uploads
    )
    return

  length = (len(chunk_ranges) // parallel) or 1
  chunk_ranges_by_process = []
  for i in range(0, len(chunk_ranges), length):
    chunk_ranges_by_process.append(
      chunk_ranges[i:i+length]
    )

  if manual_shared_memory_id:
    shared_memory_id = manual_shared_memory_id
  else:
    shared_memory_id = vol.shared_memory_id
    array_like, renderbuffer = shm.ndarray(
      shape=img.shape, dtype=img.dtype, 
      location=shared_memory_id, order=manual_shared_memory_order, 
      lock=fs_lock
    )
    renderbuffer[:] = img

  mpu = partial(multi_process_upload, 
    vol, img.shape, offset, 
    shared_memory_id, 
    manual_shared_memory_bbox, 
    manual_shared_memory_order, 
    vol.cache.enabled, 
    delete_black_uploads
  )

  cleanup_shm = shared_memory_id if not manual_shared_memory_id else None
  parallel_execution(mpu, chunk_ranges_by_process, parallel, cleanup_shm=cleanup_shm)

  # If manual mode is enabled, it's the 
  # responsibilty of the user to clean up
  if not manual_shared_memory_id:
    array_like.close()
    shm.unlink(vol.shared_memory_id)

def multi_process_upload(
    vol, img_shape, offset, 
    shared_memory_id, 
    manual_shared_memory_bbox, 
    manual_shared_memory_order, 
    caching, 
    delete_black_uploads,
    chunk_ranges
  ):
  global fs_lock
  reset_connection_pools()
  vol.init_submodules(caching)

  shared_shape = img_shape
  if manual_shared_memory_bbox:
    shared_shape = list(manual_shared_memory_bbox.size3()) + [ vol.num_channels ]

  array_like, renderbuffer = shm.ndarray(
    shape=shared_shape, 
    dtype=vol.dtype, 
    location=shared_memory_id, 
    order=manual_shared_memory_order, 
    lock=fs_lock, 
    readonly=True
  )

  if manual_shared_memory_bbox:
    cutout_bbox = Bbox( offset, offset + img_shape[:3] )
    delta_box = cutout_bbox.clone() - manual_shared_memory_bbox.minpt
    renderbuffer = renderbuffer[ delta_box.to_slices() ]

  threaded_upload_chunks(
    vol, renderbuffer, chunk_ranges, 
    delete_black_uploads=delete_black_uploads
  )
  array_like.close()

def threaded_upload_chunks(
    vol, img, chunk_ranges, 
    n_threads=DEFAULT_THREADS,
    delete_black_uploads=False
  ):
  if vol.cache.enabled:
    mkdir(vol.cache.path)
    if vol.progress:
      print("Caching upload...")
    cachestorage = Storage('file://' + vol.cache.path, progress=vol.progress, n_threads=n_threads)

  cloudstorage = Storage(vol.layer_cloudpath, progress=vol.progress, n_threads=n_threads)
  iterator = tqdm(chunk_ranges, desc='Rechunking image', disable=(not vol.progress))

  while img.ndim < 4:
    img = img[ ..., np.newaxis ]

  def do_upload(imgchunk, cloudpath):
    encoded = chunks.encode(imgchunk, vol.encoding, vol.compressed_segmentation_block_size)

    cloudstorage.put_file(
      file_path=cloudpath, 
      content=encoded,
      content_type=content_type(vol), 
      compress=should_compress(vol),
      cache_control=cdn_cache_control(vol.cdn_cache),
    )

    if vol.cache.enabled:
      cachestorage.put_file(
        file_path=cloudpath,
        content=encoded, 
        content_type=content_type(vol), 
        compress=should_compress(vol, iscache=True)
      )

  def do_delete(cloudpath):
    cloudstorage.delete_file(cloudpath)
    if vol.cache.enabled:
      cachestorage.delete_file(cloudpath)

  for startpt, endpt, spt, ept in iterator:
    if np.array_equal(spt, ept):
      continue

    imgchunk = img[ startpt.x:endpt.x, startpt.y:endpt.y, startpt.z:endpt.z, : ]

    # handle the edge of the dataset
    clamp_ept = min2(ept, vol.bounds.maxpt)
    newept = clamp_ept - spt
    imgchunk = imgchunk[ :newept.x, :newept.y, :newept.z, : ]

    filename = "{}-{}_{}-{}_{}-{}".format(
      spt.x, clamp_ept.x,
      spt.y, clamp_ept.y, 
      spt.z, clamp_ept.z
    )

    cloudpath = os.path.join(vol.key, filename)

    if delete_black_uploads:
      if np.any(imgchunk):
        do_upload(imgchunk, cloudpath)
      else:
        do_delete(cloudpath)
    else:
      do_upload(imgchunk, cloudpath)

  desc = 'Uploading' if vol.progress else None
  cloudstorage.wait(desc)
  cloudstorage.kill_threads()
  
  if vol.cache.enabled:
    desc = 'Caching' if vol.progress else None
    cachestorage.wait(desc)
    cachestorage.kill_threads()

def check_grid_aligned(vol, img, offset):
  """Returns (is_aligned, img bounds Bbox, nearest bbox inflated to grid aligned)"""
  shape = Vec(*img.shape)[:3]
  offset = Vec(*offset)[:3]
  bounds = Bbox( offset, shape + offset)
  alignment_check = bounds.expand_to_chunk_size(vol.underlying, vol.voxel_offset)
  alignment_check = Bbox.clamp(alignment_check, vol.bounds)
  is_aligned = np.all(alignment_check.minpt == bounds.minpt) and np.all(alignment_check.maxpt == bounds.maxpt)
  return (is_aligned, bounds, alignment_check) 

def generate_chunks(vol, img, offset):
  shape = Vec(*img.shape)[:3]
  offset = Vec(*offset)[:3]

  bounds = Bbox( offset, shape + offset)

  alignment_check = bounds.round_to_chunk_size(vol.underlying, vol.voxel_offset)

  if not np.all(alignment_check.minpt == bounds.minpt):
    raise AlignmentError('Only chunk aligned writes are currently supported. Got: {}, Volume Offset: {}, Alignment Check: {}'.format(
      bounds, vol.voxel_offset, alignment_check)
    )

  bounds = Bbox.clamp(bounds, vol.bounds)

  img_offset = bounds.minpt - offset
  img_end = Vec.clamp(bounds.size3() + img_offset, Vec(0,0,0), shape)

  for startpt in xyzrange( img_offset, img_end, vol.underlying ):
    startpt = startpt.clone()
    endpt = min2(startpt + vol.underlying, shape)
    spt = (startpt + bounds.minpt).astype(int)
    ept = (endpt + bounds.minpt).astype(int)
    yield (startpt, endpt, spt, ept)