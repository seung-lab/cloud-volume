from functools import partial
import os

import numpy as np
from six.moves import range
from tqdm import tqdm

from cloudfiles import CloudFiles, reset_connection_pools, compression

from cloudvolume import lib, chunks
from cloudvolume.exceptions import AlignmentError
from cloudvolume.lib import ( 
  mkdir, clamp, xyzrange, Vec, 
  Bbox, min2, max2
)
from cloudvolume.scheduler import schedule_jobs
from cloudvolume.threaded_queue import DEFAULT_THREADS
from cloudvolume.volumecutout import VolumeCutout

import cloudvolume.sharedmemory as shm

from ... import check_grid_aligned
from .common import (
  fs_lock, parallel_execution, chunknames, 
  shade
) 
from ..common import (
  content_type, cdn_cache_control,
  should_compress
)
from .rx import download_chunks_threaded

def upload(
    meta, cache,
    image, offset, mip,
    compress=None,
    compress_level=None,
    cdn_cache=None,
    parallel=1,
    progress=False,
    delete_black_uploads=False, 
    background_color=0,
    non_aligned_writes=False,
    location=None, location_bbox=None, location_order='F',
    use_shared_memory=False, use_file=False,
    green=False, fill_missing=False, secrets=None
  ):
  """Upload img to vol with offset. This is the primary entry point for uploads."""

  if not np.issubdtype(image.dtype, np.dtype(meta.dtype).type):
    raise ValueError("""
      The uploaded image data type must match the volume data type. 

      Volume: {}
      Image: {}
      """.format(meta.dtype, image.dtype)
    )

  shape = Vec(*image.shape)[:3]
  offset = Vec(*offset)[:3]
  bounds = Bbox( offset, shape + offset)

  is_aligned = check_grid_aligned(
    meta, image, bounds, mip, 
    throw_error=(non_aligned_writes == False)
  )

  options = {
    "compress": compress,
    "compress_level": compress_level,
    "cdn_cache": cdn_cache,
    "parallel": parallel, 
    "progress": progress,
    "location": location, 
    "location_bbox": location_bbox,
    "location_order": location_order,
    "use_shared_memory": use_shared_memory,
    "use_file": use_file,
    "delete_black_uploads": delete_black_uploads,
    "background_color": background_color,
    "green": green,
    "secrets": secrets,  
  }

  if is_aligned:
    upload_aligned(
      meta, cache, 
      image, offset, mip,
      **options
    )
    return

  # Upload the aligned core
  expanded = bounds.expand_to_chunk_size(meta.chunk_size(mip), meta.voxel_offset(mip))
  retracted = bounds.shrink_to_chunk_size(meta.chunk_size(mip), meta.voxel_offset(mip))
  core_bbox = retracted.clone() - bounds.minpt

  if not core_bbox.subvoxel():
    core_img = image[ core_bbox.to_slices() ] 
    upload_aligned(
      meta, cache, 
      core_img, retracted.minpt, mip,
      **options
    )

  # Download the shell, paint, and upload
  all_chunks = set(chunknames(expanded, meta.bounds(mip), meta.key(mip), meta.chunk_size(mip)))
  core_chunks = set(chunknames(retracted, meta.bounds(mip), meta.key(mip), meta.chunk_size(mip)))
  shell_chunks = all_chunks.difference(core_chunks)

  def shade_and_upload(img3d, bbox):
    # decode is returning non-writable chunk
    # we're throwing them away so safe to write
    img3d.setflags(write=1) 
    shade(img3d, bbox, image, bounds)
    threaded_upload_chunks(
      meta, cache, 
      img3d, mip,
      (( Vec(0,0,0), Vec(*img3d.shape[:3]), bbox.minpt, bbox.maxpt),), 
      compress=compress, cdn_cache=cdn_cache,
      progress=False, n_threads=0, 
      delete_black_uploads=delete_black_uploads,
      green=green, secrets=secrets
    )

  compress_cache = should_compress(meta.encoding(mip), compress, cache, iscache=True)

  download_chunks_threaded(
    meta, cache, mip, shell_chunks, fn=shade_and_upload,
    fill_missing=fill_missing, 
    progress=("Shading Border" if progress else None), 
    compress_cache=compress_cache,
    green=green, secrets=secrets
  )

def upload_aligned(
    meta, cache,
    img, offset, mip,
    compress=None,
    compress_level=None,
    cdn_cache=None,
    progress=False,
    parallel=1, 
    location=None, 
    location_bbox=None, 
    location_order='F', 
    use_shared_memory=False,
    use_file=False,
    delete_black_uploads=False,
    background_color=0,
    green=False,
    secrets=None,
  ):
  global fs_lock

  chunk_ranges = list(generate_chunks(meta, img, offset, mip))

  if parallel == 1:
    threaded_upload_chunks(
      meta, cache, 
      img, mip, chunk_ranges, 
      progress=progress,
      compress=compress, cdn_cache=cdn_cache,
      delete_black_uploads=delete_black_uploads,
      background_color=background_color,
      green=green, compress_level=compress_level,
      secrets=secrets
    )
    return

  length = (len(chunk_ranges) // parallel) or 1
  chunk_ranges_by_process = []
  for i in range(0, len(chunk_ranges), length):
    chunk_ranges_by_process.append(
      chunk_ranges[i:i+length]
    )

  # use_shared_memory means use a predetermined
  # shared memory location, not no shared memory 
  # at all.
  if not use_shared_memory:
    array_like, renderbuffer = shm.ndarray(
      shape=img.shape, dtype=img.dtype, 
      location=location, order=location_order, 
      lock=fs_lock
    )
    renderbuffer[:] = img

  cup = partial(child_upload_process, 
    meta, cache, 
    img.shape, offset, mip,
    compress, cdn_cache, progress,
    location, location_bbox, location_order, 
    delete_black_uploads, background_color, 
    green, compress_level=compress_level,
    secrets=secrets
  )

  parallel_execution(cup, chunk_ranges_by_process, parallel, cleanup_shm=location)

  # If manual mode is enabled, it's the 
  # responsibilty of the user to clean up
  if not use_shared_memory:
    array_like.close()
    shm.unlink(location)

def child_upload_process(
    meta, cache, 
    img_shape, offset, mip,
    compress, cdn_cache, progress,
    location, location_bbox, location_order, 
    delete_black_uploads, background_color,
    green, chunk_ranges, compress_level=None,
    secrets=None
  ):
  global fs_lock
  reset_connection_pools()

  shared_shape = img_shape
  if location_bbox:
    shared_shape = list(location_bbox.size3()) + [ meta.num_channels ]

  array_like, renderbuffer = shm.ndarray(
    shape=shared_shape, 
    dtype=meta.dtype, 
    location=location, 
    order=location_order, 
    lock=fs_lock, 
    readonly=True
  )

  if location_bbox:
    cutout_bbox = Bbox( offset, offset + img_shape[:3] )
    delta_box = cutout_bbox.clone() - location_bbox.minpt
    renderbuffer = renderbuffer[ delta_box.to_slices() ]

  threaded_upload_chunks(
    meta, cache, 
    renderbuffer, mip, chunk_ranges, 
    compress=compress, cdn_cache=cdn_cache, progress=progress,
    delete_black_uploads=delete_black_uploads, 
    background_color=background_color,
    green=green, compress_level=compress_level,
    secrets=secrets
  )
  array_like.close()

def threaded_upload_chunks(
    meta, cache, 
    img, mip, chunk_ranges, 
    compress, cdn_cache, progress,
    n_threads=DEFAULT_THREADS,
    delete_black_uploads=False,
    background_color=0,
    green=False,
    compress_level=None,
    secrets=None,
  ):
  
  if cache.enabled:
    mkdir(cache.path)

  while img.ndim < 4:
    img = img[ ..., np.newaxis ]

  remote = CloudFiles(meta.cloudpath, secrets=secrets)
  local = CloudFiles('file://' + cache.path, secrets=secrets)

  def do_upload(imgchunk, cloudpath):
    encoded = chunks.encode(imgchunk, meta.encoding(mip), meta.compressed_segmentation_block_size(mip))

    remote_compress = should_compress(meta.encoding(mip), compress, cache)
    cache_compress = should_compress(meta.encoding(mip), compress, cache, iscache=True)
    remote_compress = compression.normalize_encoding(remote_compress)
    cache_compress = compression.normalize_encoding(cache_compress)

    encoded = compression.compress(encoded, remote_compress)
    cache_encoded = encoded
    if remote_compress != cache_compress:
      cache_encoded = compression.compress(encoded, cache_compress)
    
    remote.put(
        path=cloudpath, 
        content=encoded,
        content_type=content_type(meta.encoding(mip)), 
        compress=remote_compress,
        compression_level=compress_level,
        cache_control=cdn_cache_control(cdn_cache),
        raw=True,
      )

    if cache.enabled:
      local.put(
        path=cloudpath,
        content=cache_encoded, 
        content_type=content_type(meta.encoding(mip)), 
        compress=cache_compress,
        raw=True,
      )

  def do_delete(cloudpath):
    remote.delete(cloudpath)
    
    if cache.enabled:
      local.delete(cloudpath)

  def process(startpt, endpt, spt, ept):
    if np.array_equal(spt, ept):
      return

    imgchunk = img[ startpt.x:endpt.x, startpt.y:endpt.y, startpt.z:endpt.z, : ]

    # handle the edge of the dataset
    clamp_ept = min2(ept, meta.bounds(mip).maxpt)
    newept = clamp_ept - spt
    imgchunk = imgchunk[ :newept.x, :newept.y, :newept.z, : ]

    filename = "{}-{}_{}-{}_{}-{}".format(
      spt.x, clamp_ept.x,
      spt.y, clamp_ept.y, 
      spt.z, clamp_ept.z
    )

    cloudpath = meta.join(meta.key(mip), filename)

    if delete_black_uploads:
      if np.any(imgchunk != background_color):
        do_upload(imgchunk, cloudpath)
      else:
        do_delete(cloudpath)
    else:
      do_upload(imgchunk, cloudpath)

  schedule_jobs(
    fns=( partial(process, *vals) for vals in chunk_ranges ), 
    concurrency=n_threads, 
    progress=('Uploading' if progress else None),
    total=len(chunk_ranges),
    green=green,
  )

def generate_chunks(meta, img, offset, mip):
  shape = Vec(*img.shape)[:3]
  offset = Vec(*offset)[:3]

  bounds = Bbox( offset, shape + offset)

  alignment_check = bounds.round_to_chunk_size(meta.chunk_size(mip), meta.voxel_offset(mip))

  if not np.all(alignment_check.minpt == bounds.minpt):
    raise AlignmentError("""
      Only chunk aligned writes are supported by this function. 

      Got:             {}
      Volume Offset:   {} 
      Nearest Aligned: {}
    """.format(
      bounds, meta.voxel_offset(mip), alignment_check)
    )

  bounds = Bbox.clamp(bounds, meta.bounds(mip))

  img_offset = bounds.minpt - offset
  img_end = Vec.clamp(bounds.size3() + img_offset, Vec(0,0,0), shape)

  for startpt in xyzrange( img_offset, img_end, meta.chunk_size(mip) ):
    startpt = startpt.clone()
    endpt = min2(startpt + meta.chunk_size(mip), shape)
    spt = (startpt + bounds.minpt).astype(int)
    ept = (endpt + bounds.minpt).astype(int)
    yield (startpt, endpt, spt, ept)
