from functools import partial
import math
import multiprocessing as mp
import concurrent.futures
import os
import signal

import numpy as np
from six.moves import range
from tqdm import tqdm

from . import lib, chunks
from .cacheservice import CacheService
from .lib import ( 
  toabs, colorize, red, yellow, 
  mkdir, clamp, xyzrange, Vec, 
  Bbox, min2, max2, check_bounds, 
  jsonify, generate_slices
)
from .storage import Storage, SimpleStorage, DEFAULT_THREADS, reset_connection_pools
from .threaded_queue import ThreadedQueue
from .volumecutout import VolumeCutout
from . import sharedmemory as shm

class EmptyVolumeException(Exception):
  """Raised upon finding a missing chunk."""
  pass

class EmptyRequestException(Exception):
  """
  Requesting uploading or downloading 
  a bounding box of less than one cubic voxel
  is impossible.
  """
  pass

class AlignmentError(Exception):
  """Signals that an operation requiring chunk alignment was not aligned."""
  pass 

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
    Volume Offset:   {offset}
    Received:        {got} 
    Nearest Aligned: {check}
""")

# Used in sharedmemory to emulate shared memory on 
# OS X using a file, which has that facility but is 
# more limited than on Linux.
fs_lock = mp.Lock()

def parallel_execution(fn, items, parallel, cleanup_shm=None):
  def cleanup(signum, frame):
    if cleanup_shm:
      shm.unlink(cleanup_shm)

  prevsigint = signal.getsignal(signal.SIGINT)
  prevsigterm = signal.getsignal(signal.SIGTERM)

  signal.signal(signal.SIGINT, cleanup)
  signal.signal(signal.SIGTERM, cleanup)

  with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as executor:
    executor.map(fn, items)

  signal.signal(signal.SIGINT, prevsigint)
  signal.signal(signal.SIGTERM, prevsigterm)


def multi_process_download(cv, bufferbbox, caching, cloudpaths):
  global fs_lock
  reset_connection_pools() # otherwise multi-process hangs
  cv.init_submodules(caching)

  array_like, renderbuffer = shm.bbox2array(cv, bufferbbox, lock=fs_lock)
  def process(img3d, bbox):
    shade(renderbuffer, bufferbbox, img3d, bbox)
  download_multiple(cv, cloudpaths, fn=process)
  array_like.close()

def multi_process_cutout(vol, requested_bbox, cloudpaths, parallel, 
  shared_memory_location, output_to_shared_memory):
  global fs_lock

  cloudpaths_by_process = []
  length = int(math.ceil(len(cloudpaths) / float(parallel)) or 1)
  for i in range(0, len(cloudpaths), length):
    cloudpaths_by_process.append(
      cloudpaths[i:i+length]
    )

  provenance = vol.provenance 
  vol.provenance = None
  spd = partial(multi_process_download, vol, requested_bbox, vol.cache.enabled)
  parallel_execution(spd, cloudpaths_by_process, parallel, cleanup_shm=shared_memory_location)
  vol.provenance = provenance

  mmap_handle, renderbuffer = shm.bbox2array(vol, requested_bbox, lock=fs_lock, location=shared_memory_location)
  if not output_to_shared_memory:
    shm.unlink(vol.shared_memory_id)
  shm.track_mmap(mmap_handle)

  return mmap_handle, renderbuffer

def cutout(vol, requested_bbox, steps, channel_slice=slice(None), parallel=1, 
  shared_memory_location=None, output_to_shared_memory=False):
  """Cutout a requested bounding box from storage and return it as a numpy array."""
  global fs_lock

  cloudpath_bbox = requested_bbox.expand_to_chunk_size(vol.underlying, offset=vol.voxel_offset)
  cloudpath_bbox = Bbox.clamp(cloudpath_bbox, vol.bounds)
  cloudpaths = list(chunknames(cloudpath_bbox, vol.bounds, vol.key, vol.underlying))
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
    download_multiple(vol, cloudpaths, fn=process)
  else:
    handle, renderbuffer = multi_process_cutout(vol, requested_bbox, cloudpaths, parallel, 
      shared_memory_location, output_to_shared_memory)
  
  renderbuffer = renderbuffer[ ::steps.x, ::steps.y, ::steps.z, channel_slice ]
  return VolumeCutout.from_volume(vol, renderbuffer, requested_bbox, handle=handle)

def download_single(vol, cloudpath, filename, cache):
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

def download_multiple(vol, cloudpaths, fn):
  locations = vol.cache.compute_data_locations(cloudpaths)
  cachedir = 'file://' + os.path.join(vol.cache.path, vol.key)
  progress = 'Downloading' if vol.progress else None

  def process(cloudpath, filename, cache, iface):
    img3d, bbox = download_single(vol, cloudpath, filename, cache)
    fn(img3d, bbox)

  with ThreadedQueue(n_threads=DEFAULT_THREADS, progress=progress) as tq:
    for filename in locations['local']:
      dl = partial(process, cachedir, filename, False)
      tq.put(dl)
    for filename in locations['remote']:
      dl = partial(process, vol.layer_cloudpath, filename, vol.cache.enabled)
      tq.put(dl)

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

def shade(renderbuffer, bufferbbox, img3d, bbox):
  """Shade a renderbuffer with a downloaded chunk. 
    The buffer will only be painted in the overlapping
    region of the content."""

  if not Bbox.intersects(bufferbbox, bbox):
    return

  spt = max2(bbox.minpt, bufferbbox.minpt)
  ept = min2(bbox.maxpt, bufferbbox.maxpt)

  ZERO3 = Vec(0,0,0)

  istart = max2(spt - bbox.minpt, ZERO3)
  iend = min2(ept - bbox.maxpt, ZERO3) + img3d.shape[:3]

  rbox = Bbox(spt, ept) - bufferbbox.minpt
  if len(img3d.shape) == 3:
    img3d = img3d[ :, :, :, np.newaxis]
  
  renderbuffer[ rbox.to_slices() ] = img3d[ istart.x:iend.x, istart.y:iend.y, istart.z:iend.z, : ]

def content_type(vol):
  if vol.encoding == 'jpeg':
    return 'image/jpeg'
  elif vol.encoding in ('compressed_segmentation', 'fpzip', 'kempressed'):
    return 'image/x.' + vol.encoding 
  return 'application/octet-stream'

def should_compress(vol, iscache=False):
  if iscache and vol.cache.compress != None:
    return vol.cache.compress

  if vol.compress is None:
    return 'gzip' if vol.encoding in ('raw', 'compressed_segmentation') else None
  elif vol.compress == True:
    return 'gzip'
  elif vol.compress == False:
    return None
  else:
    return vol.compress

def cdn_cache_control(val):
  """Translate cdn_cache into a Cache-Control HTTP header."""
  if val is None:
    return 'max-age=3600, s-max-age=3600'
  elif type(val) is str:
    return val
  elif type(val) is bool:
    if val:
      return 'max-age=3600, s-max-age=3600'
    else:
      return 'no-cache'
  elif type(val) is int:
    if val < 0:
      raise ValueError('cdn_cache must be a positive integer, boolean, or string. Got: ' + str(val))

    if val == 0:
      return 'no-cache'
    else:
      return 'max-age={}, s-max-age={}'.format(val, val)
  else:
    raise NotImplementedError(type(val) + ' is not a supported cache_control setting.')

def check_grid_aligned(vol, img, offset):
  """Returns (is_aligned, img bounds Bbox, nearest bbox inflated to grid aligned)"""
  shape = Vec(*img.shape)[:3]
  offset = Vec(*offset)[:3]
  bounds = Bbox( offset, shape + offset)
  alignment_check = bounds.expand_to_chunk_size(vol.underlying, vol.voxel_offset)
  alignment_check = Bbox.clamp(alignment_check, vol.bounds)
  is_aligned = np.all(alignment_check.minpt == bounds.minpt) and np.all(alignment_check.maxpt == bounds.maxpt)
  return (is_aligned, bounds, alignment_check) 

def upload_image(vol, img, offset, parallel=1, 
  manual_shared_memory_id=None, manual_shared_memory_bbox=None, manual_shared_memory_order='F'):
  """Upload img to vol with offset. This is the primary entry point for uploads."""
  global NON_ALIGNED_WRITE

  if str(vol.dtype) != str(img.dtype):
    raise ValueError('The uploaded image data type must match the volume data type. volume: {}, image: {}'.format(vol.dtype, img.dtype))

  (is_aligned, bounds, expanded) = check_grid_aligned(vol, img, offset)

  if is_aligned:
    upload_aligned(vol, img, offset, parallel=parallel, 
      manual_shared_memory_id=manual_shared_memory_id, manual_shared_memory_bbox=manual_shared_memory_bbox,
      manual_shared_memory_order=manual_shared_memory_order)
    return
  elif vol.non_aligned_writes == False:
    msg = NON_ALIGNED_WRITE.format(mip=vol.mip, offset=vol.voxel_offset, got=bounds, check=expanded)
    raise AlignmentError(msg)

  # Upload the aligned core
  retracted = bounds.shrink_to_chunk_size(vol.underlying, vol.voxel_offset)
  core_bbox = retracted.clone() - bounds.minpt
  core_img = img[ core_bbox.to_slices() ] 
  upload_aligned(vol, core_img, retracted.minpt, parallel=parallel, 
    manual_shared_memory_id=manual_shared_memory_id, manual_shared_memory_bbox=manual_shared_memory_bbox,
    manual_shared_memory_order=manual_shared_memory_order)

  # Download the shell, paint, and upload
  all_chunks = set(chunknames(expanded, vol.bounds, vol.key, vol.underlying))
  core_chunks = set(chunknames(retracted, vol.bounds, vol.key, vol.underlying))
  shell_chunks = all_chunks.difference(core_chunks)

  def shade_and_upload(img3d, bbox):
    # decode is returning non-writable chunk
    # we're throwing them away so safe to write
    img3d.setflags(write=1) 
    shade(img3d, bbox, img, bounds)
    single_process_upload(vol, img3d, (( Vec(0,0,0), Vec(*img3d.shape[:3]), bbox.minpt, bbox.maxpt),), n_threads=0)

  download_multiple(vol, shell_chunks, fn=shade_and_upload)

def upload_aligned(vol, img, offset, parallel=1, 
  manual_shared_memory_id=None, manual_shared_memory_bbox=None, manual_shared_memory_order='F'):
  global fs_lock

  chunk_ranges = list(generate_chunks(vol, img, offset))

  if parallel == 1:
    single_process_upload(vol, img, chunk_ranges)
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
    array_like, renderbuffer = shm.ndarray(shape=img.shape, dtype=img.dtype, 
      location=shared_memory_id, order=manual_shared_memory_order, lock=fs_lock)
    renderbuffer[:] = img

  provenance = vol.provenance 
  vol.provenance = None
  mpu = partial(multi_process_upload, vol, img.shape, offset, 
    shared_memory_id, manual_shared_memory_bbox, manual_shared_memory_order, vol.cache.enabled)

  cleanup_shm = shared_memory_id if not manual_shared_memory_id else None
  parallel_execution(mpu, chunk_ranges_by_process, parallel, cleanup_shm=cleanup_shm)
  vol.provenance = provenance

  # If manual mode is enabled, it's the 
  # responsibilty of the user to clean up
  if not manual_shared_memory_id:
    array_like.close()
    shm.unlink(vol.shared_memory_id)

def multi_process_upload(vol, img_shape, offset, 
  shared_memory_id, manual_shared_memory_bbox, manual_shared_memory_order, caching, chunk_ranges):
  global fs_lock
  reset_connection_pools()
  vol.init_submodules(caching)

  shared_shape = img_shape
  if manual_shared_memory_bbox:
    shared_shape = list(manual_shared_memory_bbox.size3()) + [ vol.num_channels ]

  array_like, renderbuffer = shm.ndarray(shape=shared_shape, dtype=vol.dtype, 
      location=shared_memory_id, order=manual_shared_memory_order, lock=fs_lock, readonly=True)

  if manual_shared_memory_bbox:
    cutout_bbox = Bbox( offset, offset + img_shape[:3] )
    delta_box = cutout_bbox.clone() - manual_shared_memory_bbox.minpt
    renderbuffer = renderbuffer[ delta_box.to_slices() ]

  single_process_upload(vol, renderbuffer, chunk_ranges)
  array_like.close()

def single_process_upload(vol, img, chunk_ranges, n_threads=DEFAULT_THREADS):
  if vol.cache.enabled:
    mkdir(vol.cache.path)
    if vol.progress:
      print("Caching upload...")
    cachestorage = Storage('file://' + vol.cache.path, progress=vol.progress, n_threads=n_threads)

  cloudstorage = Storage(vol.layer_cloudpath, progress=vol.progress, n_threads=n_threads)
  iterator = tqdm(chunk_ranges, desc='Rechunking image', disable=(not vol.progress))

  if len(img.shape) == 3:
    img = img[:, :, :, np.newaxis ]

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

  desc = 'Uploading' if vol.progress else None
  cloudstorage.wait(desc)
  cloudstorage.kill_threads()
  
  if vol.cache.enabled:
    desc = 'Caching' if vol.progress else None
    cachestorage.wait(desc)
    cachestorage.kill_threads()


def generate_chunks(vol, img, offset):
  shape = Vec(*img.shape)[:3]
  offset = Vec(*offset)[:3]

  bounds = Bbox( offset, shape + offset)

  alignment_check = bounds.round_to_chunk_size(vol.underlying, vol.voxel_offset)

  if not np.all(alignment_check.minpt == bounds.minpt):
    raise ValueError('Only chunk aligned writes are currently supported. Got: {}, Volume Offset: {}, Alignment Check: {}'.format(
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

def chunknames(bbox, volume_bbox, key, chunk_size):
  for x,y,z in xyzrange( bbox.minpt, bbox.maxpt, chunk_size ):
    highpt = min2(Vec(x,y,z) + chunk_size, volume_bbox.maxpt)
    filename = "{}-{}_{}-{}_{}-{}".format(
      x, highpt.x,
      y, highpt.y, 
      z, highpt.z
    )
    yield os.path.join(key, filename)


