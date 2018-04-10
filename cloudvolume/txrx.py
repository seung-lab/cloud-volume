from functools import partial
import math
import multiprocessing as mp
import os

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
    Volume Offset:   {offset}
    Received:        {got} 
    Nearest Aligned: {check}
""")

def multi_process_download(cv, bufferbbox, caching, cloudpaths):
  reset_connection_pools() # otherwise multi-process hangs
  cv.init_submodules(caching)

  array_like, renderbuffer = shm.bbox2array(cv, bufferbbox)
  def process(img3d, bbox):
    shade(renderbuffer, bufferbbox, img3d, bbox)
  download_multiple(cv, cloudpaths, fn=process)
  array_like.close()

def multi_process_cutout(vol, requested_bbox, cloudpaths, parallel):
  cloudpaths_by_process = []
  length = int(math.ceil(len(cloudpaths) / float(parallel)) or 1)
  for i in range(0, len(cloudpaths), length):
    cloudpaths_by_process.append(
      cloudpaths[i:i+length]
    )

  provenance = vol.provenance 
  vol.provenance = None
  spd = partial(multi_process_download, vol, requested_bbox, vol.cache.enabled)
  pool = mp.Pool(parallel)
  pool.map(spd, cloudpaths_by_process)
  pool.close()
  vol.provenance = provenance
  
  mmap_handle, renderbuffer = shm.bbox2array(vol, requested_bbox)
  if not vol.output_to_shared_memory:
    renderbuffer = np.copy(renderbuffer)
    mmap_handle.close()
    shm.unlink(vol.shared_memory_id)
  else:
    shm.track_mmap(mmap_handle)

  return mmap_handle, renderbuffer

def cutout(vol, requested_bbox, steps, channel_slice=slice(None), parallel=1):
  """Cutout a requested bounding box from storage and return it as a numpy array."""
  cloudpath_bbox = requested_bbox.expand_to_chunk_size(vol.underlying, offset=vol.voxel_offset)
  cloudpath_bbox = Bbox.clamp(cloudpath_bbox, vol.bounds)
  cloudpaths = chunknames(cloudpath_bbox, vol.bounds, vol.key, vol.underlying)
  shape = list(requested_bbox.size3()) + [ vol.num_channels ]

  handle = None

  if parallel == 1:
    if vol.output_to_shared_memory:
      array_like, renderbuffer = shm.bbox2array(vol, requested_bbox)
      shm.track_mmap(array_like)
    else:
      renderbuffer = np.zeros(shape=shape, dtype=vol.dtype)

    def process(img3d, bbox):
      shade(renderbuffer, requested_bbox, img3d, bbox)
    download_multiple(vol, cloudpaths, fn=process)
  else:
    handle, renderbuffer = multi_process_cutout(vol, requested_bbox, cloudpaths, parallel)
  
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
        compress=should_compress(vol),
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
      content, vol.encoding, shape, vol.dtype
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
  return 'application/octet-stream'

def should_compress(vol):
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

def upload_image(vol, img, offset):
  """Upload img to vol with offset. This is the primary entry point for uploads."""
  global NON_ALIGNED_WRITE

  if str(vol.dtype) != str(img.dtype):
    raise ValueError('The uploaded image data type must match the volume data type. volume: {}, image: {}'.format(vol.dtype, img.dtype))

  (is_aligned, bounds, expanded) = check_grid_aligned(vol, img, offset)

  if is_aligned:
    upload_aligned(vol, img, offset)
    return
  elif vol.non_aligned_writes == False:
    msg = NON_ALIGNED_WRITE.format(offset=vol.voxel_offset, got=bounds, check=expanded)
    raise AlignmentError(msg)

  # Upload the aligned core
  retracted = bounds.shrink_to_chunk_size(vol.underlying, vol.voxel_offset)
  core_bbox = retracted.clone() - bounds.minpt
  core_img = img[ core_bbox.to_slices() ] 
  upload_aligned(vol, core_img, retracted.minpt)

  # Download the shell, paint, and upload
  all_chunks = set(chunknames(expanded, vol.bounds, vol.key, vol.underlying))
  core_chunks = set(chunknames(retracted, vol.bounds, vol.key, vol.underlying))
  shell_chunks = all_chunks.difference(core_chunks)

  def shade_and_upload(img3d, bbox):
    # decode is returning non-writable chunk
    # we're throwing them away so safe to write
    img3d.setflags(write=1) 
    shade(img3d, bbox, img, bounds)
    upload_chunks(vol, ((img3d, bbox.minpt, bbox.maxpt),), n_threads=0)

  download_multiple(vol, shell_chunks, fn=shade_and_upload)

def upload_aligned(vol, img, offset):
  iterator = tqdm(generate_chunks(vol, img, offset), desc='Rechunking image', disable=(not vol.progress))
  upload_chunks(vol, iterator)

def upload_chunks(vol, iterator, n_threads=DEFAULT_THREADS):
  if vol.cache.enabled:
    mkdir(vol.cache.path)
    if vol.progress:
      print("Caching upload...")
    cachestorage = Storage('file://' + vol.cache.path, progress=vol.progress, n_threads=n_threads)

  cloudstorage = Storage(vol.layer_cloudpath, progress=vol.progress, n_threads=n_threads)
  for imgchunk, spt, ept in iterator:
    if np.array_equal(spt, ept):
      continue

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
    encoded = chunks.encode(imgchunk, vol.encoding)

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
        compress=should_compress(vol)
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
  bounds = Bbox.clamp(bounds, vol.bounds)

  img_offset = bounds.minpt - offset
  img_end = Vec.clamp(bounds.size3() + img_offset, Vec(0,0,0), shape)

  if len(img.shape) == 3:
    img = img[:, :, :, np.newaxis ]

  for startpt in xyzrange( img_offset, img_end, vol.underlying ):
    endpt = min2(startpt + vol.underlying, shape)
    chunkimg = img[ startpt.x:endpt.x, startpt.y:endpt.y, startpt.z:endpt.z, : ]

    spt = (startpt + bounds.minpt).astype(int)
    ept = (endpt + bounds.minpt).astype(int)
  
    yield chunkimg, spt, ept 

def chunknames(bbox, volume_bbox, key, chunk_size):
  paths = []

  for x,y,z in xyzrange( bbox.minpt, bbox.maxpt, chunk_size ):
    highpt = min2(Vec(x,y,z) + chunk_size, volume_bbox.maxpt)
    filename = "{}-{}_{}-{}_{}-{}".format(
      x, highpt.x,
      y, highpt.y, 
      z, highpt.z
    )
    paths.append( os.path.join(key, filename) )

  return paths
