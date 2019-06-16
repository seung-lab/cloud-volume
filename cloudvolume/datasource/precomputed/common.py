from functools import partial
import itertools
import math
import multiprocessing as mp
import concurrent.futures
import os
import signal

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

def chunknames(bbox, volume_bbox, key, chunk_size):
  for x,y,z in xyzrange( bbox.minpt, bbox.maxpt, chunk_size ):
    highpt = min2(Vec(x,y,z) + chunk_size, volume_bbox.maxpt)
    filename = "{}-{}_{}-{}_{}-{}".format(
      x, highpt.x,
      y, highpt.y, 
      z, highpt.z
    )
    yield os.path.join(key, filename)

def shade(dest_img, dest_bbox, src_img, src_bbox):
  """
  Shade dest_img at coordinates dest_bbox using the
  image contained in src_img at coordinates src_bbox.

  The buffer will only be painted in the overlapping
  region of the content.

  Returns: void
  """
  if not Bbox.intersects(dest_bbox, src_bbox):
    return

  spt = max2(src_bbox.minpt, dest_bbox.minpt)
  ept = min2(src_bbox.maxpt, dest_bbox.maxpt)
  dbox = Bbox(spt, ept) - dest_bbox.minpt

  ZERO3 = Vec(0,0,0)
  istart = max2(spt - src_bbox.minpt, ZERO3)
  iend = min2(ept - src_bbox.maxpt, ZERO3) + src_img.shape[:3]
  sbox = Bbox(istart, iend)

  while src_img.ndim < 4:
    src_img = src_img[..., np.newaxis]
  
  dest_img[ dbox.to_slices() ] = src_img[ sbox.to_slices() ]
