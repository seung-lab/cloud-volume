import concurrent.futures
import copy
from functools import partial
import itertools
import json
import math
import multiprocessing as mp
import os
import posixpath
import signal

import numpy as np

from ....lib import xyzrange, min2, max2, Vec, Bbox
from .... import sharedmemory as shm

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

def chunknames(bbox, volume_bbox, key, chunk_size, protocol=None):
  path = posixpath if protocol != 'file' else os.path

  for x,y,z in xyzrange( bbox.minpt, bbox.maxpt, chunk_size ):
    highpt = min2(Vec(x,y,z) + chunk_size, volume_bbox.maxpt)
    filename = "{}-{}_{}-{}_{}-{}".format(
      x, highpt.x,
      y, highpt.y, 
      z, highpt.z
    )
    yield path.join(key, filename)

def gridpoints(bbox, volume_bbox, chunk_size):
  """
  Consider a volume as divided into a grid with the 
  first chunk labeled 1, the second 2, etc. 

  Return the grid x,y,z coordinates of a cutout as a
  sequence.
  """
  chunk_size = Vec(*chunk_size)

  grid_size = np.ceil(volume_bbox.size3() / chunk_size).astype(np.int64)
  cutout_grid_size = np.ceil(bbox.size3() / chunk_size).astype(np.int64)
  cutout_grid_offset = np.ceil((bbox.minpt - volume_bbox.minpt) / chunk_size).astype(np.int64)

  grid_cutout = Bbox( cutout_grid_offset, cutout_grid_offset + cutout_grid_size )

  for x,y,z in xyzrange( grid_cutout.minpt, grid_cutout.maxpt, (1,1,1) ):
    yield Vec(x,y,z)

def compressed_morton_code(gridpt, grid_size):
  gridpt = np.asarray(gridpt, dtype=np.uint32)
  single_input = False
  if gridpt.ndim == 1:
    gridpt = np.atleast_2d(gridpt)
    single_input = True

  code = np.zeros((gridpt.shape[0],), dtype=np.uint64)
  num_bits = max(( math.ceil(math.log2(size)) for size in grid_size ))
  j = np.uint64(0)
  one = np.uint64(1)

  for i in range(num_bits):
    for dim in range(3):
      if 2 ** i <= grid_size[dim]:
        bit = (((np.uint64(gridpt[:, dim]) >> np.uint64(i)) & one) << j)
        code |= bit
        j += one

  if single_input:
    return code[0]
  return code

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

