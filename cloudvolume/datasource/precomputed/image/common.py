import copy
from functools import partial
import itertools
import json
import math
import multiprocessing as mp
import os
import platform
import posixpath
import signal
import traceback

import numpy as np
import concurrent.futures
from tqdm import tqdm

from ....lib import (
  xyzrange, min2, max2, Vec, Bbox, 
  sip, totalfn
)
from .... import sharedmemory as shm

error_queue = None
progress_queue = None

def check_error_queue():
  if error_queue.empty():
    return

  errors = []
  while not error_queue.empty():
    err = error_queue.get()
    if err is not StopIteration:
      errors.append(err)
  if len(errors):
    raise Exception(errors)

def progress_queue_listener(q, total, desc):
  pbar = tqdm(total=total, desc=desc)
  for ct in iter(q.get, None):
    pbar.update(ct)

def error_capturing_fn(fn, *args, **kwargs):
  try:
    return fn(*args, **kwargs)
  except Exception as err:
    traceback.print_exception(type(err), err, err.__traceback__)
    error_queue.put(err)
    return 0

def initialize_synchronization(progress_queue, fs_lock):
  from . import rx, tx
  rx.progress_queue = progress_queue
  tx.progress_queue = progress_queue
  rx.fs_lock = fs_lock
  tx.fs_lock = fs_lock

def parallel_execution(
  fn, items, parallel, 
  progress, desc="Progress",
  total=None, cleanup_shm=None,
  block_size=1000, min_block_size=10
):
  global error_queue

  error_queue = mp.Queue()
  progress_queue = mp.Queue()
  fs_lock = mp.Lock()

  if parallel is True:
    parallel = mp.cpu_count()
  elif parallel <= 0:
    raise ValueError(f"Parallel must be a positive number or boolean (True: all cpus). Got: {parallel}")

  def cleanup(signum, frame):
    if cleanup_shm:
      shm.unlink(cleanup_shm)

  fn = partial(error_capturing_fn, fn)
  total = totalfn(items, total)

  if total is not None and (total / parallel) < block_size:
    block_size = int(math.ceil(total / parallel))
    block_size = max(block_size, min_block_size)

  prevsigint = signal.getsignal(signal.SIGINT)
  prevsigterm = signal.getsignal(signal.SIGTERM)

  signal.signal(signal.SIGINT, cleanup)
  signal.signal(signal.SIGTERM, cleanup)

  # Fix for MacOS which can segfault due to 
  # urllib calling libdispatch which is not fork-safe
  # https://bugs.python.org/issue30385
  no_proxy = os.environ.get("no_proxy", "")
  if platform.system().lower() == "darwin":
    os.environ["no_proxy"] = "*"

  try:
    if progress:
      proc = mp.Process(
        target=progress_queue_listener, 
        args=(progress_queue,total,desc)
      )
      proc.start()

    with concurrent.futures.ProcessPoolExecutor(
      max_workers=parallel,
      initializer=initialize_synchronization,
      initargs=(progress_queue,fs_lock),
    ) as pool:
      pool.map(fn, sip(items, block_size))
  finally: 
    if platform.system().lower() == "darwin":
      os.environ["no_proxy"] = no_proxy

    signal.signal(signal.SIGINT, prevsigint)
    signal.signal(signal.SIGTERM, prevsigterm)

    if progress:
      progress_queue.put(None)
      proc.join()
      proc.close()

    progress_queue.close()
    progress_queue.join_thread()

  check_error_queue()
  error_queue.close()
  error_queue.join_thread()

def chunknames(bbox, volume_bbox, key, chunk_size, protocol=None):
  path = posixpath if protocol != 'file' else os.path

  class ChunkNamesIterator():
    def __len__(self):
      # round up and avoid conversion to float
      n_chunks = (bbox.dx + chunk_size[0] - 1) // chunk_size[0]
      n_chunks *= (bbox.dy + chunk_size[1] - 1) // chunk_size[1]
      n_chunks *= (bbox.dz + chunk_size[2] - 1) // chunk_size[2]
      return n_chunks
    def __iter__(self):
      for x,y,z in xyzrange( bbox.minpt, bbox.maxpt, chunk_size ):        
        xf = min(x + chunk_size.x, volume_bbox.maxpt.x)
        yf = min(y + chunk_size.y, volume_bbox.maxpt.y)
        zf = min(z + chunk_size.z, volume_bbox.maxpt.z)
        filename = f"{x}-{xf}_{y}-{yf}_{z}-{zf}"
        yield path.join(key, filename)

  return ChunkNamesIterator()

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
  if hasattr(gridpt, "__len__") and len(gridpt) == 0: # generators don't have len
    return np.zeros((0,), dtype=np.uint32)

  gridpt = np.asarray(gridpt, dtype=np.uint32)
  single_input = False
  if gridpt.ndim == 1:
    gridpt = np.atleast_2d(gridpt)
    single_input = True

  code = np.zeros((gridpt.shape[0],), dtype=np.uint64)
  num_bits = [ math.ceil(math.log2(size)) for size in grid_size ]
  j = np.uint64(0)
  one = np.uint64(1)

  if sum(num_bits) > 64:
    raise ValueError(f"Unable to represent grids that require more than 64 bits. Grid size {grid_size} requires {num_bits} bits.")

  max_coords = np.max(gridpt, axis=0)
  if np.any(max_coords >= grid_size):
    raise ValueError(f"Unable to represent grid points larger than the grid. Grid size: {grid_size} Grid points: {gridpt}")

  for i in range(max(num_bits)):
    for dim in range(3):
      if 2 ** i < grid_size[dim]:
        bit = (((np.uint64(gridpt[:, dim]) >> np.uint64(i)) & one) << j)
        code |= bit
        j += one

  if single_input:
    return code[0]
  return code

def morton_code_to_bbox(code, volume_bbox, chunk_size):
  chunk_size = Vec(*chunk_size)

  grid_size = np.ceil(volume_bbox.size3() / chunk_size).astype(np.int64)
  
  gridpt = morton_code_to_gridpt(code, grid_size)
  
  bbox = Bbox(gridpt, gridpt + 1) 
  bbox *= chunk_size
  bbox += volume_bbox.minpt
  return bbox

def morton_code_to_gridpt(code, grid_size):
  gridpt = np.zeros([3,], dtype=int)

  num_bits = [ math.ceil(math.log2(size)) for size in grid_size ]
  j = np.uint64(0)
  one = np.uint64(1)

  if sum(num_bits) > 64:
    raise ValueError(f"Unable to represent grids that require more than 64 bits. Grid size {grid_size} requires {num_bits} bits.")

  max_coords = np.max(gridpt, axis=0)
  if np.any(max_coords >= grid_size):
    raise ValueError(f"Unable to represent grid points larger than the grid. Grid size: {grid_size} Grid points: {gridpt}")

  code = np.uint64(code)

  for i in range(max(num_bits)):
    for dim in range(3):
      i = np.uint64(i)
      if 2 ** i < grid_size[dim]:
        bit = np.uint64((code >> j) & one)
        gridpt[dim] += (bit << i)
        j += one

  return gridpt


def shade(dest_img, dest_bbox, src_img, src_bbox, channel=None):
  """
  Shade dest_img at coordinates dest_bbox using the
  image contained in src_img at coordinates src_bbox.

  The buffer will only be painted in the overlapping
  region of the content.

  Note: channel is useful for zarr, not precomputed

  Returns: void
  """
  if not Bbox.intersects(dest_bbox, src_bbox) or src_img is None:
    return

  spt = np.maximum(src_bbox.minpt, dest_bbox.minpt)
  ept = np.minimum(src_bbox.maxpt, dest_bbox.maxpt)

  dest_minpt = np.minimum(spt, ept) - dest_bbox.minpt
  dest_maxpt = np.maximum(spt, ept) - dest_bbox.minpt

  ZERO3 = np.zeros((3,), dtype=spt.dtype)
  istart = np.maximum(spt - src_bbox.minpt, ZERO3)
  iend = np.minimum(ept - src_bbox.maxpt, ZERO3) + src_img.shape[:3]

  while src_img.ndim < 4:
    src_img = src_img[..., np.newaxis]
  
  if channel:
    dest_img[ 
      dest_minpt[0]:dest_maxpt[0],
      dest_minpt[1]:dest_maxpt[1],
      dest_minpt[2]:dest_maxpt[2],
      channel
    ] = src_img[
      istart[0]:iend[0], 
      istart[1]:iend[1],
      istart[2]:iend[2]
    ]
  else:
    dest_img[ 
      dest_minpt[0]:dest_maxpt[0],
      dest_minpt[1]:dest_maxpt[1],
      dest_minpt[2]:dest_maxpt[2],
    ] = src_img[
      istart[0]:iend[0], 
      istart[1]:iend[1],
      istart[2]:iend[2]
    ]
