import posix_ipc
from posix_ipc import O_CREAT
import mmap
import os

import numpy as np

from .lib import Bbox, Vec

mmaps = []

def reinit():
  """For use after a process fork only. Trashes bad file descriptors and resets tracking."""
  global mmaps
  mmaps = []

def bbox2array(vol, bbox):
  shape = list(bbox.size3()) + [ vol.num_channels ]
  return ndarray(shape=shape, dtype=vol.dtype, location=vol.shared_memory_id)

def ndarray(shape, dtype, location):
  nbytes = Vec(*shape).rectVolume() * np.dtype(dtype).itemsize
  shared = posix_ipc.SharedMemory(location, flags=O_CREAT, size=int(nbytes))
  array_like = mmap.mmap(shared.fd, shared.size)
  os.close(shared.fd)
  renderbuffer = np.ndarray(buffer=array_like, dtype=dtype, shape=shape)
  return array_like, renderbuffer

def track_mmap(array_like):
  global mmaps
  mmaps.append(array_like)

def cleanup(vol=None):
  global mmaps 

  for array_like in mmaps:
    array_like.close()
  mmaps = []

def unlink(vol):
  try:
    posix_ipc.unlink_shared_memory(vol.shared_memory_id)
  except posix_ipc.ExistentialError:
    pass
