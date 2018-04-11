import errno
import mmap
import os

import posix_ipc
from posix_ipc import O_CREAT
import numpy as np
import psutil

from .lib import Bbox, Vec

mmaps = []

class MemoryAllocationError(Exception):
  pass

def reinit():
  """For use after a process fork only. Trashes bad file descriptors and resets tracking."""
  global mmaps
  mmaps = []

def bbox2array(vol, bbox):
  shape = list(bbox.size3()) + [ vol.num_channels ]
  return ndarray(shape=shape, dtype=vol.dtype, location=vol.shared_memory_id)

def ndarray(shape, dtype, location):
  nbytes = Vec(*shape).rectVolume() * np.dtype(dtype).itemsize
  available = psutil.virtual_memory().available

  preexisting = 0
  # This might only work on Ubuntu
  shmloc = os.path.join('/dev/shm/', location)
  if os.path.exists(shmloc):
    preexisting = os.path.getsize(shmloc)

  if (nbytes - preexisting) > available:
    overallocated = nbytes - preexisting - available
    overpercent = (100 * overallocated / (preexisting + available))
    raise MemoryAllocationError("""
      Requested more memory than is available. 

      Shared Memory Location:  {}

      Shape:                   {}
      Requested Bytes:         {} 
      
      Available Bytes:         {} 
      Preexisting Bytes*:      {} 

      Overallocated Bytes*:    {} (+{:.2f}%)

      * Preexisting is only correct on linux systems that support /dev/shm/""" \
        .format(location, shape, nbytes, available, preexisting, overallocated, overpercent))

  try:
    shared = posix_ipc.SharedMemory(location, flags=O_CREAT, size=int(nbytes))
    array_like = mmap.mmap(shared.fd, shared.size)
    os.close(shared.fd)
    renderbuffer = np.ndarray(buffer=array_like, dtype=dtype, shape=shape)
  except OSError as err:
    if err.errno == errno.ENOMEM: # Out of Memory
      posix_ipc.unlink_shared_memory(location)      
    raise

  return array_like, renderbuffer

def track_mmap(array_like):
  global mmaps
  mmaps.append(array_like)

def cleanup():
  global mmaps 

  for array_like in mmaps:
    if not array_like.closed:
      array_like.close()
  mmaps = []

def unlink(location):
  try:
    posix_ipc.unlink_shared_memory(location)
  except posix_ipc.ExistentialError:
    return False
  return True
