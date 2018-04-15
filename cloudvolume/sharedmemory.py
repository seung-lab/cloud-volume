import errno
import mmap
import os
import sys

import multiprocessing as mp

from six.moves import range
import posix_ipc
from posix_ipc import O_CREAT
import numpy as np
import psutil

import time

from .lib import Bbox, Vec, mkdir

mmaps = []

SHM_DIRECTORY = '/dev/shm/'
OSX_SHM_DIRECTORY = '/tmp/cloudvolume-shm'

PLATFORM_SHM_DIRECTORY = SHM_DIRECTORY
if sys.platform == 'darwin':
  PLATFORM_SHM_DIRECTORY = OSX_SHM_DIRECTORY

class MemoryAllocationError(Exception):
  pass

def reinit():
  """For use after a process fork only. Trashes bad file descriptors and resets tracking."""
  global mmaps
  mmaps = []

def bbox2array(vol, bbox, lock=None):
  shape = list(bbox.size3()) + [ vol.num_channels ]
  return ndarray(shape=shape, dtype=vol.dtype, location=vol.shared_memory_id, lock=lock)

def ndarray(shape, dtype, location, lock=None):
  # OS X has problems with shared memory so 
  # emulate it using a file on disk
  if sys.platform == 'darwin':
    return ndarray_fs(shape, dtype, location, lock)
  return ndarray_shm(shape, dtype, location)

def ndarray_fs(shape, dtype, location, lock):
  nbytes = Vec(*shape).rectVolume() * np.dtype(dtype).itemsize
  block = 10 * 1024 * 1024 # 10 MiB
  directory = mkdir(OSX_SHM_DIRECTORY)
  filename = os.path.join(directory, location)

  if lock:
    lock.acquire(timeout=2)

  if not os.path.exists(filename):
    zeros = np.zeros(shape=shape, dtype=dtype)
    with open(filename, 'wb') as f:
      f.write(zeros.tostring('F'))
    del zeros

  if lock:
    lock.release()

  with open(filename, 'r+b') as f:
    array_like = mmap.mmap(f.fileno(), 0) # map entire file
  
  renderbuffer = np.ndarray(buffer=array_like, dtype=dtype, shape=shape)
  return array_like, renderbuffer

def ndarray_shm(shape, dtype, location):
  nbytes = Vec(*shape).rectVolume() * np.dtype(dtype).itemsize
  available = psutil.virtual_memory().available

  preexisting = 0
  # This might only work on Ubuntu
  shmloc = os.path.join(SHM_DIRECTORY, location)
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
  if sys.platform == 'darwin':
    directory = mkdir(OSX_SHM_DIRECTORY)
    try:
      filename = os.path.join(directory, location)
      os.unlink(filename)
      return True
    except OSError:
      return False

  try:
    posix_ipc.unlink_shared_memory(location)
  except posix_ipc.ExistentialError:
    return False
  return True
