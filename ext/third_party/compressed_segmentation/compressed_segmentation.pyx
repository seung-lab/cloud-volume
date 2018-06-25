from libc.stdio cimport FILE, fopen, fwrite, fclose
from libc.stdlib cimport calloc, free
from libc.stdint cimport uint32_t, uint64_t
from cpython cimport array 
import array
import sys

from libcpp.vector cimport vector

cimport numpy as numpy

import numpy as np

cdef extern "compress_segmentation.h":
  cdef void CompressChannels[Label](
    Label* input, 
    const ptrdiff_t input_strides[4],
    const ptrdiff_t volume_size[4],
    const ptrdiff_t block_size[3],
    vector[uint32_t] output
  )


cdef extern "decompress_segmentation.h":
  cdef void DecompressChannels[Label](
    const uint32_t* input,
    const ptrdiff_t volume_size[4],
    const ptrdiff_t block_size[3],
    vector<Label>* output
  )


cdef np.ndarray[np.int32_t, ndim=1] arr_from_vector[T](vector[T] v):
  cdef:
    int i = 0
    int size = v.size()
    T[::1] a = np.empty(v.size(), T)
  
  while i < size:
    a[i] = v[i]
    i += 1
  return a


def compress(data, block_size=(8,8,8)):
  cdef vector[uint32_t] *output = new vector[uint32_t]()

  cdef uint32_t[:,:,:,:] arr_memview = data 
  cdef ptrdiff_t volume_size[4] 
  volume_size[:] = data.shape[:4]

  cdef ptrdiff_t block_sizeptr[3]
  block_sizeptr[:] = block_size[:3]

  cdef ptrdiff_t input_strides[3]
  input_strides[:] = [ 1, 1, 1 ]

  if cube.dtype == np.uint32:
    CompressChannels[uint32_t](
      <uint32_t*>&arr_memview[0,0,0,0],
      input_strides,
      volume_size,
      block_sizeptr,
      output
    )
  else:
    CompressChannels[uint64_t](
      <uint64_t*>&arr_memview[0,0,0,0],
      input_strides,
      volume_size,
      block_sizeptr,
      output
    )

  return arr_from_vector(output).tobytes('C')

def decompress(bytes buf):
  pass



