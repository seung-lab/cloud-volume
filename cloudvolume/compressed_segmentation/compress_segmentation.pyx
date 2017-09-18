# Cython interface file for wrapping the object
#
#

from libc.stdint cimport uint32_t, int64_t
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np

# c++ interface to cython
cdef extern from "compress_segmentation.cc" namespace "neuroglancer::compress_segmentation":
    cdef void CompressChannels[uint32_t](
        uint32_t* input, 
        ptrdiff_t input_strides[4],
        ptrdiff_t volume_size[4],
        ptrdiff_t block_size[3],
        vector[uint32_t]* output_vec
    )

    cdef void CompressChannels[uint64_t](
        uint64_t* input, 
        ptrdiff_t input_strides[4],
        ptrdiff_t volume_size[4],
        ptrdiff_t block_size[3],
        vector[uint32_t]* output_vec
    )


def compress(image):
    cdef np.ndarray[uint32_t, ndim=4, mode='c'] np_buff = np.ascontiguousarray(image, dtype=np.uint32)
    cdef uint32_t* input_data = <uint32_t*>np_buff.data

    cdef ptrdiff_t* input_stride = [ 1, image.shape[0], image.shape[0] * image.shape[1] ]
    cdef ptrdiff_t* volume_size = [ image.shape[0], image.shape[1], image.shape[2], image.shape[3] ]
    cdef ptrdiff_t* block_size = [ 8, 8, 8 ]

    cdef vector[uint32_t] output = []

    CompressChannels[uint32_t](
        input_data,
        input_stride,
        volume_size,
        block_size,
        &output
    )

    return np.array(output)

# def decompress(compressed_data):
#     pass

def hello():
    print "HI!"