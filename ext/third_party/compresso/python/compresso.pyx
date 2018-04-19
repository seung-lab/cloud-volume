# cimports
cimport numpy as np
cimport cython

# python imports
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
import numpy as np

ctypedef fused Type:
    uint8_t
    uint16_t
    uint32_t
    uint64_t

# import c++ functions
cdef extern from "compresso.hxx" namespace "Compresso":
    unsigned char *Compress(...)
    Type *Decompress[Type](...)


class Compresso(object):
    @staticmethod
    def name():
        return 'Compresso'

    @staticmethod
    def compress(Type[:,:,:] data, res, steps):
        # call the c++ compression function
        cdef long *cpp_res = [res[0], res[1], res[2]]
        cdef long *cpp_steps = [steps[0], steps[1], steps[2]]
        cdef long *nentries = [0]
        cdef unsigned char *compressed_data = Compress(&(data[0,0,0]), cpp_res, cpp_steps, nentries)

        # convert to numpy array
        cdef unsigned char[:] tmp_compressed_data = <unsigned char[:nentries[0]]> compressed_data

        return np.asarray(tmp_compressed_data)

    @staticmethod
    def decompress(data):
        # get the number of bytes per uint (1, 2, 4, or 8)
        # the 76 comes from the offset in the header
        BYTE_OFFSET = 76
        nbytes = data[BYTE_OFFSET]

        # call the c++ decompression function
        cdef long *res = [0, 0, 0]
        cdef np.ndarray[unsigned char, ndim=1, mode='c'] cpp_data = np.ascontiguousarray(data)

        # just call this as unsigned long and convert later    
        # TODO this is a bad hack
        cdef unsigned long *cpp_decompressed_data = Decompress['unsigned long'](&(cpp_data[0]), res)

        # convert the c++ pointer to a numpy array
        nentries = res[0] * res[1] * res[2]
        cdef unsigned long[:] tmp_decompressed_data = <unsigned long[:nentries]> cpp_decompressed_data
        decompressed_data = np.asarray(tmp_decompressed_data).reshape((res[0], res[1], res[2]))

        # convert to a different data type if needed
        if nbytes == 1: decompressed_data = decompressed_data.astype(np.uint8)
        elif nbytes == 2: decompressed_data = decompressed_data.astype(np.uint16)
        elif nbytes == 4: decompressed_data = decompressed_data.astype(np.uint32)

        return np.asarray(decompressed_data)