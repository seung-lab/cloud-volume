from libc.stdio cimport FILE, fopen, fwrite, fclose
from libc.stdlib cimport calloc, free
from cpython cimport array 
import array
import sys

cimport numpy as numpy

import numpy as np

__VERSION__ = '1.2.0'

SUPPORTED_PYTHON_VERSION = (sys.version_info[0] == 3)

FPZ_ERROR_STRINGS = [
  "success",
  "cannot read stream",
  "cannot write stream",
  "not an fpz stream",
  "fpz format version not supported",
  "precision not supported",
  "memory buffer overflow"
]

cdef extern from "fpzip.h":
  ctypedef struct FPZ:
    int type
    int prec
    int nx
    int ny
    int nz
    int nf

  cdef FPZ* fpzip_read_from_file(FILE* file)
  cdef FPZ* fpzip_read_from_buffer(void* buffer) 
  cdef int fpzip_read_header(FPZ* fpz)
  cdef size_t fpzip_read(FPZ* fpz, void* data)
  cdef void fpzip_read_close(FPZ* fpz)
  
  cdef FPZ* fpzip_write_to_file(FILE* file)
  cdef FPZ* fpzip_write_to_buffer(void* buffer, size_t size)
  cdef int fpzip_write_header(FPZ* fpz)
  cdef int fpzip_write(FPZ* fpz, const void* data)
  cdef void fpzip_write_close(FPZ* fpz)

  ctypedef enum fpzipError:
    fpzipSuccess             = 0, # no error 
    fpzipErrorReadStream     = 1, # cannot read stream 
    fpzipErrorWriteStream    = 2, # cannot write stream 
    fpzipErrorBadFormat      = 3, # magic mismatch; not an fpz stream 
    fpzipErrorBadVersion     = 4, # fpz format version not supported 
    fpzipErrorBadPrecision   = 5, # precision not supported 
    fpzipErrorBufferOverflow = 6  # compressed buffer overflow 

  cdef fpzipError fpzip_errno = 0

class FpzipError(Exception):
  pass

class FpzipWriteError(FpzipError):
  pass

class FpzipReadError(FpzipError):
  pass

cpdef allocate(typecode, ct):
  cdef array.array array_template = array.array(typecode, [])
  # create an array with 3 elements with same type as template
  return array.clone(array_template, ct, zero=True)

def compress(data, precision=0):
  """
  fpzip.compress(data, precision=0)

  Takes a 3d or 4d numpy array of floats or doubles and returns
  a compressed bytestring.
  """
  if not SUPPORTED_PYTHON_VERSION:
    raise NotImplementedError("This fpzip extension only supports Python 3.")

  assert data.dtype in (np.float32, np.float64)

  if len(data.shape) == 3:
    data = data[:,:,:, np.newaxis ]

  data = np.asfortranarray(data)

  header_bytes = 28 # read.cpp:fpzip_read_header + 4 for some reason

  fptype = 'f' if data.dtype == np.float32 else 'd'
  cdef array.array compression_buf = allocate(fptype, data.size + header_bytes)

  cdef FPZ* fpz_ptr
  if fptype == 'f':
    fpz_ptr = fpzip_write_to_buffer(compression_buf.data.as_floats, data.nbytes + header_bytes)
  else:
    fpz_ptr = fpzip_write_to_buffer(compression_buf.data.as_doubles, data.nbytes + header_bytes)

  if data.dtype == np.float32:
    fpz_ptr[0].type = 0 # float
  else:
    fpz_ptr[0].type = 1 # double

  fpz_ptr[0].prec = precision
  fpz_ptr[0].nx = data.shape[0]
  fpz_ptr[0].ny = data.shape[1]
  fpz_ptr[0].nz = data.shape[2]
  fpz_ptr[0].nf = data.shape[3]

  if fpzip_write_header(fpz_ptr) == 0:
    raise FpzipWriteError("Cannot write header. %s" % FPZ_ERROR_STRINGS[fpzip_errno])

  if data.size == 0:
    fpzip_write_close(fpz_ptr)
    return bytes(compression_buf)[:header_bytes] 

  # can't get raw ptr from numpy object directly, implicit magic
  # float or double shouldn't matter since we're about to cast to void pointer
  cdef float[:,:,:,:] arr_memview = data 
  cdef size_t outbytes = fpzip_write(fpz_ptr, <void*>&arr_memview[0,0,0,0])
  if outbytes == 0:
    raise FpzipWriteError("Compression failed. %s" % FPZ_ERROR_STRINGS[fpzip_errno])

  fpzip_write_close(fpz_ptr)
  return bytes(compression_buf)[:outbytes] 

def decompress(bytes encoded):
  """
  fpzip.decompress(encoded)

  Accepts an fpzip encoded bytestring (e.g. b'fpy)....') and 
  returns the 4d numpy array that generated it.
  """
  if not SUPPORTED_PYTHON_VERSION:
    raise NotImplementedError("This fpzip extension only supports Python 3.")
  
  # line below necessary to convert from PyObject to a naked pointer
  cdef unsigned char *encodedptr = <unsigned char*>encoded 
  cdef FPZ* fpz_ptr = fpzip_read_from_buffer(<void*>encodedptr)

  if fpzip_read_header(fpz_ptr) == 0:
    raise FpzipReadError("cannot read header: %s" % FPZ_ERROR_STRINGS[fpzip_errno])

  fptype = 'f' if fpz_ptr[0].type == 0 else 'd'
  nx, ny, nz, nf = fpz_ptr[0].nx, fpz_ptr[0].ny, fpz_ptr[0].nz, fpz_ptr[0].nf

  cdef array.array buf = allocate(fptype, nx * ny * nz * nf)

  cdef size_t read_bytes = 0;
  if fptype == 'f':
    read_bytes = fpzip_read(fpz_ptr, buf.data.as_floats)
  else:
    read_bytes = fpzip_read(fpz_ptr, buf.data.as_doubles)

  if read_bytes == 0:
    raise FpzipReadError("decompression failed: %s" % FPZ_ERROR_STRINGS[fpzip_errno])

  fpzip_read_close(fpz_ptr)

  dtype = np.float32 if fptype == 'f' else np.float64
  return np.frombuffer(buf, dtype=dtype).reshape( (nx, ny, nz, nf), order='F')


