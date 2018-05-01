from libc.stdio cimport FILE, fopen, fwrite, fclose
from libc.stdlib cimport calloc, free
from cpython cimport array 
import array

cimport numpy as numpy

import numpy as np

__VERSION__ = '1.2.0'

FPZ_ERROR_STRINGS = [
  "success",
  "cannot read stream",
  "cannot write stream",
  "not an fpz stream",
  "fpz format version not supported",
  "precision not supported",
  "memory buffer overflow"
]

cdef extern from "inc/fpzip.h":
  struct FPZ:
    int type
    int prec
    int nx
    int ny
    int nz
    int nf
  enum fpzipError:
    fpzipSuccess             = 0, # no error 
    fpzipErrorReadStream     = 1, # cannot read stream 
    fpzipErrorWriteStream    = 2, # cannot write stream 
    fpzipErrorBadFormat      = 3, # magic mismatch; not an fpz stream 
    fpzipErrorBadVersion     = 4, # fpz format version not supported 
    fpzipErrorBadPrecision   = 5, # precision not supported 
    fpzipErrorBufferOverflow = 6  # compressed buffer overflow 
  fpzipError fpzip_errno

  FPZ* fpzip_read_from_file(FILE* file)
  FPZ* fpzip_read_from_buffer(void* buffer) 
  int fpzip_read_header(FPZ* fpz)
  size_t fpzip_read(FPZ* fpz, void* data)
  void fpzip_read_close(FPZ* fpz)
  
  FPZ* fpzip_write_to_file(FILE* file)
  FPZ* fpzip_write_to_buffer(void* buffer, size_t size);
  int fpzip_write_header(FPZ* fpz)
  size_t fpzip_write(FPZ* fpz, const void* data)
  void fpzip_write_close(FPZ* fpz)

class FpzipError(Exception):
  pass

class FpzipWriteError(FpzipError):
  pass

class FpzipReadError(FpzipError):
  pass

cpdef allocate(dtype, ct):
  cdef array.array array_template = array.array(dtype, [])
  # create an array with 3 elements with same type as template
  return array.clone(array_template, ct, zero=True)

def compress(data, precision=0):
  assert data.dtype in (np.float32, np.float64)

  if len(data.shape) == 3:
    data = data[:,:,:, np.newaxis ]

  fptype = 'f' if data.dtype == np.float32 else 'd'
  cdef array.array compression_buf = allocate(fptype, data.itemsize)
  
  cdef FPZ* fpz_ptr
  if fptype == 'f':
    fpz_ptr = fpzip_write_to_buffer(compression_buf.data.as_floats, data.nbytes)
  else:
    fpz_ptr = fpzip_write_to_buffer(compression_buf.data.as_doubles, data.nbytes)

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

  cdef size_t outbytes = fpzip_write(fpz_ptr, <void*>data.data)
  if outbytes == 0:
    raise FpzipWriteError("Compression failed. %s" % FPZ_ERROR_STRINGS[fpzip_errno])

  fpzip_write_close(fpz_ptr)
  return bytes(compression_buf)

def decompress(encoded):
  cdef FPZ* fpz_ptr = fpzip_read_from_buffer(encoded.data.as_voidptr)

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

  return np.asarray(buf.data)