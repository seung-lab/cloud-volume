# @license
# Copyright 2017 The Neuroglancer Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import zlib
import io
import numpy as np
from PIL import Image

from .lib import yellow

try:
  import _compressed_segmentation as cseg
  ACCELERATED_CSEG = True # C extension version
except ImportError:
  ACCELERATED_CSEG = False # Pure Python implementation
# ACCELERATED_CSEG= False

from . import compressed_segmentation as csegpy

try:
  import fpzip 
except ImportError:
  fpziperrormsg = yellow("CloudVolume: fpzip codec is not available. Was it installed? pip install fpzip")
  class fpzip():
    @classmethod
    def compress(cls, content):
      raise NotImplementedError(fpziperrormsg)
    @classmethod
    def decompress(cls, content):
      raise NotImplementedError(fpziperrormsg)

def encode(img_chunk, encoding, block_size=None):
  if encoding == "raw":
    return encode_raw(img_chunk)
  elif encoding == "kempressed":
    return encode_kempressed(img_chunk)
  elif encoding == "fpzip":
    img_chunk = np.asfortranarray(img_chunk)
    return fpzip.compress(img_chunk)
  elif encoding == "compressed_segmentation":
    return encode_compressed_segmentation(img_chunk, block_size=block_size)
  elif encoding == "jpeg":
    return encode_jpeg(img_chunk)
  elif encoding == "npz":
    return encode_npz(img_chunk)
  elif encoding == "npz_uint8":
    chunk = img_chunk * 255
    chunk = chunk.astype(np.uint8)
    return encode_npz(chunk)
  else:
    raise NotImplementedError(encoding)

def decode(filedata, encoding, shape=None, dtype=None, block_size=None):
  if (shape is None or dtype is None) and encoding not in ('npz', 'fpzip', 'kempressed'):
    raise ValueError("Only npz encoding can omit shape and dtype arguments. {}".format(encoding))

  if filedata is None or len(filedata) == 0:
    return np.zeros(shape=shape, dtype=dtype)
  elif encoding == "raw":
    return decode_raw(filedata, shape=shape, dtype=dtype)
  elif encoding == "kempressed":
    return decode_kempressed(filedata)
  elif encoding == "fpzip":
    return fpzip.decompress(filedata, order='F')
  elif encoding == "compressed_segmentation":
    return decode_compressed_segmentation(filedata, shape=shape, dtype=dtype, block_size=block_size)
  elif encoding == "jpeg":
    return decode_jpeg(filedata, shape=shape, dtype=dtype)
  elif encoding == "npz":
    return decode_npz(filedata)
  else:
    raise NotImplementedError(encoding)

def encode_jpeg(arr):
  assert arr.dtype == np.uint8

  # simulate multi-channel array for single channel arrays
  if len(arr.shape) == 3:
    arr = np.expand_dims(arr, 3) # add channels to end of x,y,z

  arr = arr.T # channels, z, y, x
  reshaped = arr.reshape(arr.shape[3] * arr.shape[2], arr.shape[1] * arr.shape[0])
  if arr.shape[0] == 1:
    img = Image.fromarray(reshaped, mode='L')
  elif arr.shape[0] == 3:
    img = Image.fromarray(reshaped, mode='RGB')
  else:
    raise ValueError("Number of image channels should be 1 or 3. Got: {}".format(arr.shape[3]))

  f = io.BytesIO()
  img.save(f, "JPEG")
  return f.getvalue()

def encode_npz(subvol):
  """
  This file format is unrelated to np.savez
  We are just saving as .npy and the compressing
  using zlib. 
  The .npy format contains metadata indicating
  shape and dtype, instead of np.tobytes which doesn't
  contain any metadata.
  """
  fileobj = io.BytesIO()
  if len(subvol.shape) == 3:
    subvol = np.expand_dims(subvol, 0)
  np.save(fileobj, subvol)
  cdz = zlib.compress(fileobj.getvalue())
  return cdz

def encode_compressed_segmentation(subvol, block_size, accelerated=ACCELERATED_CSEG):
  assert np.dtype(subvol.dtype) in (np.uint32, np.uint64)

  if accelerated:
    return encode_compressed_segmentation_c_ext(subvol, block_size)  
  return encode_compressed_segmentation_pure_python(subvol, block_size)

def encode_compressed_segmentation_c_ext(subvol, block_size):
  subvol = np.squeeze(subvol, axis=3)
  subvol = np.copy(subvol, order='C')
  return cseg.compress(subvol, block_size=block_size, order='F')

def encode_compressed_segmentation_pure_python(subvol, block_size):
  return csegpy.encode_chunk(subvol.T, block_size=block_size)

def encode_raw(subvol):
  return subvol.tostring('F')

def encode_kempressed(subvol):
  data = 2.0 + np.swapaxes(subvol, 2,3)
  return fpzip.compress(data)

def decode_kempressed(bytestring):
  """subvol not bytestring since numpy conversion is done inside fpzip extension."""
  subvol = fpzip.decompress(bytestring)
  return np.swapaxes(subvol, 3,2) - 2.0

def decode_npz(string):
  fileobj = io.BytesIO(zlib.decompress(string))
  return np.load(fileobj)

def decode_jpeg(bytestring, shape, dtype):
  img = Image.open(io.BytesIO(bytestring))
  data = np.array(img.getdata(), dtype=dtype)

  return data.reshape(shape, order='F')

def decode_raw(bytestring, shape, dtype):
  return np.frombuffer(bytestring, dtype=dtype).reshape(shape, order='F')

def decode_compressed_segmentation(bytestring, shape, dtype, block_size, accelerated=ACCELERATED_CSEG):
  assert block_size is not None

  if accelerated:
    return decode_compressed_segmentation_c_ext(bytestring, shape, dtype, block_size)

  return decode_compressed_segmentation_pure_python(bytestring, shape, dtype, block_size)

def decode_compressed_segmentation_c_ext(bytestring, shape, dtype, block_size):
  return cseg.decompress(bytes(bytestring), shape, dtype, block_size)

def decode_compressed_segmentation_pure_python(bytestring, shape, dtype, block_size):
  chunk = np.empty(shape=shape[::-1], dtype=dtype)
  csegpy.decode_chunk_into(chunk, bytestring, block_size=block_size)
  return chunk.T

