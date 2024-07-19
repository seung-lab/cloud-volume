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
from typing import Any, Optional, Sequence, Dict, Union, Tuple, Callable

import zlib
import io
import numpy as np

# codecs

NEEDS_INSTALL = {}

try:
  import fpzip
except ImportError:
  NEEDS_INSTALL['fpzip'] = 'fpzip'

try:
  import pyspng
except ImportError:
  NEEDS_INSTALL['png'] = 'pyspng-seunglab'

try:
  import simplejpeg
except ImportError:
  NEEDS_INSTALL['jpeg'] = 'simplejpeg'

try:
  import compresso
except ImportError:
  NEEDS_INSTALL['compresso'] = 'compresso'

try:
  import crackle
except ImportError:
  NEEDS_INSTALL['crackle'] = 'crackle-codec'

try:
  import zfpc
except ImportError:
  NEEDS_INSTALL['zfpc'] = 'zfpc'

try:
  import imagecodecs
except ImportError:
  NEEDS_INSTALL['jpegxl'] = 'imagecodecs'

try:
  import compressed_segmentation as cseg
except ImportError:
  NEEDS_INSTALL['compressed_segmentation'] = 'compressed-segmentation'

def check_installed(encoding):
  if encoding in NEEDS_INSTALL:
    raise ImportError(f"Optional codec {encoding} is not installed. Run: pip install {NEEDS_INSTALL[encoding]}")

import fastremap

from tqdm import tqdm

from .lib import yellow, nvl
from .types import ShapeType

DEFAULT_CSEG_BLOCK_SIZE = (8,8,8)

SUPPORTED_ENCODINGS = (
  "raw", "kempressed", "fpzip",
  "compressed_segmentation", "compresso",
  "crackle", "jpeg", "jpegxl", "png", "zfpc"
)

def encode(
  img_chunk:np.ndarray, 
  encoding:str, 
  block_size:Optional[Sequence[int]] = None,
  compression_params:dict = {},
) -> bytes:
  level = compression_params.get("level", None)

  check_installed(encoding)

  if encoding == "raw":
    return encode_raw(img_chunk)
  elif encoding == "kempressed":
    return encode_kempressed(img_chunk)
  elif encoding == "fpzip":
    img_chunk = np.asfortranarray(img_chunk)
    return fpzip.compress(img_chunk, order='F', precision=nvl(level, 0))
  elif encoding == "zfpc":
    return zfpc.compress(np.asfortranarray(img_chunk), **compression_params)
  elif encoding == "compressed_segmentation":
    if block_size is None:
      block_size = compression_params.get("compressed_segmentation_block_size", None)
    return encode_compressed_segmentation(img_chunk, block_size=block_size)
  elif encoding == "compresso":
    return compresso.compress(img_chunk[:,:,:,0])
  elif encoding == "crackle":
    return crackle.compress(img_chunk[:,:,:,0])
  elif encoding == "jpeg":
    return encode_jpeg(img_chunk, nvl(level, 85))
  elif encoding == "jpegxl":
    return encode_jpegxl(img_chunk, nvl(level, 85))
  elif encoding == "png":
    return encode_png(img_chunk, nvl(level, 9))
  elif encoding == "npz":
    return encode_npz(img_chunk)
  elif encoding == "npz_uint8":
    chunk = img_chunk * 255
    chunk = chunk.astype(np.uint8)
    return encode_npz(chunk)
  else:
    raise NotImplementedError(encoding)

def decode(
  filedata:bytes, 
  encoding:str, 
  shape:Optional[Sequence[int]] = None, 
  dtype:Any = None, 
  block_size:Optional[Sequence[int]] = None, 
  background_color:int = 0
) -> np.ndarray:
  if (shape is None or dtype is None) and encoding not in ('npz', 'fpzip', 'kempressed', 'crackle', 'compresso'):
    raise ValueError(
      f"Only npz, fpzip, kempressed, crackle, and compresso encoding "
      f"can omit shape and dtype arguments. Got: {encoding}"
    )

  check_installed(encoding)

  if filedata is None or len(filedata) == 0:
    if background_color == 0:
      return np.zeros(shape=shape, dtype=dtype, order="F")
    else:
      return np.full(shape=shape, fill_value=background_color, dtype=dtype, order="F")
  elif encoding == "raw":
    return decode_raw(filedata, shape=shape, dtype=dtype)
  elif encoding == "kempressed":
    return decode_kempressed(filedata)
  elif encoding == "fpzip":
    return fpzip.decompress(filedata, order='F')
  elif encoding == "zfpc":
    return zfpc.decompress(filedata)
  elif encoding == "compressed_segmentation":
    return decode_compressed_segmentation(filedata, shape=shape, dtype=dtype, block_size=block_size)
  elif encoding == "compresso":
    return compresso.decompress(filedata).reshape(shape)
  elif encoding == "crackle":
    return crackle.decompress(filedata).reshape(shape)
  elif encoding == "jpeg":
    return decode_jpeg(filedata, shape=shape, dtype=dtype)
  elif encoding == "jpegxl":
    return decode_jpegxl(filedata, shape=shape)
  elif encoding == "png":
    return decode_png(filedata, shape=shape, dtype=dtype)
  elif encoding == "npz":
    return decode_npz(filedata)
  else:
    raise NotImplementedError(encoding)

def decode_binary_image(
  label:int,
  filedata:bytes,
  encoding:str, 
  shape:Optional[Sequence[int]] = None, 
  dtype:Any = None, 
  block_size:Optional[Sequence[int]] = None, 
  background_color:int = 0,
):
  check_installed(encoding)

  if encoding == "crackle":
    return crackle.decompress(filedata, label=label).reshape(shape)

  labels = decode(filedata, encoding, shape, dtype, block_size, background_color)
  return labels == label

def as2d(arr):
  # simulate multi-channel array for single channel arrays
  while arr.ndim < 4:
    arr = arr[..., np.newaxis] # add channels to end of x,y,z

  num_channel = arr.shape[3]
  reshaped = arr.T
  reshaped = np.moveaxis(reshaped, 0, -1)
  reshaped = reshaped.reshape(
    reshaped.shape[0] * reshaped.shape[1], reshaped.shape[2], num_channel
  )
  return reshaped, num_channel

def encode_jpegxl(arr, level):
  if not np.issubdtype(arr.dtype, np.uint8):
    raise ValueError("Only accepts uint8 arrays. Got: " + str(arr.dtype))

  arr, num_channel = as2d(arr)
  lossless = level >= 100

  if num_channel == 1:
    return imagecodecs.jpegxl_encode(
      arr[:,:,0],
      photometric="GRAY",
      level=level,
      lossless=lossless,
    )
  elif num_channel == 3:
    arr = np.transpose(arr, axes=[2, 0, 1])
    return imagecodecs.jpegxl_encode(
      arr,
      photometric="RGB",
      level=level,
      lossless=lossless,
    )
  raise ValueError("Number of image channels should be 1 or 3. Got: {}".format(arr.shape[3]))

def decode_jpegxl(binary:bytes, shape):
  data = imagecodecs.jpegxl_decode(binary)
  if shape[3] == 3:
    data = np.transpose(data, axes=[1, 2, 0])

  return data.ravel().reshape(shape, order='F')

def encode_jpeg(arr, quality=85):
  if not np.issubdtype(arr.dtype, np.uint8):
    raise ValueError("Only accepts uint8 arrays. Got: " + str(arr.dtype))

  arr, num_channel = as2d(arr)
  arr = np.ascontiguousarray(arr)

  if num_channel == 1:
    return simplejpeg.encode_jpeg(
      arr, 
      colorspace="GRAY",
      colorsubsampling="GRAY",
      quality=quality,
    )
  elif num_channel == 3:
    return simplejpeg.encode_jpeg(
      arr,
      colorspace="RGB",
      quality=quality,
    )
  raise ValueError("Number of image channels should be 1 or 3. Got: {}".format(arr.shape[3]))

def encode_png(arr, compress_level=9):
  if arr.dtype not in (np.uint8, np.uint16):
    raise ValueError("Only accepts uint8 and uint16 arrays. Got: " + str(arr.dtype))

  arr, num_channel = as2d(arr)
  return pyspng.encode(arr, compress_level=compress_level)

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

def encode_compressed_segmentation(
  subvol:np.ndarray, 
  block_size:Sequence[int],
) -> bytes:
  if np.dtype(subvol.dtype) not in (np.uint32, np.uint64):
    raise ValueError("compressed_segmentation only supports uint32 and uint64 datatypes. Got: " + str(subvol.dtype))

  subvol = np.squeeze(subvol, axis=3)
  if subvol.flags.c_contiguous:
    order = 'C' 
  elif subvol.flags.f_contiguous:
    order = 'F'
  else:
    order = 'F'
    subvol = np.asfortranarray(subvol)

  if not subvol.flags.writeable:
    subvol = np.copy(subvol)

  return cseg.compress(subvol, block_size=block_size, order=order)

def encode_raw(subvol):
  return subvol.tobytes('F')

def encode_kempressed(subvol):
  data = 2.0 + np.swapaxes(subvol, 2,3)
  return fpzip.compress(data, order='F')

def decode_kempressed(bytestring):
  """subvol not bytestring since numpy conversion is done inside fpzip extension."""
  subvol = fpzip.decompress(bytestring, order='F')
  return np.swapaxes(subvol, 3,2) - 2.0

def decode_npz(string):
  fileobj = io.BytesIO(zlib.decompress(string))
  return np.load(fileobj)

def decode_jpeg(bytestring, shape, dtype):
  colorspace = "RGB" if len(shape) > 3 and shape[3] > 1 else "GRAY"
  data = simplejpeg.decode_jpeg(
    bytestring, 
    colorspace=colorspace,
  ).ravel()
  return data.reshape(shape, order='F')

def decode_png(bytestring: bytes, shape, dtype):
  img = pyspng.load(bytestring).reshape(-1)
  img = img.astype(dtype, copy=False)
  return img.reshape(shape, order='F')

def decode_raw(bytestring, shape, dtype):
  return np.frombuffer(bytestring, dtype=dtype).reshape(shape, order='F')

def decode_compressed_segmentation(bytestring, shape, dtype, block_size):
  if block_size is None:
    raise ValueError("block_size parameter must not be None.")

  return cseg.decompress(bytes(bytestring), shape, dtype, block_size, order='F')

def labels(
  filedata:bytes, encoding:str, 
  shape=None, dtype=None, 
  block_size=None, background_color:int = 0
) -> np.ndarray:
  """
  Extract unique labels from a chunk using
  the most efficient means possible for the
  encoding type.

  Returns: numpy array of unique values
  """
  check_installed(encoding)

  if filedata is None or len(filedata) == 0:
    return np.zeros((0,), dtype=dtype)
  elif encoding == "raw":
    img = decode(filedata, encoding, shape, dtype, block_size, background_color)
    return fastremap.unique(img)
  elif encoding == "compressed_segmentation":
    return cseg.labels(
      filedata, shape=shape[:3], 
      dtype=dtype, block_size=block_size
    )
  elif encoding == "compresso":
    return compresso.labels(filedata)
  elif encoding == "crackle":
    return crackle.labels(filedata)
  else:
    raise NotImplementedError(f"Encoding {encoding} is not supported. Try: raw, compressed_segmentation, or compresso.")

def remap(
  filedata:bytes, encoding:str, 
  mapping:Dict[int,int],
  preserve_missing_labels=False,
  shape=None, dtype=None,
  block_size=None
) -> bytes:
  check_installed(encoding)

  if filedata is None or len(filedata) == 0:
    return filedata
  elif encoding == "compressed_segmentation":
    return cseg.remap(
      filedata, shape, dtype, mapping, 
      preserve_missing_labels=preserve_missing_labels, block_size=block_size
    )
  elif encoding == "compresso":
    return compresso.remap(filedata, mapping, preserve_missing_labels=preserve_missing_labels)
  elif encoding == "crackle":
    return crackle.remap(filedata, mapping, preserve_missing_labels=preserve_missing_labels)
  else:
    img = decode(filedata, encoding, shape, dtype, block_size)
    fastremap.remap(img, mapping, preserve_missing_labels=preserve_missing_labels, in_place=True)
    return encode(img, encoding, block_size)

def read_voxel(
  xyz:Sequence[int], 
  filedata:bytes, 
  encoding:str, 
  shape:Optional[Sequence[int]] = None, 
  dtype:Any = None, 
  block_size:Optional[Sequence[int]] = None,
  background_color:int = 0
) -> np.ndarray:
  check_installed(encoding)

  if encoding == "compressed_segmentation":
    arr = cseg.CompressedSegmentationArray(
      filedata, shape=shape[:3], dtype=dtype, block_size=block_size
    )
    out = np.empty((1,1,1,1), dtype=dtype, order="F")
    out[0,0,0,0] = arr[tuple(xyz)]
    return out
  elif encoding == "compresso":
    arr = compresso.CompressoArray(filedata)
    out = np.empty((1,1,1,1), dtype=dtype, order="F")
    out[0,0,0,0] = arr[tuple(xyz)]
    return out
  elif encoding == "crackle":
    arr = crackle.CrackleArray(filedata)
    out = np.empty((1,1,1,1), dtype=dtype, order="F")
    out[0,0,0,0] = arr[tuple(xyz)]
    return out
  else:
    img = decode(filedata, encoding, shape, dtype, block_size, background_color)
    return img[tuple(xyz)][:, np.newaxis, np.newaxis, np.newaxis]

def contains(
  filedata:bytes,
  label:int,
  encoding:str, 
  shape:Optional[Sequence[int]] = None, 
  dtype:Any = None, 
  block_size:Optional[Sequence[int]] = DEFAULT_CSEG_BLOCK_SIZE,
) -> bool:
  check_installed(encoding)

  if encoding == "compressed_segmentation":
    arr = cseg.CompressedSegmentationArray(
      filedata, shape=shape[:3], dtype=dtype, block_size=block_size
    )
    return label in arr
  elif encoding == "compresso":
    arr = compresso.CompressoArray(filedata)
    return label in arr
  elif encoding == "crackle":
    arr = crackle.CrackleArray(filedata)
    return label in arr
  else:
    arr = decode(filedata, encoding, shape, dtype, block_size, 0)
    return bool(np.isin(label, arr))

def transcode(
  image_chunks:Union[
    Dict[Union[str,int], bytes], # { label: binary }
    Sequence[Tuple[Union[str,int], bytes]] # ( (label, binary) for label, binary in ... )
  ], 
  src_encoding:str, 
  dest_encoding:str,
  chunk_size_fn:Callable[[Union[str,int]], ShapeType],
  dtype:Union[str,np.dtype],
  background_color:int = 0,
  progress:bool = False,
  in_place:bool = False,
  src_block_size:ShapeType = DEFAULT_CSEG_BLOCK_SIZE,
  dest_block_size:ShapeType = DEFAULT_CSEG_BLOCK_SIZE,
  compression_params:dict = {},
  force:bool = False,
  total:Optional[int] = None,
):
  """
  Convert one image encoding into another in the most efficient way
  available.

  image_chunks: {
    # where chunkid can be an integer (shards) or a path (unsharded)
    chunk_id: binary,
    ...
  }
  src_encoding: the binary's current encoding (e.g. "raw", "jpeg", "compressed_segmentation", etc.)
  dest_encoding: the desired encoding
  chunk_size_fn: function that takes the chunks's ID and returns its chunk_size
    This is a constant for sharded format, and the path->chunk_size for unsharded.
  dtype: data type of both source and dest
  background_color: what to color missing chunks
  progress: display progress bar
  in_place: it's okay to modify the data in the original dict
  src_block_size/dest: parameters for compressed_segentation type. can be ignored
    for other types.
  compression_params: additional params, especially "level" to configure, e.g. 
    png, jpeg, jpegxl, zfpc, etc compression levels.
  force: perform compression even if the destination type matches
    (useful for debugging or altering compression level)

  Yields (label, binary)
  """
  if isinstance(image_chunks, dict):
    inner_itr = image_chunks.items()
  else:
    inner_itr = image_chunks

  check_installed(src_encoding)
  check_installed(dest_encoding)

  if total is None and hasattr(image_chunks, "__len__"):
    total = len(image_chunks)

  itr = tqdm(inner_itr, disable=(not progress), desc="Transcoding", total=total)

  if src_encoding.lower() == dest_encoding.lower() and not force:
    yield from itr
  elif src_encoding == "jpeg" and dest_encoding == "jpegxl":
    from imagecodecs import jpegxl_encode_jpeg

    for label, binary in itr:
      new_binary = jpegxl_encode_jpeg(binary)

      yield (label, new_binary)
  elif src_encoding == "jpegxl" and dest_encoding == "jpeg":
    from imagecodecs import jpegxl_decode_jpeg

    for label, binary in itr:
      new_binary = jpegxl_decode_jpeg(binary)
      yield (label, new_binary)
  else:
    for label, binary in itr:
      image = decode(
        binary, 
        encoding=src_encoding,
        shape=chunk_size_fn(label),
        dtype=dtype,
        block_size=src_block_size,
        background_color=background_color,
      )
      while image.ndim < 4:
        image = image[..., np.newaxis]
      new_binary = encode(
        image, 
        encoding=dest_encoding,
        block_size=dest_block_size,
        compression_params=compression_params,
      )
      yield (label, new_binary)

