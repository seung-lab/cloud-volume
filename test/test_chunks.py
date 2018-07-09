import pytest

import sys

import numpy as np

from cloudvolume.chunks import encode, decode

SUPPORTED_PYTHON_VERSION = (sys.version_info[0] == 3)

def encode_decode(data, format):
  encoded = encode(data, format)
  result = decode(encoded, format, shape=(64,64,64,1), dtype=np.uint8)

  assert np.all(result.shape == data.shape)
  assert np.all(data == result)

def test_kempression():
  # fpzip extension only supports python 3
  if not SUPPORTED_PYTHON_VERSION:
    return
  data = np.random.random_sample(size=1024 * 3).reshape( (64, 4, 4, 3) ).astype(np.float32)
  encoded = encode(data, 'kempressed')
  result = decode(encoded, 'kempressed', shape=(64, 4, 4, 3), dtype=np.float32)
  assert np.all(result.shape == data.shape)
  assert np.all(np.abs(data - result) <= np.finfo(np.float32).eps)

def test_compressed_segmentation():
  data = np.random.randint(255, size=(64,64,64,1), dtype=np.uint32)
  encoded = encode(data, 'compressed_segmentation', block_size=(8,8,8))

  compressed = np.frombuffer(encoded, dtype=np.uint32)

  assert compressed[0] == 1 # one channel

  # at least check headers for integrity
  # 64 bit block header 
  # encoded bits (8 bit), lookup table offset (24 bit), encodedValuesOffset (32)
  for i in range(8*8*8):
    assert ((compressed[2*i + 1] & 0xff000000) >> 24) in (0,1,2,4,8,16,32) 
    assert (compressed[2*i + 1] & 0x00ffffff) < len(compressed)
    assert compressed[2*i + 2] < len(compressed)

  result = decode(encoded, 'compressed_segmentation', 
    shape=(64,64,64,1),
    dtype=np.uint32,
    block_size=(8,8,8)) 

  data = np.asfortranarray(data)
  assert np.all(data == result)

def test_fpzip():
  # fpzip extension only supports python 3
  if not SUPPORTED_PYTHON_VERSION:
    return
  
  for N in range(100):
    flts = np.array(range(N), dtype=np.float32).reshape( (N,1,1,1) )
    compressed = encode(flts, 'fpzip')
    assert compressed != flts
    decompressed = decode(compressed, 'fpzip')
    assert np.all(decompressed == flts)

  for N in range(0, 200, 2):
    flts = np.array(range(N), dtype=np.float32).reshape( (N // 2, 2, 1, 1) )
    compressed = encode(flts, 'fpzip')
    assert compressed != flts
    decompressed = decode(compressed, 'fpzip')
    assert np.all(decompressed == flts)

def test_raw():
  random_data = np.random.randint(255, size=(64,64,64,1), dtype=np.uint8)
  encode_decode(random_data, 'raw')

def test_npz():
  random_data = np.random.randint(255, size=(64,64,64,1), dtype=np.uint8)
  encode_decode(random_data, 'npz')

def test_jpeg():
  data = np.zeros(shape=(64,64,64,1), dtype=np.uint8)
  encode_decode(data, 'jpeg')
  encode_decode(data + 255, 'jpeg')

  # Random jpeg won't decompress to exactly the same image
  # but it should have nearly the same average power
  random_data = np.random.randint(255, size=(64,64,64,1), dtype=np.uint8)
  pre_avg = random_data.copy().flatten().mean()
  encoded = encode(random_data, 'jpeg')
  decoded = decode(encoded, 'jpeg', shape=(64,64,64,1), dtype=np.uint8)
  post_avg = decoded.copy().flatten().mean()

  assert abs(pre_avg - post_avg) < 1

