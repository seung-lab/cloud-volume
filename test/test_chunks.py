import pytest

import sys

import numpy as np

from cloudvolume.chunks import encode, decode
from cloudvolume import chunks

def encode_decode(data, format, shape=(64,64,64), num_chan=1, level=None):
  compression_params = { "level": level }
  encoded = encode(data, format, compression_params)
  result = decode(encoded, format, shape=list(shape) + [ num_chan ], dtype=np.uint8)

  assert np.all(result.shape == data.shape)
  assert np.all(data == result)

def test_kempression():
  data = np.random.random_sample(size=1024 * 3).reshape( (64, 4, 4, 3) ).astype(np.float32)
  encoded = encode(data, 'kempressed')
  result = decode(encoded, 'kempressed', shape=(64, 4, 4, 3), dtype=np.float32)
  assert np.all(result.shape == data.shape)
  assert np.all(np.abs(data - result) <= np.finfo(np.float32).eps)

def test_compressed_segmentation():
  def run_test(shape, block_size, accelerated):
    data = np.random.randint(255, size=shape, dtype=np.uint32)
    encoded = chunks.encode_compressed_segmentation(data, block_size, accelerated)

    compressed = np.frombuffer(encoded, dtype=np.uint32)

    assert compressed[0] == 1 # one channel

    # at least check headers for integrity
    # 64 bit block header 
    # encoded bits (8 bit), lookup table offset (24 bit), encodedValuesOffset (32)
    grid = np.ceil(np.array(shape[3:], dtype=np.float32) / np.array(block_size, dtype=np.float32))
    grid = grid.astype(np.uint32)
    for i in range(np.prod(grid)):
      encodedbits = (compressed[2*i + 1] & 0xff000000) >> 24
      table_offset = compressed[2*i + 1] & 0x00ffffff
      encoded_offset = compressed[2*i + 2]

      assert encodedbits in (0, 1, 2, 4, 8, 16, 32)
      assert table_offset < len(compressed)
      assert encoded_offset < len(compressed)

    result = chunks.decode_compressed_segmentation(encoded, 
      shape=shape,
      dtype=np.uint32,
      block_size=block_size,
      accelerated=accelerated,
    ) 

    assert np.all(data == result)

  try:
    import _compressed_segmentation
    test_options = (True, False)
  except:
    test_options = (False,)

  for accelerated in test_options:
    run_test( ( 2, 2, 2, 1), (2,2,2), accelerated )
    run_test( ( 1, 2, 2, 1), (2,2,2), accelerated )
    run_test( ( 2, 1, 2, 1), (2,2,2), accelerated )
    run_test( ( 2, 2, 1, 1), (2,2,2), accelerated )
    run_test( (64,64,64,1), (8,8,8), accelerated )
    run_test( (16,16,16,1), (8,8,8), accelerated )
    run_test( (8,8,8,1), (8,8,8), accelerated )
    run_test( (4,4,4,1), (8,8,8), accelerated )
    run_test( (4,4,4,1), (2,2,2), accelerated )
    run_test( (2,4,4,1), (2,2,2), accelerated )
  
  if True in test_options:
    run_test( (10,8,8,1), (10,8,8), True ) # known bug in pure python verison

def test_fpzip():
  for N in range(0,100):
    flts = np.array(range(N), dtype=np.float32).reshape( (N,1,1,1) )
    compressed = encode(flts, 'fpzip')
    assert isinstance(compressed, bytes)
    decompressed = decode(compressed, 'fpzip')
    assert np.all(decompressed == flts)

  for N in range(0, 200, 2):
    flts = np.array(range(N), dtype=np.float32).reshape( (N // 2, 2, 1, 1) )
    compressed = encode(flts, 'fpzip')
    assert isinstance(compressed, bytes)
    decompressed = decode(compressed, 'fpzip')
    assert np.all(decompressed == flts)

def test_raw():
  random_data = np.random.randint(255, size=(64,64,64,1), dtype=np.uint8)
  encode_decode(random_data, 'raw')

@pytest.mark.parametrize('dtype', (np.uint8, np.uint16, np.uint32, np.uint64))
def test_compresso(dtype):
  random_data = np.random.randint(255, size=(64,64,64,1), dtype=dtype)
  encode_decode(random_data, 'compresso')

@pytest.mark.parametrize('dtype', (np.uint8, np.uint16, np.uint32, np.uint64))
def test_crackle(dtype):
  random_data = np.random.randint(255, size=(64,64,64,1), dtype=dtype)
  encode_decode(random_data, 'crackle')

def test_npz():
  random_data = np.random.randint(255, size=(64,64,64,1), dtype=np.uint8)
  encode_decode(random_data, 'npz')

@pytest.mark.parametrize("level", (None,0,5,9))
def test_png(level):
  size = [64,64,64]
  random_data = np.random.randint(255, size=size + [1], dtype=np.uint8)
  encode_decode(random_data, 'png', shape=size, level=level)

@pytest.mark.parametrize("shape", ( (64,64,64), (64,61,50), (128,128,16), ))
@pytest.mark.parametrize("num_channels", (1,3))
@pytest.mark.parametrize("quality", (None,85,75))
def test_jpeg(shape, num_channels, quality):
  import simplejpeg

  xshape = list(shape) + [ num_channels ]
  data = np.zeros(shape=xshape, dtype=np.uint8)
  encode_decode(data, 'jpeg', shape, num_channels, level=quality)
  encode_decode(data + 255, 'jpeg', shape, num_channels, level=quality)

  jpg = simplejpeg.decode_jpeg(
    encode(data, 'jpeg', compression_params={ "level": quality }), 
    colorspace="GRAY",
  )
  assert jpg.shape[0] == shape[1] * shape[2]
  assert jpg.shape[1] == shape[0]

  # Random jpeg won't decompress to exactly the same image
  # but it should have nearly the same average power
  random_data = np.random.randint(255, size=xshape, dtype=np.uint8)
  pre_avg = random_data.copy().flatten().mean()
  encoded = encode(random_data, 'jpeg')
  decoded = decode(encoded, 'jpeg', shape=xshape, dtype=np.uint8)
  post_avg = decoded.copy().flatten().mean()

  assert abs(pre_avg - post_avg) < 1


@pytest.mark.parametrize("encoding", [
  "raw", "compressed_segmentation", "compresso", "crackle"
])
def test_contains(encoding):
  labels = np.arange(10*10*10, dtype=np.uint64).reshape([10,10,10,1], order="F")
  binary = encode(labels, encoding, [8,8,8])

  print(binary)

  testfn = lambda seg: chunks.contains(binary, seg, encoding, labels.shape, labels.dtype)

  assert testfn(50)
  assert testfn(0)
  assert testfn(1000) == False
  assert testfn(9124124) == False
  assert testfn(800)








