import pytest

import numpy as np

from cloudvolume.chunks import encode, decode

def encode_decode(data, format):
  encoded = encode(data, format)
  result = decode(encoded, format, shape=(64,64,64,1), dtype=np.uint8)

  assert np.all(result.shape == data.shape)
  assert np.all(data == result)


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

