import pytest

import sys

import numpy as np

from cloudvolume.compression import compress, decompress

@pytest.mark.parametrize("compression_method", ("gzip", "br"))
def test_compression(compression_method):
  for N in range(100):
    flts = np.array(range(N), dtype=np.float32).reshape( (N,1,1,1) ).tostring()
    compressed = compress(flts, compression_method)
    assert compressed != flts
    decompressed = decompress(compressed, compression_method)
    assert decompressed == flts

def test_br_compress_level():
  N=10000
  x = np.array(range(N), dtype=np.float32).reshape( (N,1,1,1) )
  content = np.ascontiguousarray(x, dtype=np.float32).tostring()

  compr_rate = []
  compress_levels = range(1, 7, 2)
  for compress_level in compress_levels:
    compressed = compress(content, "br", compress_level=compress_level)
    
    assert compressed != content
    decompressed = decompress(compressed, "br")
    assert decompressed == content

    compr_rate.append(len(compressed) / len(content))

  # make sure we get better compression at highest level than lowest level
  assert compr_rate[-1] < compr_rate[0]

  # make sure we dont get worse compr rates with each level
  assert all(x >= y for x, y in zip(compr_rate, compr_rate[1:]))