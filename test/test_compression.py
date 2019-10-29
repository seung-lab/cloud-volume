import pytest

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

@pytest.mark.parametrize("compression_method", ("gzip", "br"))
def test_compress_level(compression_method):
  N = 10000
  x = np.array(range(N), dtype=np.float32).reshape( (N, 1, 1, 1) )
  content = np.ascontiguousarray(x, dtype=np.float32).tostring()

  compr_rate = []
  compress_levels = (1, 8)
  for compress_level in compress_levels:
    print(compress_level)
    compressed = compress(content, compression_method, compress_level=compress_level)

    assert compressed != content
    decompressed = decompress(compressed, compression_method)
    assert decompressed == content

    compr_rate.append(float(len(compressed)) / float(len(content)))

  # make sure we get better compression at highest level than lowest level
  assert compr_rate[-1] < compr_rate[0]