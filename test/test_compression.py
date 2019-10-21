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
