import pytest

import sys

import numpy as np

from cloudvolume.compression import compress, decompress

def test_gzip():
  for N in range(100):
    flts = np.array(range(N), dtype=np.float32).reshape( (N,1,1,1) ).tostring()
    compressed = compress(flts, 'gzip')
    assert compressed != flts
    decompressed = decompress(compressed, 'gzip')
    assert decompressed == flts

  