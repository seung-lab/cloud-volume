import pytest

import sys

import numpy as np

from cloudvolume.compression import compress, decompress

SUPPORTED_PYTHON_VERSION = (sys.version_info[0] == 3)

def test_gzip():
	for N in range(100):
		flts = np.array(range(N), dtype=np.float32).reshape( (N,1,1,1) ).tostring()
		compressed = compress(flts, 'gzip')
		assert compressed != flts
		decompressed = decompress(compressed, 'gzip')
		assert decompressed == flts


def test_fpzip():
	# extension only supports python 3
	if not SUPPORTED_PYTHON_VERSION:
		return
	
	for N in range(100):
		flts = np.array(range(N), dtype=np.float32).reshape( (N,1,1,1) )
		compressed = compress(flts, 'fpzip')
		assert compressed != flts
		decompressed = decompress(compressed, 'fpzip')
		assert np.all(decompressed == flts)

	for N in range(0, 200, 2):
		flts = np.array(range(N), dtype=np.float32).reshape( (N // 2, 2, 1, 1) )
		compressed = compress(flts, 'fpzip')
		assert compressed != flts
		decompressed = decompress(compressed, 'fpzip')
		assert np.all(decompressed == flts)
	