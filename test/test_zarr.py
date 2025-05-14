import pytest

import os
import shutil

import zarr
from cloudvolume import CloudVolume

import numpy as np

TEST_DIR = os.path.dirname("/tmp/removeme/cloudvolume/")

def test_zarr3_unsharded():
	test_location = os.path.join(TEST_DIR, "zarr_unsharded.zarr")

	shape = [1000, 1000, 50]
	data = np.zeros(shape, dtype=np.uint8)
	data[:20] = 1

	arr = zarr.open(store=test_location, shape=shape, chunks=(10, 10, 10), dtype='uint8', mode='w')
	arr[:] = data
	arr.store.close()

	cv = CloudVolume(f"zarr3://file://{test_location}", fill_missing=True)

	assert np.all(cv[:][...,0] == data.T)


