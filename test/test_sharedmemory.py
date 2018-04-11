import os

import numpy as np
import posix_ipc
import psutil

import cloudvolume.sharedmemory as shm

def test_ndarray():
	location = 'cloudvolume-shm-test-ndarray'
	array_like, array = shm.ndarray(shape=(2,2,2), dtype=np.uint8, location=location)
	assert np.all(array == np.zeros(shape=(2,2,2), dtype=np.uint8))
	array[:] = 100
	array_like.close()

	array_like, array = shm.ndarray(shape=(2,2,2), dtype=np.uint8, location=location)
	assert np.all(array[:] == 100)
	array_like.close()

	assert os.path.exists('/dev/shm/' + location)
	assert os.path.getsize('/dev/shm/' + location) == 8

	assert shm.unlink(location) == True
	assert shm.unlink(location) == False

	assert not os.path.exists('/dev/shm/' + location)

	available = psutil.virtual_memory().available
	array_like, array = shm.ndarray(shape=(available // 10,2,2), dtype=np.uint8, location=location)
	array_like.close()
	try:
		array_like, array = shm.ndarray(shape=(available,2,2), dtype=np.uint8, location=location)
		assert False
	except shm.MemoryAllocationError:
		pass

	assert shm.unlink(location) == True
	assert shm.unlink(location) == False
