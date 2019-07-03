import os
import sys

import numpy as np

import cloudvolume.sharedmemory as shm

def test_ndarray_fs():
	location = 'cloudvolume-shm-test-ndarray'
	array_like, array = shm.ndarray_fs(
		shape=(2,2,2), dtype=np.uint8, location=location, 
		lock=None, emulate_shm=True
	)
	assert np.all(array == np.zeros(shape=(2,2,2), dtype=np.uint8))
	array[:] = 100
	array_like.close()

	array_like, array = shm.ndarray_fs(
		shape=(2,2,2), dtype=np.uint8, location=location, 
		lock=None, emulate_shm=True
	)
	assert np.all(array[:] == 100)
	array_like.close()

	full_loc = os.path.join(shm.EMULATED_SHM_DIRECTORY, location)

	assert os.path.exists(full_loc)
	assert os.path.getsize(full_loc) == 8

	assert shm.unlink_fs(location) == True
	assert shm.unlink_fs(location) == False

	try:
		array_like, array = shm.ndarray_fs(
			shape=(2,2,2), dtype=np.uint8, location=location, 
			lock=None, readonly=True, emulate_shm=True
		)
		assert False
	except shm.SharedMemoryReadError:
		pass

	array_like, array = shm.ndarray_fs(
		shape=(2,2,2), dtype=np.uint8, location=location, 
		lock=None, emulate_shm=True
	)
	try:
		array_like, array = shm.ndarray_fs(
			shape=(200,200,200), dtype=np.uint8, location=location, 
			lock=None, readonly=True, emulate_shm=True
		)
		assert False
	except shm.SharedMemoryReadError:
		pass

	assert shm.unlink_fs(location) == True
	assert shm.unlink_fs(location) == False

	assert not os.path.exists(full_loc)

def test_ndarray_sh():
	# Don't bother testing on unsupported platforms.
	if shm.EMULATE_SHM:
		return

	import psutil

	location = 'cloudvolume-shm-test-ndarray'
	array_like, array = shm.ndarray_shm(shape=(2,2,2), dtype=np.uint8, location=location)
	assert np.all(array == np.zeros(shape=(2,2,2), dtype=np.uint8))
	array[:] = 100
	array_like.close()

	array_like, array = shm.ndarray_shm(shape=(2,2,2), dtype=np.uint8, location=location)
	assert np.all(array[:] == 100)
	array_like.close()

	filename = os.path.join(shm.SHM_DIRECTORY, location)

	assert os.path.exists(filename)
	assert os.path.getsize(filename) == 8

	assert shm.unlink_shm(location) == True
	assert shm.unlink_shm(location) == False

	assert not os.path.exists(filename)

	available = psutil.virtual_memory().available
	array_like, array = shm.ndarray_shm(shape=(available // 10,2,2), dtype=np.uint8, location=location)
	array_like.close()
	try:
		array_like, array = shm.ndarray_shm(shape=(available,2,2), dtype=np.uint8, location=location)
		assert False
	except shm.SharedMemoryAllocationError:
		pass

	assert shm.unlink_shm(location) == True
	assert shm.unlink_shm(location) == False

	try:
		array_like, array = shm.ndarray_shm(shape=(2,2,2), dtype=np.uint8, location=location, readonly=True)
		assert False
	except shm.SharedMemoryReadError:
		pass

	array_like, array = shm.ndarray_shm(shape=(2,2,2), dtype=np.uint8, location=location)
	try:
		array_like, array = shm.ndarray_shm(shape=(200,200,200), dtype=np.uint8, location=location, readonly=True)
		assert False
	except shm.SharedMemoryReadError:
		pass

	assert shm.unlink_shm(location) == True
	assert shm.unlink_shm(location) == False
