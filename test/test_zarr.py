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
    data = np.zeros(shape, dtype=np.uint8, order="C")
    data[:20] = 1

    arr = zarr.open(store=test_location, shape=shape, chunks=(100, 100, 10), dtype='uint8', mode='w')
    arr[:] = data
    arr.store.close()

    cv = CloudVolume(f"zarr3://file://{test_location}", fill_missing=True)

    # by default zarr3 in C order [z,y,x], so inverting axes
    # results in a full transposition [x,y,z]
    assert np.all(cv[:][...,0] == data.T)

    cv[:] = 2

    arr = zarr.open(store=test_location, shape=shape, chunks=(100, 100, 10), dtype='uint8', mode='r')
    assert np.all(arr[:] == 2)
    assert np.all(cv[:] == 2)

    shutil.rmtree(test_location)

def test_zarr3_crc():
    test_location = os.path.join(TEST_DIR, "zarr_unsharded_crc.zarr")

    shape = [100, 100, 50]
    data = np.zeros(shape, dtype=np.uint8, order="C")
    data[:20] = 1

    arr = zarr.open(store=test_location, shape=shape, chunks=(100, 100, 10), dtype='uint8', mode='w')
    arr[:] = data
    arr.store.close()

    cv = CloudVolume(f"zarr3://file://{test_location}", fill_missing=True)

    codecs = cv.meta.codecs(0)
    codecs.append({
        "name": "crc32c",
        "configuration": { "endian": "little" },
    })

    cv[:] = 3

    assert np.all(cv[:] == 3)

    shutil.rmtree(test_location)

def test_zarr3_blosc():
    test_location = os.path.join(TEST_DIR, "zarr_unsharded_blosc.zarr")

    shape = [100, 100, 100]
    data = np.zeros(shape, dtype=np.uint8, order="C")
    data[:20] = 1

    arr = zarr.open(store=test_location, shape=shape, chunks=(20, 20, 20), dtype='uint8', mode='w')
    arr[:] = data
    arr.store.close()

    cv = CloudVolume(f"zarr3://file://{test_location}", fill_missing=True)

    codecs = cv.meta.codecs(0)

    codecs[1] = {
        "name": "blosc",
        "configuration": {
            "cname": "lz4",
            "clevel": 1,
            "shuffle": "shuffle",
            "typesize": 4,
            "blocksize": 0
        }
    }
    while len(codecs) > 1:
        codecs.pop()

    cv.commit_info()

    cv = CloudVolume(f"zarr3://file://{test_location}", fill_missing=True)
    cv[:] = 4

    assert np.all(cv[:] == 4)

    shutil.rmtree(test_location)





