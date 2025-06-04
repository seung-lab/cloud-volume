import pytest

import os
import shutil

import zarr
from cloudvolume import CloudVolume, Bbox

import numpy as np

TEST_DIR = os.path.dirname("/tmp/removeme/cloudvolume/")

def create_simple_dataset():
    test_location = os.path.join(TEST_DIR, "zarr3_simple_unsharded.zarr")

    if os.path.exists(test_location):
        shutil.rmtree(test_location)

    shape = [1000, 1000, 50]
    data = np.ones(shape, dtype=np.uint8, order="C")

    arr = zarr.open(store=test_location, shape=shape, chunks=(100, 100, 10), dtype='uint8', mode='w')
    arr[:] = data
    arr.store.close()

    return test_location

def create_simple_ragged_dataset():
    test_location = os.path.join(TEST_DIR, "zarr3_simple_unsharded.zarr")

    if os.path.exists(test_location):
        shutil.rmtree(test_location)

    shape = [1005, 1003, 51]
    data = np.ones(shape, dtype=np.uint8, order="C")

    arr = zarr.open(store=test_location, shape=shape, chunks=(100, 100, 10), dtype='uint8', mode='w')
    arr[:] = data
    arr.store.close()

    return test_location

def test_zarr3_unsharded_read_write():
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

    test_location = create_simple_ragged_dataset()

    cv = CloudVolume(f"zarr3://file://{test_location}", fill_missing=False)

    # by default zarr3 in C order [z,y,x], so inverting axes
    # results in a full transposition [x,y,z]
    assert np.all(cv[:] == 1)

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

    binimg = cv.download(cv.bounds, label=4)
    assert np.all(binimg == 1)
    assert binimg.dtype == bool

    shutil.rmtree(test_location)

def test_zarr3_exists():
    simple_dataset_loc = create_simple_dataset()

    chunknames = os.listdir(os.path.join(simple_dataset_loc, 'c', '0', '0'))
    chunknames.sort()
    assert chunknames == ['0', '1', '2', '3', '4']

    cv = CloudVolume("zarr3://file://" + simple_dataset_loc)
    res = cv.exists(cv.bounds)

    paths = []
    for z in range(10):
        for y in range(10):
            for x in range(5):
                paths.append(os.path.join("c", str(z), str(y), str(x)))

    ans_paths = set(paths)
    res_paths = set(res.keys())

    assert ans_paths == res_paths

    shutil.rmtree(simple_dataset_loc)

def test_zarr3_delete_all():

    simple_dataset_loc = create_simple_dataset()

    arr = zarr.open(store=simple_dataset_loc)

    chunknames = os.listdir(os.path.join(simple_dataset_loc, 'c', '0', '0'))
    chunknames.sort()
    assert chunknames == ['0', '1', '2', '3', '4']

    cv = CloudVolume("zarr3://file://" + simple_dataset_loc)
    cv.delete(cv.bounds)

    for i in range(10):
        chunknames = os.listdir(os.path.join(simple_dataset_loc, 'c', '0', str(i)))
        chunknames.sort()
        assert chunknames == []

    shutil.rmtree(simple_dataset_loc)

def test_zarr3_delete_some():
    simple_dataset_loc = create_simple_dataset()
    arr = zarr.open(store=simple_dataset_loc)

    chunknames = os.listdir(os.path.join(simple_dataset_loc, 'c', '0', '0'))
    chunknames.sort()
    assert chunknames == [ str(i) for i in range(0,5) ]

    cv = CloudVolume("zarr3://file://" + simple_dataset_loc)
    bbx = cv.bounds.clone()
    bbx.maxpt.x = 10
    cv.delete(bbx)

    for i in range(10):
        chunknames = os.listdir(os.path.join(simple_dataset_loc, 'c', '0', str(i)))
        chunknames.sort()
        assert chunknames == [ str(i) for i in range(1,5) ]

    shutil.rmtree(simple_dataset_loc)

def test_zarr3_transfer_to():
    simple_dataset_loc = create_simple_dataset()

    precomputed_loc = os.path.join(TEST_DIR, "precomputed_simple_unsharded.zarr")

    cv_zarr = CloudVolume("zarr3://file://" + simple_dataset_loc)
    cv_zarr.transfer_to("precomputed://file://" + precomputed_loc, cv_zarr.bounds)

    cv_precomputed = CloudVolume("precomputed://file://" + precomputed_loc)

    assert np.all(cv_zarr[:] == cv_precomputed[:])

    shutil.rmtree(simple_dataset_loc)
    shutil.rmtree(precomputed_loc)


def test_zarr3_metadata_modification():
    simple_dataset_loc = create_simple_dataset()

    cv_zarr = CloudVolume("zarr3://file://" + simple_dataset_loc)

    assert all(cv_zarr.chunk_size == [10,100,100])

    cv_zarr.scale["chunk_sizes"] = [[25,200,200]]
    cv_zarr.commit_info()

    cv_zarr.refresh_info()

    assert all(cv_zarr.chunk_size == [25,200,200])


def test_zarr3_transfer_from_precomputed():
    loc = os.path.join(TEST_DIR, "precomputed_simple_unsharded.precomputed")
    loc2 = os.path.join(TEST_DIR, "zarr_simple_unsharded.zarr")

    volume_size = [ 1003, 1001, 105 ]

    info = CloudVolume.create_new_info(
        num_channels    = 1,
        layer_type      = 'image',
        data_type       = 'uint8', 
        encoding        = 'raw', 
        resolution      = [4, 4, 40], 
        voxel_offset    = [5, 5, 5],
        chunk_size      = [ 100, 100, 10 ], 
        volume_size     = volume_size,
    )

    cv = CloudVolume(f"precomputed://file://{loc}", info=info)
    cv.commit_info()
    data = np.random.randint(0, 255, size=volume_size, dtype=np.uint8)
    cv[:] = data
    
    zarr_cv = cv.transfer_to(f"zarr3://file://{loc2}", cv.bounds)

    assert cv.meta.path.format == "precomputed"
    assert zarr_cv.meta.path.format == "zarr3"

    assert np.all(zarr_cv[:][...,0] == data)

    shutil.rmtree(loc)
    shutil.rmtree(loc2)
    







