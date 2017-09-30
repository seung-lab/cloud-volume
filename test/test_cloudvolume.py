import pytest

import json
import os
import numpy as np
import shutil
import gzip
import json

from cloudvolume import CloudVolume, chunks, Storage
from cloudvolume.lib import mkdir, Bbox
from layer_harness import (
    TEST_NUMBER, create_image, 
    delete_layer, create_layer,
    create_volume_from_image
)

def test_aligned_read():
    delete_layer()
    cv, data = create_layer(size=(50,50,50,1), offset=(0,0,0))
    # the last dimension is the number of channels
    assert cv[0:50,0:50,0:50].shape == (50,50,50,1)
    assert np.all(cv[0:50,0:50,0:50] == data)
    
    delete_layer()
    cv, data = create_layer(size=(128,64,64,1), offset=(0,0,0))
    # the last dimension is the number of channels
    assert cv[0:64,0:64,0:64].shape == (64,64,64,1) 
    assert np.all(cv[0:64,0:64,0:64] ==  data[:64,:64,:64,:])

    delete_layer()
    cv, data = create_layer(size=(128,64,64,1), offset=(10,20,0))
    cutout = cv[10:74,20:84,0:64]
    # the last dimension is the number of channels
    assert cutout.shape == (64,64,64,1) 
    assert np.all(cutout == data[:64,:64,:64,:])
    # get the second chunk
    cutout2 = cv[74:138,20:84,0:64]
    assert cutout2.shape == (64,64,64,1) 
    assert np.all(cutout2 == data[64:128,:64,:64,:])


def test_non_aligned_read():
    delete_layer()
    cv, data = create_layer(size=(128,64,64,1), offset=(0,0,0))

    # the last dimension is the number of channels
    assert cv[31:65,0:64,0:64].shape == (34,64,64,1) 
    assert np.all(cv[31:65,0:64,0:64] == data[31:65,:64,:64,:])

    # read a single pixel
    delete_layer()
    cv, data = create_layer(size=(64,64,64,1), offset=(0,0,0))
    # the last dimension is the number of channels
    assert cv[22:23,22:23,22:23].shape == (1,1,1,1) 
    assert np.all(cv[22:23,22:23,22:23] == data[22:23,22:23,22:23,:])

    # Test steps (negative steps are not supported)
    img1 = cv[::2, ::2, ::2, :]
    img2 = cv[:, :, :, :][::2, ::2, ::2, :]
    assert np.array_equal(img1, img2)

def test_write():
    delete_layer()
    cv, data = create_layer(size=(50,50,50,1), offset=(0,0,0))

    replacement_data = np.zeros(shape=(50,50,50,1), dtype=np.uint8)
    cv[0:50,0:50,0:50] = replacement_data
    assert np.all(cv[0:50,0:50,0:50] == replacement_data)

    replacement_data = np.random.randint(255, size=(50,50,50,1), dtype=np.uint8)
    cv[0:50,0:50,0:50] = replacement_data
    assert np.all(cv[0:50,0:50,0:50] == replacement_data)

    # out of bounds
    delete_layer()
    cv, data = create_layer(size=(128,64,64,1), offset=(10,20,0))
    with pytest.raises(ValueError):
        cv[74:150,20:84,0:64] = np.ones(shape=(64,64,64,1), dtype=np.uint8)
    
    # non-aligned writes
    delete_layer()
    cv, data = create_layer(size=(128,64,64,1), offset=(10,20,0))
    with pytest.raises(ValueError):
        cv[21:85,0:64,0:64] = np.ones(shape=(64,64,64,1), dtype=np.uint8)

def test_writer_last_chunk_smaller():
    """
    we make it believe the last chunk is smaller by hacking the info file
    """
    delete_layer()
    cv, data = create_layer(size=(128,64,64,1), offset=(0,0,0))
    
    chunks = [ chunk for chunk in cv._generate_chunks(data[:100,:,:,:], (0,0,0)) ]

    assert len(chunks) == 2

    img, spt, ept = chunks[0]
    assert np.array_equal(spt, (0,0,0))
    assert np.array_equal(ept, (64,64,64))
    assert img.shape == (64,64,64,1)

    img, spt, ept = chunks[1]
    assert np.array_equal(spt, (64,0,0))
    assert np.array_equal(ept, (100,64,64))
    assert img.shape == (36,64,64,1)

# def test_reader_negative_indexing():
#     """negative indexing is supported"""
#     delete_layer()
#     cv, data = create_layer(size=(128,64,64,1), offset=(0,0,0))

#     # Test negative beginnings
#     img1 = cv[-1:, -1:, -1:, :]
#     img2 = cv[:, :, :, :][-1:, -1:, -1:, :]

#     assert np.array_equal(img1, img2)    

#     # Test negative ends
#     with pytest.raises(ValueError):
#         img1 = cv[::-1, ::-1, ::-1, :]

def test_setitem_mismatch():
    delete_layer()
    cv, data = create_layer(size=(64,64,64,1), offset=(0,0,0))

    with pytest.raises(ValueError):
        cv[0:64,0:64,0:64] = np.zeros(shape=(5,5,5,1), dtype=np.uint8)

def test_bounds():
    delete_layer()
    cv, data = create_layer(size=(128,64,64,1), offset=(100,100,100))
    cv.bounded = True

    try:
        cutout = cv[0:,0:,0:,:]
        cutout = cv[100:229,100:165,100:165,0]
        cutout = cv[99:228,100:164,100:164,0]
    except ValueError:
        pass
    else:
        assert False

    # don't die
    cutout = cv[100:228,100:164,100:164,0]

    cv.bounded = False
    cutout = cv[0:,0:,0:,:]
    assert cutout.shape == (228, 164, 164, 1)

    assert np.count_nonzero(cutout) != 0

    cutout[100:,100:,100:,:] = 0

    assert np.count_nonzero(cutout) == 0

def test_extract_path():
    def shoulderror(url):
        try:
            path = CloudVolume.extract_path(url)
            assert False, url
        except:
            pass

    def okgoogle(url):
        path = CloudVolume.extract_path(url)
        assert path.protocol == 'gs', url
        assert path.bucket == 'bucket', url
        assert path.dataset == 'dataset', url
        assert path.layer == 'layer', url
       

    shoulderror('ou3bouqjsa fkj aojsf oaojf ojsaf')

    okgoogle('gs://bucket/dataset/layer/')
    shoulderror('gs://bucket/dataset/layer/info')

    path = CloudVolume.extract_path('s3://bucketxxxxxx/datasetzzzzz91h8__3/layer1br9bobasjf/')
    assert path.protocol == 's3'
    assert path.bucket == 'bucketxxxxxx'
    assert path.dataset == 'datasetzzzzz91h8__3'
    assert path.layer == 'layer1br9bobasjf'

    path = CloudVolume.extract_path('file://bucket/dataset/layer/')
    assert path.protocol == 'file'
    assert path.bucket == 'bucket'
    assert path.dataset == 'dataset'
    assert path.layer == 'layer'

    shoulderror('lucifer://bucket/dataset/layer/')
    shoulderror('gs://///')
    shoulderror('gs://seunglab-test//segmentation')

    path = CloudVolume.extract_path('file:///tmp/removeme/layer/')
    assert path.protocol == 'file'
    assert path.bucket == '/tmp'
    assert path.dataset == 'removeme'
    assert path.layer == 'layer'

def test_provenance():
    delete_layer()
    cv, data = create_layer(size=(64,64,64,1), offset=(0,0,0))

    provobj = json.loads(cv.provenance.serialize())
    assert provobj == {"sources": [], "owners": [], "processing": [], "description": ""}

    cv.provenance.sources.append('cooldude24@princeton.edu')
    cv.commit_provenance()
    cv.refresh_provenance()

    assert cv.provenance.sources == [ 'cooldude24@princeton.edu' ]

def test_info_provenance_cache():
    image = np.zeros(shape=(128,128,128,1), dtype=np.uint8)
    vol = create_volume_from_image(
        image=image, 
        offset=(0,0,0), 
        layer_path='gs://seunglab-test/cloudvolume/caching', 
        layer_type='image', 
        resolution=(1,1,1), 
        encoding='raw'
    )

    # Test Info
    vol.cache = True
    vol.flush_cache()
    info = vol.refresh_info()
    assert info is not None

    with open(os.path.join(vol.cache_path, 'info'), 'r') as infof:
        info = infof.read()
        info = json.loads(info)

    with open(os.path.join(vol.cache_path, 'info'), 'w') as infof:
        infof.write(json.dumps({ 'wow': 'amaze' }))

    info = vol.refresh_info()
    assert info == { 'wow': 'amaze' }
    vol.cache = False
    info = vol.refresh_info()
    assert info != { 'wow': 'amaze' }

    # Test Provenance
    vol.cache = True
    vol.flush_cache()
    prov = vol.refresh_provenance()
    assert prov is not None

    with open(os.path.join(vol.cache_path, 'provenance'), 'r') as provf:
        prov = provf.read()
        prov = json.loads(prov)

    with open(os.path.join(vol.cache_path, 'provenance'), 'w') as provf:
        prov['description'] = 'wow'
        provf.write(json.dumps(prov))

    prov = vol.refresh_provenance()
    assert prov['description'] == 'wow'
    vol.cache = False
    prov = vol.refresh_provenance()
    assert prov['description'] == ''


def test_caching():
    image = np.zeros(shape=(128,128,128,1), dtype=np.uint8)
    image[0:64,0:64,0:64] = 1
    image[64:128,0:64,0:64] = 2
    image[0:64,64:128,0:64] = 3
    image[0:64,0:64,64:128] = 4
    image[64:128,64:128,0:64] = 5
    image[64:128,0:64,64:128] = 6
    image[0:64,64:128,64:128] = 7
    image[64:128,64:128,64:128] = 8

    dirpath = '/tmp/cloudvolume/caching-volume-' + str(TEST_NUMBER)
    layer_path = 'file://' + dirpath

    vol = create_volume_from_image(
        image=image, 
        offset=(0,0,0), 
        layer_path=layer_path, 
        layer_type='image', 
        resolution=(1,1,1), 
        encoding='raw'
    )

    vol.cache = True
    vol.flush_cache()

    # Test that reading populates the cache
    read1 = vol[:,:,:]
    assert np.all(read1 == image)

    read2 = vol[:,:,:]
    assert np.all(read2 == image)

    assert len(os.listdir(os.path.join(vol.cache_path, vol.key))) > 0

    files = os.listdir(os.path.join(vol.cache_path, vol.key))
    validation_set = [
        '0-64_0-64_0-64',
        '64-128_0-64_0-64',
        '0-64_64-128_0-64',
        '0-64_0-64_64-128',
        '64-128_64-128_0-64',
        '64-128_0-64_64-128',
        '0-64_64-128_64-128',
        '64-128_64-128_64-128'
    ]
    assert set([ os.path.splitext(fname)[0] for fname in files ]) == set(validation_set)

    for i in range(8):
        fname = os.path.join(vol.cache_path, vol.key, validation_set[i]) + '.gz'
        with gzip.GzipFile(fname, mode='rb') as gfile:
            chunk = gfile.read()
        img3d = chunks.decode(
          chunk, 'raw', (64,64,64,1), np.uint8
        )
        assert np.all(img3d == (i+1))

    vol.flush_cache()
    assert not os.path.exists(vol.cache_path)

    # Test that writing populates the cache
    vol[:,:,:] = image

    assert os.path.exists(vol.cache_path)
    assert np.all(vol[:,:,:] == image)

    vol.flush_cache()

    # Test that partial reads work too
    result = vol[0:64,0:64,:]
    assert np.all(result == image[0:64,0:64,:])
    files = os.listdir(os.path.join(vol.cache_path, vol.key))
    assert len(files) == 2
    result = vol[:,:,:]
    assert np.all(result == image)
    files = os.listdir(os.path.join(vol.cache_path, vol.key))
    assert len(files) == 8

    vol.flush_cache()

    # Test Non-standard Cache Destination
    dirpath = '/tmp/cloudvolume/caching-cache-' + str(TEST_NUMBER)
    vol.cache = dirpath
    vol[:,:,:] = image

    assert len(os.listdir(os.path.join(dirpath, vol.key))) == 8

    vol.flush_cache()


def test_cache_validity():
    image = np.zeros(shape=(128,128,128,1), dtype=np.uint8)
    dirpath = '/tmp/cloudvolume/caching-validity-' + str(TEST_NUMBER)
    layer_path = 'file://' + dirpath

    vol = create_volume_from_image(
        image=image, 
        offset=(1,1,1), 
        layer_path=layer_path, 
        layer_type='image', 
        resolution=(1,1,1), 
        encoding='raw'
    )
    vol.cache = True
    vol.flush_cache()
    vol.commit_info()

    def test_with_mock_cache_info(info, shoulderror):
        finfo = os.path.join(vol.cache_path, 'info')
        with open(finfo, 'w') as f:
            f.write(json.dumps(info))

        if shoulderror:
            try:
                CloudVolume(vol.layer_cloudpath, cache=True)
            except ValueError:
                pass
            else:
                assert False
        else:
            CloudVolume(vol.layer_cloudpath, cache=True)

    test_with_mock_cache_info(vol.info, shoulderror=False)

    info = vol.info.copy()
    info['scales'][0]['size'][0] = 666
    test_with_mock_cache_info(info, shoulderror=False)

    test_with_mock_cache_info({ 'zomg': 'wow' }, shoulderror=True)

    def tiny_change(key, val):
        info = vol.info.copy()
        info[key] = val
        test_with_mock_cache_info(info, shoulderror=True)

    tiny_change('type', 'zoolander')
    tiny_change('data_type', 'uint32')
    tiny_change('num_channels', 2)
    tiny_change('mesh', 'mesh')

    def scale_change(key, val, mip=0):
        info = vol.info.copy()
        info['scales'][mip][key] = val
        test_with_mock_cache_info(info, shoulderror=True)

    scale_change('voxel_offset', [ 1, 2, 3 ])
    scale_change('resolution', [ 1, 2, 3 ])
    scale_change('encoding', 'npz')

    vol.flush_cache()

    # Test no info file at all    
    CloudVolume(vol.layer_cloudpath, cache=True)

    vol.flush_cache()

def test_exists():

    # Bbox version
    delete_layer()
    cv, data = create_layer(size=(128,64,64,1), offset=(0,0,0))

    defexists = Bbox( (0,0,0), (128,64,64) )
    results = cv.exists(defexists)
    assert len(results) == 2
    assert results['1_1_1/0-64_0-64_0-64'] == True
    assert results['1_1_1/64-128_0-64_0-64'] == True

    fpath = os.path.join(cv.layer_cloudpath, cv.key, '64-128_0-64_0-64')
    fpath = fpath.replace('file://', '') + '.gz'
    os.remove(fpath)

    results = cv.exists(defexists)
    assert len(results) == 2
    assert results['1_1_1/0-64_0-64_0-64'] == True
    assert results['1_1_1/64-128_0-64_0-64'] == False

    # Slice version
    delete_layer()
    cv, data = create_layer(size=(128,64,64,1), offset=(0,0,0))

    defexists = np.s_[ 0:128, :, : ]

    results = cv.exists(defexists)
    assert len(results) == 2
    assert results['1_1_1/0-64_0-64_0-64'] == True
    assert results['1_1_1/64-128_0-64_0-64'] == True

    fpath = os.path.join(cv.layer_cloudpath, cv.key, '64-128_0-64_0-64')
    fpath = fpath.replace('file://', '') + '.gz'
    os.remove(fpath)

    results = cv.exists(defexists)
    assert len(results) == 2
    assert results['1_1_1/0-64_0-64_0-64'] == True
    assert results['1_1_1/64-128_0-64_0-64'] == False

def test_delete():

    # Bbox version
    delete_layer()
    cv, data = create_layer(size=(128,64,64,1), offset=(0,0,0))

    defexists = Bbox( (0,0,0), (128,64,64) )
    results = cv.exists(defexists)
    assert len(results) == 2
    assert results['1_1_1/0-64_0-64_0-64'] == True
    assert results['1_1_1/64-128_0-64_0-64'] == True


    cv.delete(defexists)
    results = cv.exists(defexists)
    assert len(results) == 2
    assert results['1_1_1/0-64_0-64_0-64'] == False
    assert results['1_1_1/64-128_0-64_0-64'] == False

    # Slice version
    delete_layer()
    cv, data = create_layer(size=(128,64,64,1), offset=(0,0,0))

    defexists = np.s_[ 0:128, :, : ]

    results = cv.exists(defexists)
    assert len(results) == 2
    assert results['1_1_1/0-64_0-64_0-64'] == True
    assert results['1_1_1/64-128_0-64_0-64'] == True

    cv.delete(defexists)
    results = cv.exists(defexists)
    assert len(results) == 2
    assert results['1_1_1/0-64_0-64_0-64'] == False
    assert results['1_1_1/64-128_0-64_0-64'] == False

    # Check errors
    delete_layer()
    cv, data = create_layer(size=(128,64,64,1), offset=(0,0,0))

    try:
        results = cv.exists( np.s_[1:129, :, :] )
    except ValueError:
        pass
    else:
        assert False

