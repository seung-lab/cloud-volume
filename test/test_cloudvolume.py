import pytest

import copy
import gzip
import json
import numpy as np
import os
import shutil
import sys

from cloudvolume import exceptions
from cloudvolume.exceptions import AlignmentError, ReadOnlyException
from cloudvolume import CloudVolume, chunks
from cloudvolume.lib import Bbox, Vec, yellow, mkdir
import cloudvolume.sharedmemory as shm
from layer_harness import (
  TEST_NUMBER,  
  delete_layer, create_layer
)
from cloudvolume.datasource.precomputed.common import cdn_cache_control
from cloudvolume.datasource.precomputed.image.tx import generate_chunks

def test_from_numpy():
  arr = np.random.randint(0, high=256, size=(128,128, 128))
  arr = np.asarray(arr, dtype=np.uint8)
  vol = CloudVolume.from_numpy(arr, max_mip=1)
  arr2 = vol[:,:,:]
  np.alltrue(arr == arr2)
  
  arr = np.random.randn(128,128, 128, 3)
  arr = np.asarray(arr, dtype=np.float32)
  vol = CloudVolume.from_numpy(arr, max_mip=1)
  arr2 = vol[:,:,:]
  np.alltrue(arr == arr2)
  shutil.rmtree('/tmp/image')

def test_cloud_access():
  CloudVolume('gs://seunglab-test/test_v0/image')
  CloudVolume('s3://seunglab-test/test_dataset/image')

def test_fill_missing():
  info = CloudVolume.create_new_info(
    num_channels=1, # Increase this number when we add more tests for RGB
    layer_type='image', 
    data_type='uint8', 
    encoding='raw',
    resolution=[ 1,1,1 ], 
    voxel_offset=[0,0,0], 
    volume_size=[128,128,64],
    mesh='mesh', 
    chunk_size=[ 64,64,64 ],
  )

  vol = CloudVolume('file:///tmp/cloudvolume/empty_volume', mip=0, info=info)
  vol.commit_info()

  vol.cache.flush()

  vol = CloudVolume('file:///tmp/cloudvolume/empty_volume', mip=0, fill_missing=True)
  assert np.count_nonzero(vol[:]) == 0

  vol = CloudVolume('file:///tmp/cloudvolume/empty_volume', mip=0, fill_missing=True, cache=True)
  assert np.count_nonzero(vol[:]) == 0
  assert np.count_nonzero(vol[:]) == 0

  vol.cache.flush()
  delete_layer('/tmp/cloudvolume/empty_volume')

def test_background_color():
  info = CloudVolume.create_new_info(
    num_channels=1, 
    layer_type='image', 
    data_type='uint8', 
    encoding='raw',
    resolution=[ 1,1,1 ], 
    voxel_offset=[0,0,0], 
    volume_size=[128,128,1],
    mesh='mesh', 
    chunk_size=[ 64,64,1 ],
  )

  vol = CloudVolume('file:///tmp/cloudvolume/empty_volume', mip=0, info=info)
  vol.commit_info()

  vol.cache.flush()

  vol = CloudVolume('file:///tmp/cloudvolume/empty_volume', 
                    mip=0, 
                    background_color=1, 
                    fill_missing=True)
  assert np.count_nonzero(vol[:] - 1) == 0

  vol = CloudVolume('file:///tmp/cloudvolume/empty_volume', 
                    mip=0, 
                    background_color=1, 
                    fill_missing=True,
                    bounded=False)
  assert np.count_nonzero(vol[0:129,0:129,0:1]-1) == 0

  vol = CloudVolume('file:///tmp/cloudvolume/empty_volume', 
                    mip=0, 
                    background_color=1, 
                    fill_missing=True,
                    bounded=False,
                    parallel=2)
  assert np.count_nonzero(vol[0:129,0:129,0:1]-1) == 0
  vol.cache.flush()
  delete_layer('/tmp/cloudvolume/empty_volume')

def test_has_data():
  delete_layer()
  cv, data = create_layer(size=(50,50,50,1), offset=(0,0,0))
  cv.add_scale((2,2,1))

  assert cv.image.has_data(0) == True
  assert cv.image.has_data(1) == False
  assert cv.image.has_data([1,1,1]) == True
  assert cv.image.has_data([2,2,1]) == False

  try:
    cv.image.has_data(2)
    assert False
  except exceptions.ScaleUnavailableError:
    pass

def image_equal(arr1, arr2, encoding):
  assert arr1.shape == arr2.shape
  if encoding == 'raw':
    return np.all(arr1 == arr2)

  # jpeg
  u1 = arr1.flatten().mean()
  u2 = arr2.flatten().mean()
  return np.abs(u1 - u2) < 1

@pytest.mark.parametrize('green', (True, False))
@pytest.mark.parametrize('encoding', ('raw', 'jpeg'))
def test_aligned_read(green, encoding):
  delete_layer()
  cv, data = create_layer(size=(50,50,50,1), offset=(0,0,0), encoding=encoding)
  cv.green_threads = green
  # the last dimension is the number of channels
  assert cv[0:50,0:50,0:50].shape == (50,50,50,1)
  assert image_equal(cv[0:50,0:50,0:50], data, encoding)
  
  delete_layer()
  cv, data = create_layer(size=(128,64,64,1), offset=(0,0,0), encoding=encoding)
  cv.green_threads = green
  # the last dimension is the number of channels
  assert cv[0:64,0:64,0:64].shape == (64,64,64,1) 

  assert image_equal(cv[0:64,0:64,0:64], data[:64,:64,:64,:], encoding)

  delete_layer()
  cv, data = create_layer(size=(128,64,64,1), offset=(10,20,0), encoding=encoding)
  cv.green_threads = green
  cutout = cv[10:74,20:84,0:64]
  # the last dimension is the number of channels
  assert cutout.shape == (64,64,64,1) 
  assert image_equal(cutout, data[:64,:64,:64,:], encoding)
  # get the second chunk
  cutout2 = cv[74:138,20:84,0:64]
  assert cutout2.shape == (64,64,64,1) 
  assert image_equal(cutout2, data[64:128,:64,:64,:], encoding)

  assert cv[25, 25, 25].shape == (1,1,1,1)

def test_save_images():
  delete_layer()
  cv, data = create_layer(size=(50,50,50,1), offset=(0,0,0))  

  img = cv[:]
  directory = img.save_images()

  for z, fname in enumerate(sorted(os.listdir(directory))):
    assert fname == str(z).zfill(3) + '.png'

  shutil.rmtree(directory)

def test_bbox_read():
  delete_layer()
  cv, data = create_layer(size=(50,50,50,1), offset=(0,0,0))

  x = Bbox( (0,1,2), (48,49,50) )
  # the last dimension is the number of channels
  assert cv[x].shape == (48,48,48,1)
  assert np.all(cv[x] == data[0:48, 1:49, 2:50])  

def test_number_type_read():
  delete_layer()
  cv, data = create_layer(size=(50,50,50,1), offset=(0,0,0))

  for datatype in (
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.int8, np.int16, np.int32, np.int64,
    np.float16, np.float32, np.float64,
    int, float
  ):
    x = datatype(5)

    # the last dimension is the number of channels
    assert cv[x,x,x].shape == (1,1,1,1)
    assert np.all(cv[x,x,x] == data[5,5,5])  

def test_ellipsis_read():
  delete_layer()
  cv, data = create_layer(size=(50,50,50,1), offset=(0,0,0))

  img = cv[...]
  assert np.all(img == data)

  img = cv[5:10, ...]
  assert np.all(img == data[5:10, :,:,:])

  img = cv[5:10, 7, 8, ...]
  assert np.all(np.squeeze(img) == data[5:10, 7, 8, 0])

  img = cv[5:10, ..., 8, 0]
  assert np.all(np.squeeze(img) == data[5:10, :, 8, 0])

  try:
    img = cv[..., 5, ..., 0]
  except ValueError:
    pass


@pytest.mark.parametrize('protocol', ('gs', 's3'))
@pytest.mark.parametrize('parallel', (2, True))
def test_parallel_read(protocol, parallel):
  cloudpath = "{}://seunglab-test/test_v0/image".format(protocol)

  vol1 = CloudVolume(cloudpath, parallel=1)
  vol2 = CloudVolume(cloudpath, parallel=parallel)

  data1 = vol1[:512,:512,:50]
  img = vol2[:512,:512,:50]
  assert np.all(data1 == vol2[:512,:512,:50])


@pytest.mark.parametrize('protocol', ('gs', 's3'))
def test_parallel_read_shm(protocol):
  cloudpath = "{}://seunglab-test/test_v0/image".format(protocol)
  
  vol1 = CloudVolume(cloudpath, parallel=1)
  vol2 = CloudVolume(cloudpath, parallel=2)

  data1 = vol1[:512,:512,:50]
  data2 = vol2.download_to_shared_memory(np.s_[:512,:512,:50])
  assert np.all(data1 == data2)
  data2.close()
  vol2.unlink_shared_memory()

def test_parallel_write():
  delete_layer()
  cv, data = create_layer(size=(512,512,128,1), offset=(0,0,0))
  
  # aligned write
  cv.parallel = 2
  cv[:] = np.zeros(shape=(512,512,128,1), dtype=cv.dtype) + 5
  data = cv[:]
  assert np.all(data == 5)

  # non-aligned-write
  cv.parallel = 2
  cv.non_aligned_writes = True
  cv[1:,1:,1:] = np.zeros(shape=(511,511,127,1), dtype=cv.dtype) + 7
  data = cv[1:,1:,1:]
  assert np.all(data == 7)

  # thin non-aligned-write so that there's no aligned core
  cv.parallel = 2
  cv.non_aligned_writes = True
  cv[25:75,25:75,25:75] = np.zeros(shape=(50,50,50,1), dtype=cv.dtype) + 8
  data = cv[25:75,25:75,25:75]
  assert np.all(data == 8)


def test_parallel_shared_memory_write():
  delete_layer()
  cv, _ = create_layer(size=(256,256,128,1), offset=(0,0,0))

  shm_location = 'cloudvolume-test-shm-parallel-write'
  mmapfh, shareddata = shm.ndarray(shape=(256,256,128), dtype=np.uint8, location=shm_location)
  shareddata[:] = 1

  cv.parallel = 1
  cv.upload_from_shared_memory(shm_location, Bbox((0,0,0), (256,256,128)))
  assert np.all(cv[:] == 1)

  shareddata[:] = 2
  cv.parallel = 2
  cv.upload_from_shared_memory(shm_location, Bbox((0,0,0), (256,256,128)))
  assert np.all(cv[:] == 2)

  shareddata[:,:,:64] = 3
  cv.upload_from_shared_memory(shm_location, bbox=Bbox((0,0,0), (256,256,128)), 
    cutout_bbox=Bbox((0,0,0), (256,256,64)))
  assert np.all(cv[:,:,:64] == 3)    
  assert np.all(cv[:,:,64:128] == 2)    

  shareddata[:,:,:69] = 4
  cv.autocrop = True
  cv.upload_from_shared_memory(shm_location, bbox=Bbox((-5,-5,-5), (251,251,123)), 
    cutout_bbox=Bbox((-5,-5,-5), (128,128,64)))
  assert np.all(cv[:128,:128,:63] == 4)    
  assert np.all(cv[128:,128:,:64] == 3)    
  assert np.all(cv[:,:,64:128] == 2)    

  shareddata[:] = 0
  shareddata[:,0,0] = 1
  cv.upload_from_shared_memory(shm_location, bbox=Bbox((0,0,0), (256,256,128)), order='C')
  assert np.all(cv[0,0,:] == 1)
  assert np.all(cv[1,0,:] == 0)

  mmapfh.close()
  shm.unlink(shm_location)

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

  # read a single pixel
  delete_layer()
  cv, data = create_layer(size=(256,256,64,1), offset=(3,7,11))
  # the last dimension is the number of channels
  assert cv[22:77:2, 22:197:3, 22:32].shape == (28,59,10,1) 
  assert data[19:74:2, 15:190:3, 11:21,:].shape == (28,59,10,1) 
  assert np.all(cv[22:77:2, 22:197:3, 22:32] == data[19:74:2, 15:190:3, 11:21,:])

def test_autocropped_read():
  delete_layer()
  cv, data = create_layer(size=(50,50,50,1), offset=(0,0,0))

  cv.autocrop = True
  cv.bounded = False

  # left overlap
  img = cv[-25:25,-25:25,-25:25]
  assert img.shape == (25,25,25,1)
  assert np.all(img == data[:25, :25, :25])

  # right overlap
  img = cv[40:60, 40:60, 40:60]
  assert img.shape == (10,10,10,1)
  assert np.all(img == data[40:, 40:, 40:])

  # containing
  img = cv[-100:100, -100:100, -100:100]
  assert img.shape == (50,50,50,1)
  assert np.all(img == data)

  # contained
  img = cv[10:20, 10:20, 10:20]
  assert img.shape == (10,10,10,1)
  assert np.all(img == data[10:20, 10:20, 10:20])

  # non-intersecting
  img = cv[100:120, 100:120, 100:120]
  assert img.shape == (0,0,0,1)
  assert np.all(img == data[0:0, 0:0, 0:0])    

@pytest.mark.parametrize('green', (True, False))
def test_download_upload_file(green):
  delete_layer()
  cv, _ = create_layer(size=(50,50,50,1), offset=(0,0,0))
  cv.green_threads = green

  mkdir('/tmp/file/')

  cv.download_to_file('/tmp/file/test', cv.bounds)
  cv2 = CloudVolume('file:///tmp/file/test2/', info=cv.info)
  cv2.upload_from_file('/tmp/file/test', cv.bounds)

  assert np.all(cv2[:] == cv[:])
  shutil.rmtree('/tmp/file/')

def test_numpy_memmap():
  delete_layer()
  cv, data = create_layer(size=(50,50,50,1), offset=(0,0,0))

  mkdir('/tmp/file/test/')

  with open("/tmp/file/test/chunk.data", "wb") as f:
    f.write(data.tobytes("F"))

  fp = np.memmap("/tmp/file/test/chunk.data", dtype=data.dtype, mode='r', shape=(50,50,50,1), order='F')
  cv[:] = fp[:]

  shutil.rmtree('/tmp/file/')

@pytest.mark.parametrize('green', (True, False))
@pytest.mark.parametrize('encoding', ('raw', 'jpeg'))
def test_write(green, encoding):
  delete_layer()
  cv, _ = create_layer(size=(50,50,50,1), offset=(0,0,0))
  cv.green_threads = green

  replacement_data = np.zeros(shape=(50,50,50,1), dtype=np.uint8)
  cv[0:50,0:50,0:50] = replacement_data
  assert image_equal(cv[0:50,0:50,0:50], replacement_data, encoding)

  replacement_data = np.random.randint(255, size=(50,50,50,1), dtype=np.uint8)
  cv[0:50,0:50,0:50] = replacement_data
  assert image_equal(cv[0:50,0:50,0:50], replacement_data, encoding)

  replacement_data = np.random.randint(255, size=(50,50,50,1), dtype=np.uint8)
  bbx = Bbox((0,0,0), (50,50,50))
  cv[bbx] = replacement_data
  assert image_equal(cv[bbx], replacement_data, encoding)

  # out of bounds
  delete_layer()
  cv, _ = create_layer(size=(128,64,64,1), offset=(10,20,0))
  cv.green_threads = green
  with pytest.raises(ValueError):
    cv[74:150,20:84,0:64] = np.ones(shape=(64,64,64,1), dtype=np.uint8)
  
  # non-aligned writes
  delete_layer()
  cv, _ = create_layer(size=(128,64,64,1), offset=(10,20,0))
  cv.green_threads = green
  with pytest.raises(ValueError):
    cv[21:85,0:64,0:64] = np.ones(shape=(64,64,64,1), dtype=np.uint8)

  # test bounds check for short boundary chunk
  delete_layer()
  cv, _ = create_layer(size=(25,25,25,1), offset=(1,3,5))
  cv.green_threads = green
  cv.info['scales'][0]['chunk_sizes'] = [[ 11,11,11 ]]
  cv[:] = np.ones(shape=(25,25,25,1), dtype=np.uint8)

def test_non_aligned_write():
  delete_layer()
  offset = Vec(5,7,13)
  cv, _ = create_layer(size=(1024, 1024, 5, 1), offset=offset)

  cv[:] = np.zeros(shape=cv.shape, dtype=cv.dtype)

  # Write inside a single chunk

  onepx = Bbox( (10,200,15), (11,201,16) )
  try:
    cv[ onepx.to_slices() ] = np.ones(shape=onepx.size3(), dtype=cv.dtype)
    assert False
  except AlignmentError:
    pass
  
  cv.non_aligned_writes = True
  cv[ onepx.to_slices() ] = np.ones(shape=onepx.size3(), dtype=cv.dtype)
  answer = np.zeros(shape=cv.shape, dtype=cv.dtype)
  answer[ 5, 193, 2 ] = 1
  assert np.all(cv[:] == answer)

  # Write across multiple chunks
  cv[:] = np.zeros(shape=cv.shape, dtype=cv.dtype)
  cv.non_aligned_writes = True
  middle = Bbox( (512 - 10, 512 - 11, 0), (512 + 10, 512 + 11, 5) ) + offset
  cv[ middle.to_slices() ] = np.ones(shape=middle.size3(), dtype=cv.dtype)
  answer = np.zeros(shape=cv.shape, dtype=cv.dtype)
  answer[ 502:522, 501:523, : ] = 1
  assert np.all(cv[:] == answer)    

  cv.non_aligned_writes = False
  try:
    cv[ middle.to_slices() ] = np.ones(shape=middle.size3(), dtype=cv.dtype)
    assert False
  except AlignmentError:
    pass

  # Big inner shell
  delete_layer()
  cv, _ = create_layer(size=(1024, 1024, 5, 1), offset=offset)
  cv[:] = np.zeros(shape=cv.shape, dtype=cv.dtype)
  middle = Bbox( (512 - 150, 512 - 150, 0), (512 + 150, 512 + 150, 5) ) + offset

  try:
    cv[ middle.to_slices() ] = np.ones(shape=middle.size3(), dtype=cv.dtype)
    assert False
  except AlignmentError:
    pass

  cv.non_aligned_writes = True
  cv[ middle.to_slices() ] = np.ones(shape=middle.size3(), dtype=cv.dtype)
  answer = np.zeros(shape=cv.shape, dtype=cv.dtype)
  answer[ 362:662, 362:662, : ] = 1
  assert np.all(cv[:] == answer)    

from cloudvolume import view

def test_autocropped_write():
  delete_layer()
  cv, _ = create_layer(size=(100,100,100,1), offset=(0,0,0))

  cv.autocrop = True
  cv.bounded = False

  replacement_data = np.ones(shape=(300,300,300,1), dtype=np.uint8)
  cv[-100:200, -100:200, -100:200] = replacement_data
  assert np.all(cv[:,:,:] == replacement_data[0:100,0:100,0:100])
  
  replacement_data = np.random.randint(255, size=(100,100,100,1), dtype=np.uint8)
  
  cv[-50:50, -50:50, -50:50] = replacement_data
  assert np.all(cv[0:50,0:50,0:50] == replacement_data[50:, 50:, 50:])

  cv[50:150, 50:150, 50:150] = replacement_data
  assert np.all(cv[50:,50:,50:] == replacement_data[:50, :50, :50])

  cv[0:50, 0:50, 0:50] = replacement_data[:50,:50,:50]
  assert np.all(cv[0:50, 0:50, 0:50] == replacement_data[:50,:50,:50])    

  replacement_data = np.ones(shape=(100,100,100,1), dtype=np.uint8)
  cv[:] = replacement_data + 1
  cv[100:200, 100:200, 100:200] = replacement_data
  assert np.all(cv[:,:,:] != 1)

def test_writer_last_chunk_smaller():
  delete_layer()
  cv, data = create_layer(size=(100,64,64,1), offset=(0,0,0))
  cv.info['scales'][0]['chunk_sizes'] = [[ 64,64,64 ]]
  
  chunks = [ chunk for chunk in generate_chunks(cv.meta, data[:,:,:,:], (0,0,0), cv.mip) ]

  assert len(chunks) == 2

  startpt, endpt, spt, ept = chunks[0]
  assert np.array_equal(spt, (0,0,0))
  assert np.array_equal(ept, (64,64,64))
  assert np.all((endpt - startpt) == Vec(64,64,64))

  startpt, endpt, spt, ept = chunks[1]
  assert np.array_equal(spt, (64,0,0))
  assert np.array_equal(ept, (100,64,64))
  assert np.all((endpt - startpt) == Vec(36,64,64))

def test_write_compressed_segmentation():
  delete_layer()
  cv, data = create_layer(size=(128,64,64,1), offset=(0,0,0))

  cv.info['num_channels'] = 1
  cv.info['data_type'] = 'uint32'
  cv.scale['encoding'] = 'compressed_segmentation'
  cv.scale['compressed_segmentation_block_size'] = (8,8,8)
  cv.commit_info()

  cv[:] = data.astype(np.uint32)
  data2 = cv[:]

  assert np.all(data == data2)

  cv.info['data_type'] = 'uint64'
  cv.commit_info()

  cv[:] = data.astype(np.uint64)
  data2 = cv[:]
  
  assert np.all(data == data2)

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

def test_negative_coords_upload_download():
  cv, data = create_layer(size=(128,64,64,1), offset=(-64,-64,-64))

  downloaded = cv[-64:64, -64:0, -64:0]
  assert np.all(data == downloaded)

def test_setitem_mismatch():
  delete_layer()
  cv, _ = create_layer(size=(64,64,64,1), offset=(0,0,0))

  with pytest.raises(ValueError):
    cv[0:64,0:64,0:64] = np.zeros(shape=(5,5,5,1), dtype=np.uint8)

def test_bounds():
  delete_layer()
  cv, _ = create_layer(size=(128,64,64,1), offset=(100,100,100))
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

def test_provenance():
  delete_layer()
  cv, _ = create_layer(size=(64,64,64,1), offset=(0,0,0))

  provobj = json.loads(cv.provenance.serialize())
  provobj['processing'] = [] # from_numpy
  assert provobj == {
    "sources": [], 
    "owners": [], 
    "processing": [], 
    "description": ""
  }

  cv.provenance.sources.append('cooldude24@princeton.edu')
  cv.commit_provenance()
  cv.refresh_provenance()

  assert cv.provenance.sources == [ 'cooldude24@princeton.edu' ]

  # should not die
  cv = CloudVolume(cv.cloudpath, provenance={})
  cv = CloudVolume(cv.cloudpath, provenance={ 'sources': [] })
  cv = CloudVolume(cv.cloudpath, provenance={ 'owners': [] })
  cv = CloudVolume(cv.cloudpath, provenance={ 'processing': [] })
  cv = CloudVolume(cv.cloudpath, provenance={ 'description': '' })

  # should die
  try:
    cv = CloudVolume(cv.cloudpath, provenance={ 'sources': 3 })
    assert False
  except:
    pass

  cv = CloudVolume(cv.cloudpath, provenance="""{
    "sources": [ "wow" ]
  }""")

  assert cv.provenance.sources[0] == 'wow'

def test_info_provenance_cache():
  image = np.zeros(shape=(128,128,128,1), dtype=np.uint8)
  vol = CloudVolume.from_numpy(
    image, 
    voxel_offset=(0,0,0), 
    vol_path='gs://seunglab-test/cloudvolume/caching', 
    layer_type='image', 
    resolution=(1,1,1), 
    encoding='raw'
  )

  # Test Info
  vol.cache.enabled = True
  vol.cache.flush()
  info = vol.refresh_info()
  assert info is not None

  with open(os.path.join(vol.cache.path, 'info'), 'r') as infof:
    info = infof.read()
    info = json.loads(info)

  with open(os.path.join(vol.cache.path, 'info'), 'w') as infof:
    infof.write(json.dumps({ 'wow': 'amaze' }))

  info = vol.refresh_info()
  assert info == { 'wow': 'amaze' }
  vol.cache.enabled = False
  info = vol.refresh_info()
  assert info != { 'wow': 'amaze' }

  infopath = os.path.join(vol.cache.path, 'info')
  assert os.path.exists(infopath)

  vol.cache.flush_info()
  assert not os.path.exists(infopath)
  vol.cache.flush_info() # assert no error by double delete


  # Test Provenance
  vol.cache.enabled = True
  vol.cache.flush()
  prov = vol.refresh_provenance()
  assert prov is not None

  with open(os.path.join(vol.cache.path, 'provenance'), 'r') as provf:
    prov = provf.read()
    prov = json.loads(prov)

  with open(os.path.join(vol.cache.path, 'provenance'), 'w') as provf:
    prov['description'] = 'wow'
    provf.write(json.dumps(prov))

  prov = vol.refresh_provenance()
  assert prov['description'] == 'wow'
  vol.cache.enabled = False
  prov = vol.refresh_provenance()
  assert prov['description'] == ''

  provpath = os.path.join(vol.cache.path, 'provenance')
  vol.cache.flush_provenance()
  assert not os.path.exists(provpath)
  vol.cache.flush_provenance() # assert no error by double delete

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

  vol = CloudVolume.from_numpy(
    image, 
    voxel_offset=(0,0,0), 
    vol_path=layer_path, 
    layer_type='image', 
    resolution=(1,1,1), 
    encoding='raw',
    chunk_size=(64,64,64),
  )

  vol.cache.enabled = True
  vol.cache.flush()

  # Test that reading populates the cache
  read1 = vol[:,:,:]
  assert np.all(read1 == image)

  read2 = vol[:,:,:]
  assert np.all(read2 == image)

  assert len(vol.cache.list()) > 0

  files = vol.cache.list()
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
    fname = os.path.join(vol.cache.path, vol.key, validation_set[i]) + '.gz'
    with gzip.GzipFile(fname, mode='rb') as gfile:
      chunk = gfile.read()
    img3d = chunks.decode(
      chunk, 'raw', (64,64,64,1), np.uint8
    )
    assert np.all(img3d == (i+1))

  vol.cache.flush()
  assert not os.path.exists(vol.cache.path)

  # Test that writing populates the cache
  vol[:,:,:] = image

  assert os.path.exists(vol.cache.path)
  assert np.all(vol[:,:,:] == image)

  vol.cache.flush()

  # Test that partial reads work too
  result = vol[0:64,0:64,:]
  assert np.all(result == image[0:64,0:64,:])
  files = vol.cache.list()
  assert len(files) == 2
  result = vol[:,:,:]
  assert np.all(result == image)
  files = vol.cache.list()
  assert len(files) == 8

  vol.cache.flush()

  # Test Non-standard Cache Destination
  dirpath = '/tmp/cloudvolume/caching-cache-' + str(TEST_NUMBER)
  vol.cache.enabled = True
  vol.cache.path = dirpath
  vol[:,:,:] = image

  assert len(os.listdir(os.path.join(dirpath, vol.key))) == 8

  vol.cache.flush()

  # Test that caching doesn't occur when cache is not set
  vol.cache.enabled = False
  result = vol[:,:,:]
  if os.path.exists(vol.cache.path):
    files = vol.cache.list()
    assert len(files) == 0

  vol[:,:,:] = image
  if os.path.exists(vol.cache.path):
    files = vol.cache.list()
    assert len(files) == 0

  vol.cache.flush()

  # Test that deletion works too
  vol.cache.enabled = True
  vol[:,:,:] = image
  files = vol.cache.list()
  assert len(files) == 8
  vol.delete( np.s_[:,:,:] )
  files = vol.cache.list()
  assert len(files) == 0

  vol.cache.flush()    

  vol[:,:,:] = image
  files = vol.cache.list()
  assert len(files) == 8
  vol.cache.flush(preserve=np.s_[:,:,:])
  files = vol.cache.list()
  assert len(files) == 8
  vol.cache.flush(preserve=np.s_[:64,:64,:])
  files = vol.cache.list()
  assert len(files) == 2

  vol.cache.flush()

  vol[:,:,:] = image
  files = vol.cache.list()
  assert len(files) == 8
  vol.cache.flush_region(Bbox( (50, 50, 0), (100, 100, 10) ))
  files = vol.cache.list()
  assert len(files) == 4

  vol.cache.flush()

  vol[:,:,:] = image
  files = vol.cache.list()
  assert len(files) == 8
  vol.cache.flush_region(np.s_[50:100, 50:100, 0:10])
  files = vol.cache.list()
  assert len(files) == 4

  vol.cache.flush()

def test_cache_compression_setting():
  image = np.zeros(shape=(128,128,128,1), dtype=np.uint8)
  dirpath = '/tmp/cloudvolume/caching-validity-' + str(TEST_NUMBER)
  layer_path = 'file://' + dirpath

  vol = CloudVolume.from_numpy(
    image, 
    voxel_offset=(1,1,1), 
    vol_path=layer_path, 
    layer_type='image', 
    resolution=(1,1,1), 
    encoding='raw'
  )
  vol.cache.enabled = True
  vol.cache.flush()
  vol.commit_info()

  vol.cache.compress = None
  vol[:] = image
  assert all([ os.path.splitext(x)[1] == '.gz' for x in vol.cache.list() ])
  vol.cache.flush()

  vol.cache.compress = True
  vol[:] = image
  assert all([ os.path.splitext(x)[1] == '.gz' for x in vol.cache.list() ])
  vol.cache.flush()

  vol.cache.compress = False
  vol[:] = image
  assert all([ os.path.splitext(x)[1] == '' for x in vol.cache.list() ])
  vol.cache.flush()

  delete_layer(dirpath)

def test_cache_validity():
  image = np.zeros(shape=(128,128,128,1), dtype=np.uint8)
  dirpath = '/tmp/cloudvolume/caching-validity-' + str(TEST_NUMBER)
  layer_path = 'file://' + dirpath

  vol = CloudVolume.from_numpy(
    image, 
    voxel_offset=(1,1,1), 
    vol_path=layer_path, 
    layer_type='image', 
    resolution=(1,1,1), 
    encoding='raw'
  )
  vol.cache.enabled = True
  vol.cache.flush()
  vol.commit_info()

  def test_with_mock_cache_info(info, shoulderror):
    finfo = os.path.join(vol.cache.path, 'info')
    with open(finfo, 'w') as f:
      f.write(json.dumps(info))

    if shoulderror:
      try:
        CloudVolume(vol.cloudpath, cache=True)
      except ValueError:
        pass
      else:
        assert False
    else:
      CloudVolume(vol.cloudpath, cache=True)

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

  vol.cache.flush()

  # Test no info file at all    
  CloudVolume(vol.cloudpath, cache=True)

  vol.cache.flush()

def test_pickling():
  import pickle
  delete_layer()
  cv, _ = create_layer(size=(128,64,64,1), offset=(0,0,0))

  pckl = pickle.dumps(cv)
  cv2 = pickle.loads(pckl)

  assert cv2.cloudpath == cv.cloudpath
  assert cv2.mip == cv.mip

def test_multiprocess():
  from concurrent.futures import ProcessPoolExecutor, as_completed

  delete_layer()
  cv, _ = create_layer(size=(128,64,64,1), offset=(0,0,0))
  cv.commit_info()

  # "The ProcessPoolExecutor class has known (unfixable) 
  # problems on Python 2 and should not be relied on 
  # for mission critical work."
  # https://pypi.org/project/futures/

  if sys.version_info[0] < 3:
    print(yellow("External multiprocessing not supported in Python 2."))
    return
  
  futures = []
  with ProcessPoolExecutor(max_workers=4) as ppe:
    for _ in range(0, 5):
      futures.append(ppe.submit(cv.refresh_info))

    for future in as_completed(futures):
      # an error should be re-raised in one of the futures
      future.result()

  delete_layer()

def test_exists():
  # Bbox version
  delete_layer()
  cv, _ = create_layer(size=(128,64,64,1), offset=(0,0,0))

  defexists = Bbox( (0,0,0), (128,64,64) )
  results = cv.exists(defexists)
  assert len(results) == 2
  assert results['1_1_1/0-64_0-64_0-64'] == True
  assert results['1_1_1/64-128_0-64_0-64'] == True

  fpath = os.path.join(cv.cloudpath, cv.key, '64-128_0-64_0-64')
  fpath = fpath.replace('file://', '') + '.gz'
  os.remove(fpath)

  results = cv.exists(defexists)
  assert len(results) == 2
  assert results['1_1_1/0-64_0-64_0-64'] == True
  assert results['1_1_1/64-128_0-64_0-64'] == False

  # Slice version
  delete_layer()
  cv, _ = create_layer(size=(128,64,64,1), offset=(0,0,0))

  defexists = np.s_[ 0:128, :, : ]

  results = cv.exists(defexists)
  assert len(results) == 2
  assert results['1_1_1/0-64_0-64_0-64'] == True
  assert results['1_1_1/64-128_0-64_0-64'] == True

  fpath = os.path.join(cv.cloudpath, cv.key, '64-128_0-64_0-64')
  fpath = fpath.replace('file://', '') + '.gz'
  os.remove(fpath)

  results = cv.exists(defexists)
  assert len(results) == 2
  assert results['1_1_1/0-64_0-64_0-64'] == True
  assert results['1_1_1/64-128_0-64_0-64'] == False

def test_delete():

  # Bbox version
  delete_layer()
  cv, _ = create_layer(size=(128,64,64,1), offset=(0,0,0))

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
  cv, _ = create_layer(size=(128,64,64,1), offset=(0,0,0))

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
  cv, _ = create_layer(size=(128,64,64,1), offset=(0,0,0))

  try:
    results = cv.exists( np.s_[1:129, :, :] )
    print(results)
  except exceptions.OutOfBoundsError:
    pass
  else:
    assert False

def test_delete_black_uploads():
  for parallel in (1,2):
    delete_layer()
    cv, _ = create_layer(size=(256,256,256,1), offset=(0,0,0))

    ls = os.listdir('/tmp/removeme/layer/1_1_1/')
    assert len(ls) == 64

    cv.parallel = parallel
    cv.delete_black_uploads = True
    cv[64:64+128,64:64+128,64:64+128] = 0

    ls = os.listdir('/tmp/removeme/layer/1_1_1/')
    assert len(ls) == (64 - 8) 

    cv[64:64+128,64:64+128,64:64+128] = 0

    ls = os.listdir('/tmp/removeme/layer/1_1_1/')
    assert len(ls) == (64 - 8) 

    cv.image.background_color = 1
    cv[:] = 1 
    ls = os.listdir('/tmp/removeme/layer/1_1_1/')
    assert len(ls) == 0

    cv[:] = 0
    ls = os.listdir('/tmp/removeme/layer/1_1_1/')
    assert len(ls) == 64


def test_transfer():
  # Bbox version
  delete_layer()
  cv, _ = create_layer(size=(128,64,64,1), offset=(0,0,0))

  img = cv[:]

  cv.transfer_to('file:///tmp/removeme/transfer/', cv.bounds)

  ls = os.listdir('/tmp/removeme/transfer/1_1_1/')

  assert '0-64_0-64_0-64.gz' in ls
  assert len(ls) == 2

  assert os.path.exists('/tmp/removeme/transfer/info')
  assert os.path.exists('/tmp/removeme/transfer/provenance')

  dcv = CloudVolume("file:///tmp/removeme/transfer")
  dcv.info["dont_touch_me_bro"] = True
  dcv.commit_info()

  cv.transfer_to('file:///tmp/removeme/transfer/', cv.bounds)
  dcv.refresh_info()

  assert 'dont_touch_me_bro' in dcv.info

  assert np.all(img == dcv[:])

def test_cdn_cache_control():
  delete_layer()
  create_layer(size=(128,10,10,1), offset=(0,0,0))

  assert cdn_cache_control(None) == 'max-age=3600, s-max-age=3600'
  assert cdn_cache_control(0) == 'no-cache'
  assert cdn_cache_control(False) == 'no-cache'
  assert cdn_cache_control(True) == 'max-age=3600, s-max-age=3600'

  assert cdn_cache_control(1337) == 'max-age=1337, s-max-age=1337'
  assert cdn_cache_control('private, must-revalidate') == 'private, must-revalidate'

  try:
    cdn_cache_control(-1)
  except ValueError:
    pass
  else:
    assert False

def test_bbox_to_mip():
  info = {
    'data_type': 'uint8',
    'mesh': '',
    'num_channels': 1,
    'scales': [
      { 
        'chunk_sizes': [[64, 64, 1]],
        'encoding': 'raw',
        'key': '4_4_40',
        'resolution': [4, 4, 40],
        'size': [1024, 1024, 32],
        'voxel_offset': [35, 0, 1],
      },
      {
        'chunk_sizes': [[64, 64, 1]],
        'encoding': 'raw',
        'key': '8_8_40',
        'resolution': [8, 8, 40],
        'size': [512, 512, 32],
        'voxel_offset': [17, 0, 1],
      },
      {
        'chunk_sizes': [[64, 64, 1]],
        'encoding': 'raw',
        'key': '16_16_40',
        'resolution': [16, 16, 40],
        'size': [256, 256, 32],
        'voxel_offset': [8, 0, 1],
      },
      {
        'chunk_sizes': [[64, 64, 1]],
        'encoding': 'raw',
        'key': '32_32_40',
        'resolution': [32, 32, 40],
        'size': [128, 128, 32],
        'voxel_offset': [4, 0, 1],
      },
    ],
    'type': 'image'
  }
  
  cv = CloudVolume('file:///tmp/removeme/bbox_to_mip', info=info)

  bbox = Bbox( (35,0,1), (1024, 1024, 32))
  res = cv.bbox_to_mip(bbox, 0, 3)
  assert res.minpt.x == 4
  assert res.minpt.y == 0
  assert res.minpt.z == 1

  bbox = Bbox( (4, 0, 1), (128, 128, 32) )
  res = cv.bbox_to_mip(bbox, 3, 0)
  assert res.minpt.x == 32
  assert res.minpt.y == 0
  assert res.minpt.z == 1  

  res = cv.bbox_to_mip(bbox, 0, 0)
  assert res == bbox

def test_slices_from_global_coords():
  delete_layer()
  cv, _ = create_layer(size=(1024, 1024, 5, 1), offset=(7,0,0))

  scale = cv.info['scales'][0]
  scale = copy.deepcopy(scale)
  scale['voxel_offset'] = [ 3, 0, 0 ]
  scale['volume_size'] = [ 512, 512, 5 ]
  scale['resolution'] = [ 2, 2, 1 ]
  scale['key'] = '2_2_1'
  cv.info['scales'].append(scale)
  cv.commit_info()

  assert len(cv.available_mips) == 2

  cv.mip = 1
  slices = cv.slices_from_global_coords( Bbox( (100, 100, 1), (500, 512, 2) ) )
  result = Bbox.from_slices(slices)
  assert result == Bbox( (50, 50, 1), (250, 256, 2) )

  cv.mip = 0
  slices = cv.slices_from_global_coords( Bbox( (100, 100, 1), (500, 512, 2) ) )
  result = Bbox.from_slices(slices)
  assert result == Bbox( (100, 100, 1), (500, 512, 2) )

  slices = cv.slices_from_global_coords( np.s_[:,:,:] )
  result = Bbox.from_slices(slices)
  assert result == Bbox( (7, 0, 0), ( 1031, 1024, 5) )


def test_slices_to_global_coords():
  delete_layer()
  cv, _ = create_layer(size=(1024, 1024, 5, 1), offset=(7,0,0))

  scale = cv.info['scales'][0]
  scale = copy.deepcopy(scale)
  scale['voxel_offset'] = [ 3, 0, 0 ]
  scale['volume_size'] = [ 512, 512, 5 ]
  scale['resolution'] = [ 2, 2, 1 ]
  scale['key'] = '2_2_1'
  cv.info['scales'].append(scale)
  cv.commit_info()

  assert len(cv.available_mips) == 2

  cv.mip = 1
  slices = cv.slices_to_global_coords( Bbox( (100, 100, 1), (500, 512, 2) ) )

  result = Bbox.from_slices(slices)
  assert result == Bbox( (200, 200, 1), (1000, 1024, 2) )

  cv.mip = 0
  slices = cv.slices_to_global_coords( Bbox( (100, 100, 1), (500, 512, 2) ) )
  result = Bbox.from_slices(slices)
  assert result == Bbox( (100, 100, 1), (500, 512, 2) )

# def test_boss_download():
#   vol = CloudVolume('gs://seunglab-test/test_v0/image')
#   bossvol = CloudVolume('boss://automated_testing/test_v0/image')

#   vimg = vol[:,:,:5]
#   bimg = bossvol[:,:,:5]

#   assert np.all(bimg == vimg)
#   assert bimg.dtype == vimg.dtype

#   vol.bounded = False
#   vol.fill_missing = True
#   bossvol.bounded = False
#   bossvol.fill_missing = True

#   assert np.all(vol[-100:100,-100:100,-10:10] == bossvol[-100:100,-100:100,-10:10])

#   # BOSS using a different algorithm for creating downsamples
#   # so hard to compare 1:1 w/ pixels.
#   bossvol.bounded = True
#   bossvol.fill_missing = False
#   bossvol.mip = 1
#   bimg = bossvol[:,:,5:6]
#   assert np.any(bimg > 0)

def test_redirects():
  info = CloudVolume.create_new_info(
    num_channels=1, # Increase this number when we add more tests for RGB
    layer_type='image', 
    data_type='uint8', 
    encoding='raw',
    resolution=[ 1,1,1 ], 
    voxel_offset=[0,0,0], 
    volume_size=[128,128,64],
    mesh='mesh', 
    chunk_size=[ 64,64,64 ],
  )

  vol = CloudVolume('file:///tmp/cloudvolume/redirects_0', mip=0, info=info)
  vol.commit_info()
  vol.refresh_info()

  vol.info['redirect'] = 'file:///tmp/cloudvolume/redirects_0'
  vol.commit_info()
  vol.refresh_info()

  del vol.info['redirect']

  for i in range(0, 10):
    info['redirect'] = 'file:///tmp/cloudvolume/redirects_' + str(i + 1)  
    vol = CloudVolume('file:///tmp/cloudvolume/redirects_' + str(i), mip=0, info=info)
    vol.commit_info()
  else:
    del vol.info['redirect']
    vol.commit_info()

  vol = CloudVolume('file:///tmp/cloudvolume/redirects_0', mip=0)

  assert vol.cloudpath == 'file:///tmp/cloudvolume/redirects_9'

  info['redirect'] = 'file:///tmp/cloudvolume/redirects_10'
  vol = CloudVolume('file:///tmp/cloudvolume/redirects_9', mip=0, info=info)
  vol.commit_info()

  try:
    CloudVolume('file:///tmp/cloudvolume/redirects_0', mip=0)  
    assert False 
  except exceptions.TooManyRedirects:
    pass

  vol = CloudVolume('file:///tmp/cloudvolume/redirects_9', max_redirects=0)
  del vol.info['redirect']
  vol.commit_info()

  vol = CloudVolume('file:///tmp/cloudvolume/redirects_5', max_redirects=0)  
  vol.info['redirect'] = 'file:///tmp/cloudvolume/redirects_1'
  vol.commit_info()

  try:
    vol = CloudVolume('file:///tmp/cloudvolume/redirects_5')
    assert False
  except exceptions.CyclicRedirect:
    pass

  vol.info['redirect'] = 'file:///tmp/cloudvolume/redirects_6'
  vol.commit_info()

  vol = CloudVolume('file:///tmp/cloudvolume/redirects_1')

  try:
    vol[:,:,:] = 1
    assert False 
  except exceptions.ReadOnlyException:
    pass

  for i in range(0, 10):
    delete_layer('/tmp/cloudvolume/redirects_' + str(i))  
  
@pytest.mark.parametrize("compress_method", ('', None, True, False, 'gzip', 'br'))
def test_compression_methods(compress_method):
  path = "file:///tmp/cloudvolume/test" + '-' + str(TEST_NUMBER)


  # create a NG volume
  info = CloudVolume.create_new_info(
      num_channels=1,
      layer_type="image",
      data_type="uint16",  # Channel images might be 'uint8'
      encoding="raw",  # raw, jpeg, compressed_segmentation, fpzip, kempressed
      resolution=[4, 4, 40],  # Voxel scaling, units are in nanometers
      voxel_offset=[0, 0, 0],  # x,y,z offset in voxels from the origin
      chunk_size=[512, 512, 16],  # units are voxels
      volume_size=[512 * 4, 512 * 4, 16 * 40],  
  )
  vol = CloudVolume(path, info=info, compress=compress_method)
  vol.commit_info()

  image = np.random.randint(low=0, high=2 ** 16 - 1, size=(512, 512, 16, 1), dtype="uint16")

  vol[0:512, 0:512, 0:16] = image

  image_test = vol[0:512, 0:512, 0:16]
  
  delete_layer('/tmp/cloudvolume/test' + '-' + str(TEST_NUMBER))

  assert vol.compress == compress_method
  assert np.array_equal(image_test, image)

def test_mip_locking():
  delete_layer()
  cv, _ = create_layer(size=(1024, 1024, 2, 1), offset=(0,0,0))

  cv.meta.lock_mips(0)
  cv.meta.lock_mips([0])

  try:
    cv[:,:,:] = 0
    assert False 
  except ReadOnlyException:
    pass 

  cv.meta.unlock_mips(0)
  cv.meta.unlock_mips([0])

  cv[:,:,:] = 0

  try:
    cv.meta.lock_mips(1)
    assert False 
  except ValueError:
    pass 

  try:
    cv.meta.unlock_mips(1)
    assert False 
  except ValueError:
    pass 

  cv.add_scale((2,2,1))
  cv.commit_info()
  cv.mip = 1 
  cv[:] = 1

  cv.meta.lock_mips([0,1])

  try:
    cv[:,:,:] = 1
    assert False 
  except ReadOnlyException:
    pass 

  cv.mip = 0

  try:
    cv[:,:,:] = 1
    assert False 
  except ReadOnlyException:
    pass 

@pytest.mark.parametrize("green", (True, False))
def test_threaded_exceptions(green):
  info = {
    'type': 'image',
    'data_type': 'float32', 
    'num_channels': 1, 
    'scales': [
      {'chunk_sizes': [[1024, 1024, 1]], 
      'encoding': 'raw', 
      'key': '4_4_40', 
      'resolution': [4, 4, 40], 
      'size': [1024, 1024, 3], 'voxel_offset': [0, 0, 0]
    }], 
  }
  cv = CloudVolume("file:///tmp/removeme/green_exceptions", info=info, green_threads=green)
  
  try:
    cv[:]
    assert False
  except exceptions.EmptyVolumeException:
    pass


