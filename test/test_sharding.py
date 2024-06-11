import pytest

import operator
from functools import reduce

from cloudvolume import CloudVolume, Skeleton
from cloudvolume.storage import SimpleStorage
from cloudvolume.datasource.precomputed.image.common import (
  gridpoints, compressed_morton_code
)
from cloudvolume.datasource.precomputed.sharding import (
  ShardingSpecification, 
  ShardReader,
  compute_shard_params_for_hashed,
  compute_shard_params_for_image,
  image_shard_shape_from_spec,
)
from cloudvolume.exceptions import SpecViolation, EmptyVolumeException

from cloudvolume import Vec, lib, Bbox
from cloudvolume import exceptions
from layer_harness import delete_layer, create_layer

from cloudfiles import CloudFile

import numpy as np

def prod(x):
  return reduce(operator.mul, x, 1)

def test_actual_example_hash():
  spec = ShardingSpecification.from_dict({
    "@type" : "neuroglancer_uint64_sharded_v1",
    "data_encoding" : "gzip",
    "hash" : "murmurhash3_x86_128",
    "minishard_bits" : 11,
    "minishard_index_encoding" : "gzip",
    "preshift_bits" : 6,
    "shard_bits" : 7
  })

  spec.validate()

  shard_loc = spec.compute_shard_location(1822975381)
  assert shard_loc.shard_number == '42'
  assert shard_loc.minishard_number == 18

  spec = ShardingSpecification.from_dict({
    "@type" : "neuroglancer_uint64_sharded_v1",
    "data_encoding" : "gzip",
    "hash" : "murmurhash3_x86_128",
    "minishard_index_encoding" : "gzip",
    "preshift_bits" : 2,
    "minishard_bits" : 3,
    "shard_bits" : 3
  })

  spec.hash = 'identity'
  label = 0b10101010
  shard_loc = spec.compute_shard_location(label)
  minishard_no = 0b010
  shard_no = 0b101

  assert shard_loc.minishard_number == minishard_no
  assert shard_loc.shard_number == str(shard_no)

def test_compressed_morton_code():
  cmc = lambda coord: compressed_morton_code(coord, grid_size=(3,3,3))

  assert cmc((0,0,0)) == 0b000000
  assert cmc((1,0,0)) == 0b000001
  assert cmc((2,0,0)) == 0b001000
  try:
    cmc((3,0,0))
    assert False
  except ValueError:
    pass
  assert cmc((2,2,0)) == 0b011000
  assert cmc((2,2,1)) == 0b011100

  cmc = lambda coord: compressed_morton_code(coord, grid_size=(2,3,1))

  assert cmc((0,0,0)) == 0b000000
  assert cmc((1,0,0)) == 0b000001
  try:
    cmc((0,0,7))
    assert False
  except ValueError:
    pass
  assert cmc((1,2,0)) == 0b000101

  assert np.array_equal(cmc([(0,0,0), (1,2,0)]), [0b000000, 0b000101])

  cmc = lambda coord: compressed_morton_code(coord, grid_size=(4,4,1))
  assert cmc((3,3,0)) == 0b1111

  cmc = lambda coord: compressed_morton_code(coord, grid_size=(8,8,2))
  assert cmc((5,5,0)) == 0b1100011

def test_image_sharding_hash():
  spec = ShardingSpecification(
    type="neuroglancer_uint64_sharded_v1",
    data_encoding="gzip",
    hash="identity",
    minishard_bits=6,
    minishard_index_encoding="gzip",
    preshift_bits=9,
    shard_bits=16,
  ) 

  point = Vec(144689, 52487, 2829)
  volume_size = Vec(*[248832, 134144, 7063])
  chunk_size = Vec(*[128, 128, 16])

  grid_size = np.ceil(volume_size / chunk_size).astype(np.uint32)
  gridpt = np.ceil(point / chunk_size).astype(np.int32)
  code = compressed_morton_code(gridpt, grid_size)
  loc = spec.compute_shard_location(code)

  assert loc.shard_number == '458d'



def test_sharding_spec_validation():
  spec = ShardingSpecification(
    type="neuroglancer_uint64_sharded_v1",
    data_encoding="gzip",
    hash="murmurhash3_x86_128",
    minishard_bits=11,
    minishard_index_encoding="gzip",
    preshift_bits=6,
    shard_bits=7,
  ) 

  spec.validate()
  
  spec.minishard_bits = 0
  spec.shard_bits = 0
  spec.validate()
  
  spec.minishard_bits = 64
  spec.shard_bits = 0
  spec.validate()

  spec.minishard_bits = 0
  spec.shard_bits = 64
  spec.validate()

  spec.minishard_bits = 1
  spec.shard_bits = 64
  try:
    spec.validate()
    assert False
  except SpecViolation:
    pass

  spec.minishard_bits = 64
  spec.shard_bits = 1
  try:
    spec.validate()
    assert False
  except SpecViolation:
    pass

  spec.minishard_bits = 11
  spec.shard_bits = 7
  
  spec.hash = 'identity'
  spec.hash = 'murmurhash3_x86_128'
  try:
    spec.hash = 'murmurhash3_X86_128'
    assert False 
  except SpecViolation:
    pass

  try:
    spec.hash = 'something else'
    assert False
  except SpecViolation:
    pass

  try:
    spec.hash = ''
  except SpecViolation:
    pass

  spec = ShardingSpecification(
    type="neuroglancer_uint64_sharded_v1",
    data_encoding="gzip",
    hash="murmurhash3_x86_128",
    minishard_bits=11,
    minishard_index_encoding="gzip",
    preshift_bits=6,
    shard_bits=7,
  ) 

  spec.preshift_bits = 0
  spec.validate()

  spec.preshift_bits = 63
  spec.validate()

  spec.preshift_bits = 32
  spec.validate()

  try:
    spec.preshift_bits = 64
    spec.validate()
    assert False
  except SpecViolation:
    pass

  try:
    spec.preshift_bits = -1
    spec.validate()
    assert False
  except SpecViolation:
    pass

  spec.preshift_bits = 5

  spec.minishard_index_encoding = 'raw'
  spec.validate()

  spec.minishard_index_encoding = 'gzip'
  spec.validate()

  spec.data_encoding = 'raw'
  spec.validate()

  spec.data_encoding = 'gzip'
  spec.validate()

  try:
    spec.type = 'lol_my_type'
    spec.validate()
    assert False
  except SpecViolation:
    pass

def test_skeleton_fidelity():
  segid = 1822975381
  cv = CloudVolume('gs://seunglab-test/sharded')
  sharded_skel = cv.skeleton.get(segid)

  cf = CloudFile(f'gs://seunglab-test/sharded/skeletons/{segid}')
  binary = cf.get()

  unsharded_skel = Skeleton.from_precomputed(binary, 
    segid=1822975381, vertex_attributes=cv.skeleton.meta.info['vertex_attributes']
  )

  assert sharded_skel == unsharded_skel

def test_skeleton_shard_unshard():
  cv = CloudVolume('gs://seunglab-test/sharded')
  
  cv.skeleton.to_unsharded()
  assert not cv.skeleton.meta.is_sharded()
  cv.skeleton.to_unsharded()
  assert not cv.skeleton.meta.is_sharded()

  cv.skeleton.to_sharded(num_labels=1000)

  assert cv.skeleton.meta.is_sharded()
  cv.skeleton.to_sharded(num_labels=1000)
  assert cv.skeleton.meta.is_sharded()

def test_mesh_shard_unshard():
  cv = CloudVolume('gs://seunglab-test/sharded')
  
  cv.mesh.to_unsharded()
  assert not cv.mesh.meta.is_sharded()
  cv.mesh.to_unsharded()
  assert not cv.mesh.meta.is_sharded()

  cv.mesh.to_sharded(num_labels=1000)

  assert cv.mesh.meta.is_sharded()
  cv.mesh.to_sharded(num_labels=1000)
  assert cv.mesh.meta.is_sharded()

def test_image_fidelity():
  point = (142195, 64376, 3130)
  cv = CloudVolume('gs://seunglab-test/sharded')
  img = cv.download_point(point, mip=0, size=128)

  N_labels = np.unique(img).shape[0]

  assert N_labels == 144

@pytest.mark.parametrize("delete_black_uploads", [False, True])
def test_write_image_shard_nonempty(delete_black_uploads):
  delete_layer()
  cv, data = create_layer(size=(256,256,256,1), offset=(0,0,0))

  spec = {
    "@type" : "neuroglancer_uint64_sharded_v1",
    "data_encoding" : "gzip",
    "hash" : "murmurhash3_x86_128",
    "minishard_bits" : 1,
    "minishard_index_encoding" : "raw",
    "preshift_bits" : 3,
    "shard_bits" : 0
  }
  cv.scale['sharding'] = spec
  cv.delete_black_uploads = delete_black_uploads

  cv[:] = data
  sharded_data = cv[:]
  assert np.all(data == sharded_data)

  spec['shard_bits'] = 1
  try:
    cv[:] = data
    assert False
  except exceptions.AlignmentError:
    pass

@pytest.mark.parametrize("delete_black_uploads", [False,True])
@pytest.mark.parametrize("background_color", [0,5])
def test_write_image_shard_empty(delete_black_uploads, background_color):
  delete_layer()
  cv, data = create_layer(size=(256,256,256,1), offset=(0,0,0))
  data[:] = background_color
  cv[:] = data
  cv.background_color = background_color

  spec = {
    "@type" : "neuroglancer_uint64_sharded_v1",
    "data_encoding" : "gzip",
    "hash" : "murmurhash3_x86_128",
    "minishard_bits" : 1,
    "minishard_index_encoding" : "raw",
    "preshift_bits" : 3,
    "shard_bits" : 0
  }
  cv.scale['sharding'] = spec
  cv.delete_black_uploads = delete_black_uploads

  cv[:] = data

  if delete_black_uploads:
    with pytest.raises(EmptyVolumeException):
      sharded_data = cv[:]
    cv.fill_missing = True

  sharded_data = cv[:]

  assert np.all(data == sharded_data)

@pytest.mark.parametrize("delete_black_uploads", [False,True])
@pytest.mark.parametrize("background_color", [0,5])
def test_write_image_shard_partly_empty(delete_black_uploads, background_color):
  delete_layer()
  cv, data = create_layer(size=(256,256,256,1), offset=(0,0,0))
  data[:64,:64,:] = background_color
  cv[:] = data
  cv.background_color = background_color

  spec = {
    "@type" : "neuroglancer_uint64_sharded_v1",
    "data_encoding" : "gzip",
    "hash" : "murmurhash3_x86_128",
    "minishard_bits" : 1,
    "minishard_index_encoding" : "raw",
    "preshift_bits" : 3,
    "shard_bits" : 0
  }
  cv.scale['sharding'] = spec
  cv.delete_black_uploads = delete_black_uploads

  cv[:] = data

  if delete_black_uploads:
    with pytest.raises(EmptyVolumeException):
      sharded_data = cv[:]
    cv.fill_missing = True

  sharded_data = cv[:]

  assert np.all(data == sharded_data)

SCALES = [
  {
    'chunk_sizes': [[128, 128, 64]],
    'encoding': 'jpeg',
    'key': '48_48_30',
    'resolution': [48, 48, 30],
    'size': [1536, 1408, 2046],
    'voxel_offset': [0, 0, 0]
  },
  {
    'chunk_sizes': [[128, 128, 64]],
    'encoding': 'jpeg',
    'key': '24_24_30',
    'resolution': [24, 24, 30],
    'size': [3072, 2816, 2046],
    'voxel_offset': [0, 0, 0]
  },
  {
    'chunk_sizes': [[128, 128, 20]],
    'encoding': 'raw',
    'key': '4_4_40',
    'resolution': [4, 4, 40],
    'size': [40960, 40960, 990],
    'voxel_offset': [69632, 36864, 4855],
  },
]

@pytest.mark.parametrize("scale", SCALES)
def test_sharded_image_bits(scale):
  dataset_size = Vec(*scale["size"])
  chunk_size = Vec(*scale["chunk_sizes"][0])

  spec = compute_shard_params_for_image( 
    dataset_size=dataset_size,
    chunk_size=chunk_size,
    encoding=scale["encoding"],
    dtype=np.uint8
  )

  shape = image_shard_shape_from_spec(
    spec.to_dict(), dataset_size, chunk_size
  )

  shape = lib.min2(shape, dataset_size)
  dataset_bbox = Bbox.from_vec(dataset_size)
  gpts = list(gridpoints(dataset_bbox, dataset_bbox, chunk_size))
  grid_size = np.ceil(dataset_size / chunk_size).astype(np.int64)

  reader = ShardReader(None, None, spec)

  morton_codes = compressed_morton_code(gpts, grid_size)
  min_num_shards = prod(dataset_size / shape)
  max_num_shards = prod(np.ceil(dataset_size / shape))
  
  assert 0 < min_num_shards <= 2 ** spec.shard_bits
  assert 0 < max_num_shards <= 2 ** spec.shard_bits

  real_num_shards = len(set(map(reader.get_filename, morton_codes)))

  assert min_num_shards <= real_num_shards <= max_num_shards

def test_broken_dataset():
  """
  This dataset was previously returning 19 total bits
  when 20 were needed to cover all the morton codes.
  """
  scale = {
    'chunk_sizes': [[128, 128, 20]],
    'encoding': 'raw',
    'key': '16_16_40',
    'resolution': [16, 16, 40],
    'size': [10240,10240,990],
    'voxel_offset': [17408,9216,4855],
  }

  dataset_size = Vec(*scale["size"])
  chunk_size = Vec(*scale["chunk_sizes"][0])

  spec = compute_shard_params_for_image( 
    dataset_size=dataset_size,
    chunk_size=chunk_size,
    encoding="jpeg",
    dtype=np.uint8
  )
  total_bits = spec.shard_bits + spec.minishard_bits + spec.preshift_bits
  assert total_bits == 20

def test_shard_bits_calculation_for_hashed():
  sb, msb, psb = compute_shard_params_for_hashed(
    num_labels=10**9, 
    shard_index_bytes=2**13, 
    minishard_index_bytes=2**15
  )
  assert psb == 0
  assert msb == 9
  assert sb == 11

  sb, msb, psb = compute_shard_params_for_hashed(
    num_labels=10**6, 
    shard_index_bytes=2**13, 
    minishard_index_bytes=2**15
  )
  assert psb == 0
  assert msb == 9
  assert sb == 1

  sb, msb, psb = compute_shard_params_for_hashed(
    num_labels=10**7, 
    shard_index_bytes=2**13, 
    minishard_index_bytes=2**15
  )
  assert psb == 0
  assert msb == 9
  assert sb == 4

  sb, msb, psb = compute_shard_params_for_hashed(
    num_labels=1000, 
    shard_index_bytes=2**13, 
    minishard_index_bytes=2**15
  )
  assert psb == 0
  assert msb == 0
  assert sb == 0

  sb, msb, psb = compute_shard_params_for_hashed(
    num_labels=1000, 
    shard_index_bytes=2**13, 
    minishard_index_bytes=2**15,
    min_shards=1000,
  )
  assert psb == 0
  assert msb == 0
  assert sb == 10

  sb, msb, psb = compute_shard_params_for_hashed(
    num_labels=0, 
    shard_index_bytes=0, 
    minishard_index_bytes=0
  )
  assert psb == 0
  assert msb == 0
  assert sb == 0

  sb, msb, psb = compute_shard_params_for_hashed(
    num_labels=10000, 
    shard_index_bytes=2**13, 
    minishard_index_bytes=2**15
  )
  assert psb == 0
  assert msb == 3
  assert sb == 0

  sb, msb, psb = compute_shard_params_for_hashed(
    num_labels=10**9, 
    shard_index_bytes=2**10, 
    minishard_index_bytes=2**15
  )
  assert psb == 0
  assert msb == 6
  assert sb == 14

  sb, msb, psb = compute_shard_params_for_hashed(
    num_labels=10**9, 
    shard_index_bytes=2**13, 
    minishard_index_bytes=2**13
  )
  assert psb == 0
  assert msb == 9
  assert sb == 13

