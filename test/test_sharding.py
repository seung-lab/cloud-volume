from cloudvolume import CloudVolume, Skeleton
from cloudvolume.storage import SimpleStorage
from cloudvolume.datasource.precomputed.sharding import ShardingSpecification
from cloudvolume.exceptions import SpecViolation

import numpy as np

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

  with SimpleStorage('gs://seunglab-test/sharded') as stor:
    binary = stor.get_file('skeletons/' + str(segid))

  unsharded_skel = Skeleton.from_precomputed(binary, 
    segid=1822975381, vertex_attributes=cv.skeleton.meta.info['vertex_attributes']
  )

  assert sharded_skel == unsharded_skel




