from cloudvolume.datasource.precomputed.sharding import ShardingSpecification
from cloudvolume.exceptions import SpecViolation

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
