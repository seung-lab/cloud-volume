from collections import namedtuple
import json

import mmh3

from ...exceptions import SpecViolation

ShardLocation = namedtuple('ShardLocation', 
  ('shard_number', 'minishard_number', 'remainder')
)

class ShardingSpecification(object):
  def __init__(
    self, type, preshift_bits, 
    hash, minishard_bits, 
    shard_bits, minishard_index_encoding, 
    data_encoding
  ):

    self.type = type 
    self.preshift_bits = int(preshift_bits)
    self.hash = hash 
    self.minishard_bits = int(minishard_bits)
    self.shard_bits = int(shard_bits)
    self.minishard_index_encoding = minishard_index_encoding
    self.data_encoding = data_encoding

    self.minishard_mask = self.compute_minishard_mask(self.minishard_bits)
    self.shard_mask = self.compute_shard_mask(self.shard_bits, self.minishard_bits)              

    self.validate()

  @property
  def hash(self):
    return self._hash

  @hash.setter
  def hash(self, val):
    if val == 'identity':
      self.hashfn = lambda x: np.uint64(x)
    elif val == 'murmurhash3_x86_128':
      self.hashfn = lambda x: np.uint64(int(mmh3.hash128(x)[8:16], base=16))
    else:
      raise SpecViolation("hash must be either 'identity' or 'murmurhash3_x86_128'")

    self._hash = val

  @property
  def minishard_bits(self):
    return self._minishard_bits
  
  @minishard_bits.setter
  def minishard_bits(self, val):
    self.minishard_mask = self.compute_minishard_mask(val)
    self._minishard_bits = int(val)

  def compute_minishard_mask(self, val):
    if val <= 0:
      raise ValueError(str(val) + " must be greater than zero.")

    minishard_mask = 1
    for i in range(val - 1):
      minishard_mask <<= 1
      minishard_mask |= 1
    return minishard_mask

  def compute_shard_mask(self, shard_bits, minishard_bits):
    ones64 = 0xffffffffffffffff
    movement = minishard_bits + shard_bits
    shard_mask = ~((ones64 >> movement) << movement)
    minishard_mask = self.compute_minishard_mask(minishard_bits)
    return shard_mask & (~minishard_mask)

  @classmethod
  def from_json(cls, vals):
    dct = json.loads(vals.decode('utf8'))
    return cls.from_dict(dct)

  @classmethod
  def from_dict(cls, vals):
    vals = copy.deepcopy(vals)
    vals['type'] = vals['@type']
    del vals['@type']
    return cls(**vals)

  def decode(self, key):
    chunkid = self.hashfn(int(key) >> int(self.preshift_bits))
    minishard_number = int(chunkid & self.minishard_mask)
    shard_number = int((chunkid & self.shard_mask) >> self.minishard_bits)
    shard_number = str(shard_number).zfill(int(np.ceil(self.shard_bits / 4.0)))
    remainder = chunkid >> (self.minishard_bits + self.shard_bits)
    
    return ShardLocation(shard_number, minishard_number, remainder)

  def validate(self):
    if self.type not in ('neuroglancer_uint64_sharded_v1',):
      raise SpecViolation(
        "@type ({}) must be 'neuroglancer_uint64_sharded_v1'." \
        .format(self.type)
      )

    if not (64 > self.preshift_bits >= 0):
      raise SpecViolation("preshift_bits must be a whole number less than 64: {}".format(self.preshift_bits))

    if not (64 > self.minishard_bits > 0):
      raise SpecViolation("minishard_bits must be a whole number less than 64: {}".format(self.minishard_bits))

    if not (64 > self.shard_bits > 0):
      raise SpecViolation("shard_bits must be a whole number less than 64: {}".format(self.shard_bits))

    if self.minishard_bits + self.shard_bits > 63:
      raise SpecViolation(
        "minishard_bits and shard_bits must leave room for a key: minishard_bits<{}> + shard_bits<{}> = {}".format(
        self.minishard_bits, self.shard_bits, self.minishard_bits + self.shard_bits
      ))

    if self.hash not in ('identity', 'murmurhash3_x86_128'):
      raise SpecViolation("hash must be either 'identity' or 'murmurhash3_x86_128'")

    if self.minishard_index_encoding not in ('raw', 'gzip'):
      raise SpecViolation("minishard_index_encoding only supports values 'raw' or 'gzip'.")

    if self.data_encoding not in ('raw', 'gzip'):
      raise SpecViolation("data_encoding only supports values 'raw' or 'gzip'.")
    
class ShardReader(object):
  def __init__(self, meta, spec):
    self.meta = meta
    self.spec = spec