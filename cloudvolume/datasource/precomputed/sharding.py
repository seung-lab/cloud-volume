from collections import namedtuple
import copy
import json

import mmh3
import numpy as np
import struct

from ... import compression
from ...exceptions import SpecViolation
from ...storage import SimpleStorage

ShardLocation = namedtuple('ShardLocation', 
  ('shard_number', 'minishard_number', 'remainder')
)

uint64 = np.uint64

class ShardingSpecification(object):
  def __init__(
    self, type, preshift_bits, 
    hash, minishard_bits, 
    shard_bits, minishard_index_encoding, 
    data_encoding
  ):

    self.type = type 
    self.preshift_bits = uint64(preshift_bits)
    self.hash = hash 
    self.minishard_bits = uint64(minishard_bits)
    self.shard_bits = uint64(shard_bits)
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
    def apply_mmh3(x):
      x_hashed = mmh3.hash128(str(x)) # 128-bit uint
      return uint64(x_hashed & 0x0000000000000000ffffffffffffffff)

    if val == 'identity':
      self.hashfn = lambda x: uint64(x)
    elif val == 'murmurhash3_x86_128':
      self.hashfn = apply_mmh3
    else:
      raise SpecViolation("hash {} must be either 'identity' or 'murmurhash3_x86_128'".format(val))

    self._hash = val

  @property
  def minishard_bits(self):
    return self._minishard_bits
  
  @minishard_bits.setter
  def minishard_bits(self, val):
    val = uint64(val)
    self.minishard_mask = self.compute_minishard_mask(val)
    self._minishard_bits = uint64(val)

  def compute_minishard_mask(self, val):
    if val <= 0:
      raise ValueError(str(val) + " must be greater than zero.")

    minishard_mask = uint64(1)
    for i in range(val - uint64(1)):
      minishard_mask <<= uint64(1)
      minishard_mask |= uint64(1)
    return uint64(minishard_mask)

  def compute_shard_mask(self, shard_bits, minishard_bits):
    ones64 = uint64(0xffffffffffffffff)
    movement = uint64(minishard_bits + shard_bits)
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

  def compute_shard_location(self, key):
    chunkid = self.hashfn(uint64(key) >> uint64(self.preshift_bits))
    minishard_number = uint64(chunkid & self.minishard_mask)
    shard_number = uint64((chunkid & self.shard_mask) >> uint64(self.minishard_bits))
    shard_number = format(shard_number, 'x').zfill(int(np.ceil(self.shard_bits / 4.0)))
    remainder = chunkid >> uint64(self.minishard_bits + self.shard_bits)
    
    return ShardLocation(shard_number, minishard_number, remainder)

  def validate(self):
    if self.type not in ('neuroglancer_uint64_sharded_v1',):
      raise SpecViolation(
        "@type ({}) must be 'neuroglancer_uint64_sharded_v1'." \
        .format(self.type)
      )

    if not (64 > self.preshift_bits >= 0):
      raise SpecViolation("preshift_bits must be a whole number less than 64: {}".format(self.preshift_bits))

    if not (64 >= self.minishard_bits >= 0):
      raise SpecViolation("minishard_bits must be between 0 and 64 inclusive: {}".format(self.minishard_bits))

    if not (64 >= self.shard_bits >= 0):
      raise SpecViolation("shard_bits must be between 0 and 64 inclusive: {}".format(self.shard_bits))

    if self.minishard_bits + self.shard_bits > 64:
      raise SpecViolation(
        "minishard_bits and shard_bits must sum to less than or equal to 64: minishard_bits<{}> + shard_bits<{}> = {}".format(
        self.minishard_bits, self.shard_bits, self.minishard_bits + self.shard_bits
      ))

    if self.hash not in ('identity', 'murmurhash3_x86_128'):
      raise SpecViolation("hash {} must be either 'identity' or 'murmurhash3_x86_128'".format(self.hash))

    if self.minishard_index_encoding not in ('raw', 'gzip'):
      raise SpecViolation("minishard_index_encoding only supports values 'raw' or 'gzip'.")

    if self.data_encoding not in ('raw', 'gzip'):
      raise SpecViolation("data_encoding only supports values 'raw' or 'gzip'.")
    
class ShardReader(object):
  def __init__(self, meta, spec):
    self.meta = meta
    self.spec = spec

  def get_index(self, label):
    shard_loc = self.spec.compute_shard_location(label)
    with SimpleStorage(self.meta.full_path) as stor:
      filename = str(shard_loc.shard_number) + ".index"
      binary = stor.get_file(filename)

    index_length = (2 ** self.spec.minishard_bits) * 16

    if len(binary) != index_length:
      return SpecViolation(
        filename + " was an incorrect length ({}) for this specification ({}).".format(
          len(binary), index_length
        ))
    
    index = np.frombuffer(binary, dtype=uint64)
    return index.reshape( (index.size // 2, 2), order='C' )

  def get_data(self, label):
    shard_loc = self.spec.compute_shard_location(label)
    index = self.get_index(label)

    bytes_start, bytes_end = index[shard_loc.minishard_number]
    request_length = (bytes_end - bytes_start + 1)

    filename = shard_loc.shard_number + ".data"

    with SimpleStorage(self.meta.full_path) as stor:
      minishard_index = stor.get_file(filename, start=bytes_start, end=bytes_end)

    if self.spec.minishard_index_encoding == 'gzip':
      minishard_index = compression.decompress(minishard_index, encoding='gzip', filename=filename)

    minishard_index = np.frombuffer(minishard_index, dtype=np.uint64)
    minishard_index = minishard_index.reshape( (3, len(minishard_index) // 3), order='C' )
    minishard_index = np.add.accumulate(minishard_index.T) # elements are delta encoded

    offset = 0
    size = 0

    for chunk_id, off, sz in minishard_index:
      if chunk_id == label:
        offset = off
        size = sz
        break
    else:
      raise IndexError(label + " was not found in the minishard_index of " + filename)

    with SimpleStorage(self.meta.full_path) as stor:
      binary = stor.get_file(filename, start=offset, end=(offset + size - 1))

    if self.spec.data_encoding == 'gzip':
      return compression.decompress(binary, encoding='gzip', filename=filename)
    else:
      return binary









