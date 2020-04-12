from __future__ import print_function

from collections import namedtuple, defaultdict
import copy
import json
import struct

import numpy as np
import pylru
from tqdm import tqdm

from . import mmh3
from ... import compression
from ...lib import jsonify, toiter
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

  def clone(self):
    return ShardingSpecification.from_dict(self.to_dict())

  def index_length(self):
    return int((2 ** self.minishard_bits) * 16)

  @property
  def hash(self):
    return self._hash

  @hash.setter
  def hash(self, val):
    if val == 'identity':
      self.hashfn = lambda x: uint64(x)
    elif val == 'murmurhash3_x86_128':
      self.hashfn = lambda x: uint64(mmh3.hash64(uint64(x).tobytes(), x64arch=False)[0]) 
    else:
      raise SpecViolation("hash {} must be either 'identity' or 'murmurhash3_x86_128'".format(val))

    self._hash = val

  @property
  def preshift_bits(self):
    return self._preshift_bits
  
  @preshift_bits.setter
  def preshift_bits(self, val):
    self._preshift_bits = uint64(val) 

  @property
  def shard_bits(self):
    return self._shard_bits
  
  @shard_bits.setter
  def shard_bits(self, val):
    self._shard_bits = uint64(val) 

  @property
  def minishard_bits(self):
    return self._minishard_bits
  
  @minishard_bits.setter
  def minishard_bits(self, val):
    val = uint64(val)
    self.minishard_mask = self.compute_minishard_mask(val)
    self._minishard_bits = uint64(val)

  def compute_minishard_mask(self, val):
    if val < 0:
      raise ValueError(str(val) + " must be greater or equal to than zero.")
    elif val == 0:
      return uint64(0)

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

  def to_json(self):
    return jsonify(self.to_dict())

  @classmethod
  def from_dict(cls, vals):
    vals = copy.deepcopy(vals)
    vals['type'] = vals['@type']
    del vals['@type']
    return cls(**vals)

  def to_dict(self):
    return {
      '@type': self.type,
      'preshift_bits': self.preshift_bits,
      'hash': self.hash,
      'minishard_bits': self.minishard_bits,
      'shard_bits': self.shard_bits,
      'minishard_index_encoding': self.minishard_index_encoding,
      'data_encoding': self.data_encoding,
    }

  def compute_shard_location(self, key):
    chunkid = uint64(key) >> uint64(self.preshift_bits)
    chunkid = self.hashfn(chunkid)
    minishard_number = uint64(chunkid & self.minishard_mask)
    shard_number = uint64((chunkid & self.shard_mask) >> uint64(self.minishard_bits))
    shard_number = format(shard_number, 'x').zfill(int(np.ceil(self.shard_bits / 4.0)))
    remainder = chunkid >> uint64(self.minishard_bits + self.shard_bits)

    return ShardLocation(shard_number, minishard_number, remainder)

  def synthesize_shards(self, data, progress=False):
    """
    Given this specification and a comprehensive listing of
    all the items that could be combined into a given shard,
    synthesize the shard files for this set of labels.

    data: { label: binary, ... }

    e.g. { 5: b'...', 7: b'...' }

    Returns: {
      $filename: binary data,
    }
    """
    return synthesize_shard_files(self, data, progress)

  def synthesize_shard(self, labels, progress=False, presorted=False):
    """
    Assemble a shard file from a group of labels that all belong in the same shard.

    Assembles the .shard file like:
    [ shard index; minishards; all minishard indices ]

    label_group: 
      If presorted is True:
        { minishardno: { label: binary, ... }, ... }
      If presorted is False:
        { label: binary }
    progress: show progress bars

    Returns: binary representing a shard file 
    """
    return synthesize_shard_file(self, labels, progress, presorted)

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
    
  def __str__(self):
    return "ShardingSpecification::" + str(self.to_dict())

class ShardReader(object):
  def __init__(
    self, meta, cache, spec,
    shard_index_cache_size=512,
    minishard_index_cache_size=128,
  ):
    """
    Reads standard Precomputed shard files. 

    meta: a PrecomputedMetadata class
    cache: a CacheService instance
    spec: a ShardingSpecification instance

    shard_index_cache_size: size of LRU cache for fixed indices 
    minishard_index_cache_size: size of LRU cache for minishard indices
    """
    self.meta = meta
    self.cache = cache
    self.spec = spec

    self.shard_index_cache = pylru.lrucache(shard_index_cache_size)
    self.minishard_index_cache = pylru.lrucache(minishard_index_cache_size)

  def get_filename(self, label):
    return self.compute_shard_location(label)[0]

  def compute_shard_location(self, label):
    """
    Returns (filename, shard_number) for meshes and skeletons. 
    Images require a different scheme.
    """
    shard_loc = self.spec.compute_shard_location(label)
    filename = str(shard_loc.shard_number) + '.shard'
    return (filename, shard_loc.minishard_number)

  def get_index(self, filename, path=""):
    """
    Retrieves the shard index which is used for 
    locating the appropriate minishard index.

    Returns: 2^minishard_bits entries of a uint64 
      array of [[ byte start, byte end ], ... ] 
    """
    index_path = self.meta.join(path, filename)
    alias_path = self.meta.join(path, filename.replace('.shard', '.index'))

    if filename in self.shard_index_cache:
      return self.shard_index_cache[filename]

    index_length = self.spec.index_length()

    binary = self.cache.download_single_as(
      index_path, alias_path,
      start=0, end=index_length,
      compress=False
    )

    if binary is None or len(binary) != index_length:
      binary_bytes = 0 if binary is None else len(binary)
      raise SpecViolation(
        filename + " was an incorrect length ({}) for this specification ({}).".format(
          binary_bytes, index_length
        ))
    
    index = np.frombuffer(binary, dtype=np.uint64)
    index = index.reshape( (index.size // 2, 2), order='C' )
    self.shard_index_cache[filename] = index
    return index

  def get_minishard_index(self, filename, index, minishard_no, path=""):
    """
    Retrieves the minishard index for a given minishard number.

    Returns: uint64 Nx3 array with multiple rows of [segid, byte start, byte end]
    """
    index_offset = self.spec.index_length()
    bytes_start, bytes_end = index[minishard_no]

    # most typically: [0,0] for an incomplete shard
    if bytes_start == bytes_end:
      return None

    bytes_start += index_offset
    bytes_end += index_offset
    bytes_start, bytes_end = int(bytes_start), int(bytes_end)

    full_path = self.meta.join(self.meta.cloudpath, path)

    cache_key = (filename, bytes_start, bytes_end)
    if cache_key in self.minishard_index_cache:
      return self.minishard_index_cache[cache_key]

    with SimpleStorage(full_path) as stor:
      minishard_index = stor.get_file(filename, start=bytes_start, end=bytes_end)

    if self.spec.minishard_index_encoding != 'raw':
      minishard_index = compression.decompress(
        minishard_index, encoding=self.spec.minishard_index_encoding, filename=filename
      )

    minishard_index = np.copy(np.frombuffer(minishard_index, dtype=np.uint64))
    minishard_index = minishard_index.reshape( (3, len(minishard_index) // 3), order='C' ).T

    for i in range(1, minishard_index.shape[0]):
      minishard_index[i, 0] += minishard_index[i-1, 0]
      minishard_index[i, 1] += minishard_index[i-1, 1] + minishard_index[i-1, 2]

    self.minishard_index_cache[cache_key] = minishard_index
    return minishard_index 

  def exists(self, labels, path="", return_byte_range=False):
    """
    Checks a shard's minishard index for whether a file exists.

    If return_byte_range = False:
      OUTPUT = SHARD_FILEPATH or None if not exists
    Else:
      OUTPUT = [ SHARD_FILEPATH or None, byte_start, num_bytes ]

    Returns:
      If labels is not an iterable:
        return OUTPUT
      Else:
        return { label_1: OUTPUT, label_2: OUTPUT, ... }
    """
    return_one = False

    try:
      iter(labels)
    except TypeError:
      return_one = True

    results = {}
    for label in set(toiter(labels)):
      filename, minishard_number = self.compute_shard_location(label)
      
      filepath = self.meta.join(path, filename)

      if self.cache.enabled:
        cached = self.cache.has(self.meta.join(path, str(label)), progress=False)
        if cached is not None:
          results[label] = filepath
          continue

      index = self.get_index(filename, path)

      minishard_index = self.get_minishard_index(
        filename, index, 
        minishard_number, path
      )

      if minishard_index is None:
        results[label] = None
        continue

      idx = np.where(minishard_index[:,0] == label)[0]
      if len(idx) == 0:
        results[label] = None
      else:
        if return_byte_range:
          _, offset, size = minishard_index[idx,:]
          results[label] = [ filepath, offset, size ]
        else:
          results[label] = filepath

    if return_one:
      return results[label]
    return results

  def get_data(self, label, path=""):
    filename, minishard_number = self.compute_shard_location(label)
    
    if self.cache.enabled:
      cached = self.cache.get_single(self.meta.join(path, str(label)), progress=False)
      if cached is not None:
        return cached

    index = self.get_index(filename, path)

    minishard_index = self.get_minishard_index(
      filename, index, 
      minishard_number, path
    )

    if minishard_index is None:
      return None

    idx = np.where(minishard_index[:,0] == label)[0]
    if len(idx) == 0:
      return None
    else:
      idx = idx[0]

    _, offset, size = minishard_index[idx,:]

    index_offset = self.spec.index_length()
    offset = int(offset + index_offset)
       
    full_path = self.meta.join(self.meta.cloudpath, path)

    with SimpleStorage(full_path) as stor:
      binary = stor.get_file(filename, start=offset, end=int(offset + size))

    if self.spec.data_encoding != 'raw':
      binary = compression.decompress(binary, encoding=self.spec.data_encoding, filename=filename)
      
    if self.cache.enabled:
      self.cache.put_single(self.meta.join(path, str(label)), binary, progress=False)

    return binary

def synthesize_shard_files(spec, data, progress=False):
  """
  From a set of data guaranteed to constitute one or more
  complete and comprehensive shards (no partial shards) 
  return a set of files ready for upload.

  WARNING: This function is only appropriate for Precomputed
  meshes and skeletons. Use the synthesize_shard_file (singular)
  function to create arbitrarily named and assigned shard files.

  spec: a ShardingSpecification
  data: { label: binary, ... }

  Returns: { filename: binary, ... }
  """
  shard_groupings = defaultdict(lambda: defaultdict(dict))
  pbar = tqdm(
    data.items(), 
    desc='Creating Shard Groupings', 
    disable=(not progress)
  )

  for label, binary in pbar:
    loc = spec.compute_shard_location(label)
    shard_groupings[loc.shard_number][loc.minishard_number][label] = binary

  shard_files = {}

  pbar = tqdm(
    shard_groupings.items(), 
    desc="Synthesizing Shard Files", 
    disable=(not progress)
  )

  for shardno, shardgrp in pbar:
    filename = str(shardno) + '.shard'
    shard_files[filename] = synthesize_shard_file(spec, shardgrp, progress=(progress > 1), presorted=True)

  return shard_files

# NB: This is going to be memory hungry and can be optimized
def synthesize_shard_file(spec, label_group, progress=False, presorted=False):
  """
  Assemble a shard file from a group of labels that all belong in the same shard.

  Assembles the .shard file like:
  [ shard index; minishards; all minishard indices ]

  spec: ShardingSpecification
  label_group: 
    If presorted is True:
      { minishardno: { label: binary, ... }, ... }
    If presorted is False:
      { label: binary }
  progress: show progress bars

  Returns: binary representing a shard file
  """
  minishardnos = []
  minishard_indicies = []
  minishards = []

  if presorted:
    minishard_mapping = label_group
  else:
    minishard_mapping = defaultdict(dict)
    pbar = tqdm(label_group.items(), disable=(not progress), desc="Assigning Minishards")
    for label, binary in pbar:
      loc = spec.compute_shard_location(label)
      minishard_mapping[loc.minishard_number][label] = binary

  del label_group

  for minishardno, minishardgrp in tqdm(minishard_mapping.items(), desc="Minishard Indices", disable=(not progress)):
    labels = sorted([ int(label) for label in minishardgrp.keys() ])
    if len(labels) == 0:
      continue

    minishard_index = np.zeros( (3, len(labels)), dtype=np.uint64, order='C')
    minishard = b''
    
    # label and offset are delta encoded
    last_label = 0
    for i, label in enumerate(labels):
      binary = minishardgrp[label]
      if spec.data_encoding != 'raw':
        binary = compression.compress(binary, method=spec.data_encoding)

      minishard_index[0, i] = label - last_label
      minishard_index[1, i] = 0 # minishard_index[2, i - 1]
      minishard_index[2, i] = len(binary)
      minishard += binary
      last_label = label
      del minishardgrp[label]
    
    minishardnos.append(minishardno)
    minishard_indicies.append(minishard_index) 
    minishards.append(minishard)

  del minishard_mapping

  cum_minishard_size = 0
  for idx, minishard in zip(minishard_indicies, minishards):
    idx[1, 0] = cum_minishard_size
    cum_minishard_size += len(minishard)

  if progress:
    print("Partial assembly of minishard indicies and data... ", end="", flush=True)

  variable_index_part = [ idx.tobytes('C') for idx in minishard_indicies ]
  if spec.minishard_index_encoding != 'raw':
    variable_index_part = [ 
      compression.compress(idx, method=spec.minishard_index_encoding) \
      for idx in variable_index_part 
    ]

  data_part = b''.join(minishards)
  del minishards

  if progress:
    print("Assembled.")

  fixed_index = np.zeros( 
    (int(2 ** spec.minishard_bits), 2), 
    dtype=np.uint64, order='C'
  )

  start = len(data_part)
  end = len(data_part)
  for i, idx in zip(minishardnos, variable_index_part):
    start = end
    end += len(idx)
    fixed_index[i, 0] = start
    fixed_index[i, 1] = end

  if progress:
    print("Final assembly... ", end="", flush=True)

  # The order here is important. The fixed index must go first because the locations
  # of the other parts are calculated with it implicitly in front. The variable
  # index must go last because otherwise compressing it will affect offset of the
  # data it is attempting to index.

  result = fixed_index.tobytes('C') + data_part + b''.join(variable_index_part) 

  if progress:
    print("Done.")

  return result