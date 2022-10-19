from typing import Optional

from collections import namedtuple, defaultdict
import copy
import json
from operator import itemgetter
from os.path import basename
import struct

import numpy as np
from tqdm import tqdm

from cloudfiles import CloudFiles

from . import mmh3
from ... import compression
from ...lib import jsonify, toiter, first
from ...lru import LRU
from ...exceptions import SpecViolation, EmptyFileException

ShardLocation = namedtuple('ShardLocation', 
  ('shard_number', 'minishard_number', 'remainder')
)

uint64 = np.uint64

class ShardingSpecification(object):
  def __init__(
    self, type, preshift_bits, 
    hash, minishard_bits, 
    shard_bits, 
    minishard_index_encoding='raw', 
    data_encoding='raw'
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

  def synthesize_shards(self, data, data_offset=None, progress=False):
    """
    Given this specification and a comprehensive listing of
    all the items that could be combined into a given shard,
    synthesize the shard files for this set of labels.

    data: { label: binary, ... }

    e.g. { 5: b'...', 7: b'...' }

    data_offset: { label: offset, ... }

    e.g. { 5: 1234, 7: 5678...' }

    Returns: {
      $filename: binary data,
    }
    """
    return synthesize_shard_files(self, data, data_offset, progress)

  def synthesize_shard(self, labels, data_offset=None, progress=False, presorted=False):
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
    return synthesize_shard_file(self, labels, data_offset, progress, presorted)

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
    green=False
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
    self.green = green

    self.shard_index_cache = LRU(shard_index_cache_size)
    self.minishard_index_cache = LRU(minishard_index_cache_size)

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
    indices = self.get_indices([ filename ], path, progress=False)
    index = list(indices.values())[0]
    if index is None:
      raise EmptyFileException(filename + " was zero bytes.")
    return index

  def get_indices(self, filenames, path="", progress=None):
    """
    For all given files, retrieves the shard index which 
    is used for locating the appropriate minishard indices.

    Returns: { 
      path_to_/filename.shard: 2^minishard_bits entries of a uint64 
            array of [[ byte start, byte end ], ... ],
      ...
    } 
    """
    filenames = toiter(filenames)
    filenames = [ self.meta.join(path, fname) for fname in filenames ]
    fufilled = { 
      fname: self.shard_index_cache[fname] \
      for fname in filenames \
      if fname in self.shard_index_cache  
    }

    requests = []
    for fname in filenames:
      if fname in fufilled:
        continue
      requests.append({
        'path': fname, 
        'local_alias': fname + '.index', 
        'start': 0, 
        'end': self.spec.index_length(),
      })

    progress = 'Shard Indices' if progress else False
    binaries = self.cache.download_as(requests, progress=progress)
    for (fname, start, end), content in binaries.items():
      try:
        index = self.decode_index(content, fname)
        self.shard_index_cache[fname] = index
        fufilled[fname] = index
      except EmptyFileException:
        self.shard_index_cache[fname] = None
        fufilled[fname] = None

    return fufilled

  def decode_index(self, binary, filename='Shard'):
    if binary is None or len(binary) == 0:
      raise EmptyFileException(filename + " was zero bytes.")
    elif len(binary) != self.spec.index_length():
      raise SpecViolation(
        filename + ": shard index was an incorrect length ({}) for this specification ({}).".format(
          len(binary), self.spec.index_length()
        ))
    
    index = np.frombuffer(binary, dtype=np.uint64)
    index = index.reshape( (index.size // 2, 2), order='C' )
    return index + self.spec.index_length()

  def decode_minishard_index(self, minishard_index, filename=''):
    """Returns [[label, offset, size], ... ] where offset and size are in bytes."""

    if self.spec.minishard_index_encoding != 'raw':
      minishard_index = compression.decompress(
        minishard_index, encoding=self.spec.minishard_index_encoding, filename=filename
      )

    minishard_index = np.copy(np.frombuffer(minishard_index, dtype=np.uint64))
    minishard_index = minishard_index.reshape( (3, len(minishard_index) // 3), order='C' ).T

    minishard_index[:,0] = np.cumsum(minishard_index[:,0])
    minishard_index[:,1] = np.cumsum(minishard_index[:,1])
    minishard_index[1:,1] += np.cumsum(minishard_index[:-1,2])
    minishard_index[:,1] += self.spec.index_length()

    return minishard_index 

  def get_minishard_index(self, filename, index, minishard_no, path=""):
    """
    Retrieves the minishard index for a given minishard number.

    Returns: uint64 Nx3 array with multiple rows of [segid, byte start, byte end]
    """
    res = self.get_minishard_indices(filename, index, minishard_no, path)
    return res[minishard_no]

  def get_minishard_indices(self, filename, index, minishard_nos, path=""):
    """
    Retrieves the minishard indices for a set of minishard numbers.

    Returns: { minishard_no: uint64 Nx3 array of [segid, byte start, byte end], ... }
    """
    res = self.get_minishard_indices_for_files(( (filename, index, minishard_nos), ), path)
    return res[basename(filename)]

  def get_minishard_indices_for_files(self, requests, path="", progress=None):
    """
    Fetches the specified minishard indices for all the specified files
    at once. This is required to get high performance as opposed to fetching
    the all minishard indices for a single file.

    requests: iterable of tuples
      [  (filename, index, minishard_numbers), ... ]

    Returns: map of filename -> minishard numbers -> minishard indices

    e.g. 
    {
      filename_1: {
          0: uint64 Nx3 array of [segid, byte start, byte end],
          1: ...,
      }
      filename_2: ...
    }
    """
    fufilled_by_filename = defaultdict(dict)
    msn_map = {}

    download_requests = []
    for filename, index, minishard_nos in requests:
      fufilled_requests, pending_requests = self.compute_minishard_index_requests(
        filename, index, minishard_nos, path
      ) 
      fufilled_by_filename[filename] = fufilled_requests
      for msn, start, end in pending_requests:
        msn_map[(basename(filename), start, end)] = msn

        filepath = self.meta.join(path, filename)

        download_requests.append({
          'path': filepath,
          'local_alias': '{}-{}.msi'.format(filepath, msn),
          'start': start,
          'end': end,
        })

    progress = 'Minishard Indices' if progress else False
    results = self.cache.download_as(download_requests, progress=progress)
  
    for (filename, start, end), content in results.items():
      filename = basename(filename)
      cache_key = (filename, start, end)
      msn = msn_map[cache_key]
      minishard_index = self.decode_minishard_index(content, filename)
      self.minishard_index_cache[cache_key] = minishard_index
      fufilled_by_filename[filename][msn] = minishard_index

    return fufilled_by_filename

  def compute_minishard_index_requests(self, filename, index, minishard_nos, path=""):
    """
    Helper method for get_minishard_indices_for_files. 
    Computes which requests must be made over the network vs can be fufilled from LRU cache.
    """
    minishard_nos = toiter(minishard_nos)

    if index is None:
      return ({ msn: None for msn in minishard_nos }, [])

    fufilled_requests = {}

    byte_ranges = {}
    for msn in minishard_nos:
      bytes_start, bytes_end = index[msn]

      # most typically: [0,0] for an incomplete shard
      if bytes_start == bytes_end:
        fufilled_requests[msn] = None
        continue

      bytes_start, bytes_end = int(bytes_start), int(bytes_end)
      byte_ranges[msn] = (bytes_start, bytes_end)

    full_path = self.meta.join(self.meta.cloudpath, path)

    pending_requests = []
    for msn, (bytes_start, bytes_end) in byte_ranges.items():
      cache_key = (filename, bytes_start, bytes_end)
      if cache_key in self.minishard_index_cache:
        fufilled_requests[msn] = self.minishard_index_cache[cache_key]
      else:
        pending_requests.append((msn, bytes_start, bytes_end))

    return (fufilled_requests, pending_requests)

  def exists(self, labels, path="", return_byte_range=False, progress=None):
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

    to_labels = defaultdict(list)
    to_all_labels = defaultdict(list)
    filename_to_minishard_num = defaultdict(list)

    for label in set(toiter(labels)):
      filename, minishard_number = self.compute_shard_location(label)
      to_labels[(filename, minishard_number)].append(label)
      to_all_labels[filename].append(label)
      filename_to_minishard_num[filename].append(minishard_number)

    indices = self.get_indices(to_all_labels.keys(), path, progress=progress)

    all_minishards = self.get_minishard_indices_for_files([ 
      (basename(filepath), index, filename_to_minishard_num[basename(filepath)]) \
      for filepath, index in indices.items()
    ], path, progress=progress)

    results = {}
    for filename, file_minishards in all_minishards.items():
      filepath = self.meta.join(path, filename)
      for mini_no, msi in file_minishards.items():
        labels = to_labels[(filename, mini_no)]

        for label in labels:
          if msi is None:
            results[label] = None
            continue

          idx = np.where(msi[:,0] == label)[0]
          if len(idx) == 0:
            results[label] = None
          else:
            if return_byte_range:
              _, offset, size = msi[idx,:][0]
              results[label] = [ filepath, int(offset), int(size) ]
            else:
              results[label] = filepath

    if return_one:
      return(list(results.values())[0])
    return results

  def disassemble_shard(self, shard):
    """
    Given an entire shard as a bytestring, convert 
    it into a dict of { label: byte content }.
    """
    index = self.decode_index(shard[:self.spec.index_length()])
    shattered = {}
    for start, end in index:
      start, end = int(start), int(end)
      if start == end:
        continue

      msi = self.decode_minishard_index(shard[start:end])
      for label, offset, size in msi:
        offset, size = int(offset), int(size)
        binary = shard[offset:offset+size]
        
        if self.spec.data_encoding != 'raw':
          binary = compression.decompress(binary, encoding=self.spec.data_encoding)
        
        shattered[label] = binary

    return shattered

  def get_data(
    self, label:int, path:str = "", 
    progress:Optional[bool] = None, parallel:int = 1,
    raw:bool = False
  ):
    """Fetches data from shards.

    label: one or more segment ids
    path: subdirectory path
    progress: display progress bars
    parallel: (int >= 0) use multiple processes
    raw: if true, don't decompress or decode stream

    Return: 
      if label is a scalar:
        a byte string
      else: (label is an iterable)
        {
          label_1: byte string,
          ....
        }
    """
    label, return_multiple = toiter(label, is_iter=True)
    label = set(( int(l) for l in label))
    if not label:
      return {}

    cached = {}
    if self.cache.enabled:
      cached = self.cache.get([ 
        self.meta.join(path, str(lbl)) for lbl in label
      ], progress=progress)

    results = {}
    for cloudpath, content in cached.items():
      lbl = int(basename(cloudpath))
      if content is not None:
        label.remove(lbl)
        
      results[lbl] = cached[cloudpath]

    del cached

    # { label: [ filename, byte start, num_bytes ] }
    exists = self.exists(label, path, return_byte_range=True, progress=progress)
    for k in list(exists.keys()):
      if exists[k] is None:
        results[k] = None
        del exists[k]

    key_label = { (basename(v[0]), v[1], v[2]): k for k,v in exists.items() }

    files = ( 
      { 'path': basename(ext[0]), 'start': int(ext[1]), 'end': int(ext[1]) + int(ext[2]) }
      for ext in exists.values()
    )

    # Requesting many individual shard chunks is slow, but due to z-ordering
    # we might be able to combine adjacent byte ranges. Especially helpful
    # when downloading entire shards!
    bundles = []
    for chunk in sorted(files, key=itemgetter("path", "start")):
      if not bundles or (chunk['path'] != bundles[-1]['path']) or (chunk['start'] != bundles[-1]['end']):
        bundles.append(dict(content=None, subranges=[], **chunk))
      else:
        bundles[-1]['end'] = chunk['end']

      bundles[-1]['subranges'].append({
          'start': chunk['start'],
          'length': chunk['end'] - chunk['start'],
          'slices': slice(chunk['start'] - bundles[-1]['start'], chunk['end'] - bundles[-1]['start'])
      })

    full_path = self.meta.join(self.meta.cloudpath, path)
    bundles_resp = CloudFiles(
      full_path, 
      progress=progress, 
      green=self.green,
      parallel=parallel,
    ).get(bundles)

    # Responses are not guaranteed to be in order of requests
    bundles_resp = { (r['path'], r['byte_range']): r for r in bundles_resp }

    binaries = {}
    for bundle_req in bundles:
      bundle_resp = bundles_resp[(bundle_req['path'], (bundle_req['start'], bundle_req['end']))]
      if bundle_resp['error']:
        raise bundle_resp['error']

      for chunk in bundle_req['subranges']:
        key = (bundle_req['path'], chunk['start'], chunk['length'])
        lbl = key_label[key]
        binaries[lbl] = bundle_resp['content'][chunk['slices']]

    del bundles
    del bundles_resp

    if not raw and self.spec.data_encoding != 'raw':
      for filepath, binary in tqdm(binaries.items(), desc="Decompressing", disable=(not progress)):
        if binary is None:
          continue
        binaries[filepath] = compression.decompress(
          binary, encoding=self.spec.data_encoding, filename=filepath
        )
    
    if self.cache.enabled:
      self.cache.put([ 
        (self.meta.join(path, str(filepath)), binary) for filepath, binary in binaries.items()
      ], progress=progress)

    results.update(binaries)

    if return_multiple:
      return results
    return first(results.values())

  def list_labels(self, filename, path="", size=False):
    """
    List all the labels in the index of a given shard file.

    size: (bool) if True, list the size in bytes of each label

    Returns: 
      if not size:
        np.uint64 array of labels 
      else:
        [ (label, size in bytes), ... ] in descending order of size
    """
    index = self.get_index(filename, path)
    all_minishard_nos = list(range(len(index)))
    minishard_indices = self.get_minishard_indices(filename, index, all_minishard_nos, path)
    minishard_indices = [  
      msi for msi in minishard_indices.values() if msi is not None
    ]
    if not size:
      labels = np.concatenate([  
        msi[:,0] for msi in minishard_indices
      ])
      return np.sort(labels)
    else:
      labels = np.concatenate([  
        msi[:,:]
        for msi in minishard_indices
      ])
      labels = [ (row[0], row[2]) for row in labels[:] ]
      return sorted(labels, key=lambda x: x[1], reverse=True)


def synthesize_shard_files(spec, data, data_offset=None, progress=False):
  """
  From a set of data guaranteed to constitute one or more
  complete and comprehensive shards (no partial shards) 
  return a set of files ready for upload.

  WARNING: This function is only appropriate for Precomputed
  meshes and skeletons. Use the synthesize_shard_file (singular)
  function to create arbitrarily named and assigned shard files.

  spec: a ShardingSpecification
  data: { label: binary, ... }
  data_offset: { label: offset, ... }

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
    shard_files[filename] = synthesize_shard_file(
        spec, shardgrp, data_offset, progress=(progress > 1), presorted=True)

  return shard_files

# NB: This is going to be memory hungry and can be optimized


def synthesize_shard_file(spec, label_group, data_offset=None, progress=False, presorted=False):
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
  data_offset: { label: offset, ... }
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
    minishard_components = []
    # label and offset are delta encoded
    last_label = 0
    for i, label in enumerate(labels):
      binary = minishardgrp[label]
      if spec.data_encoding != 'raw':
        binary = compression.compress(binary, method=spec.data_encoding)

      # delta encoded [label, offset, size]
      minishard_index[0, i] = label - last_label
      if data_offset is None:
        minishard_index[1, i] = 0 # minishard_index[2, i - 1]
        minishard_index[2, i] = len(binary)
      else:
        # add offset of the actual data if it exists
        minishard_index[1, i] = len(binary) - data_offset[label]
        minishard_index[2, i] = data_offset[label]

      minishard_components.append(binary)
      last_label = label
      del minishardgrp[label]

    minishard = b"".join(minishard_components)
    minishardnos.append(minishardno)
    minishard_indicies.append(minishard_index) 
    minishards.append(minishard)

  del minishard_mapping

  cum_minishard_size = 0
  for idx, minishard in zip(minishard_indicies, minishards):
    idx[1, 0] += cum_minishard_size
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
