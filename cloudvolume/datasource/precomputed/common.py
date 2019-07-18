from collections import namedtuple
import concurrent.futures
import copy
from functools import partial
import itertools
import json
import math
import multiprocessing as mp
import mmh3
import os
import posixpath
import signal

import numpy as np

from ...lib import xyzrange, min2, max2, Vec, Bbox
from ... import sharedmemory as shm
from ...exceptions import SpecViolation

# Used in sharedmemory to emulate shared memory on 
# OS X using a file, which has that facility but is 
# more limited than on Linux.
fs_lock = mp.Lock()

def parallel_execution(fn, items, parallel, cleanup_shm=None):
  def cleanup(signum, frame):
    if cleanup_shm:
      shm.unlink(cleanup_shm)

  prevsigint = signal.getsignal(signal.SIGINT)
  prevsigterm = signal.getsignal(signal.SIGTERM)

  signal.signal(signal.SIGINT, cleanup)
  signal.signal(signal.SIGTERM, cleanup)

  with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as executor:
    executor.map(fn, items)

  signal.signal(signal.SIGINT, prevsigint)
  signal.signal(signal.SIGTERM, prevsigterm)

def chunknames(bbox, volume_bbox, key, chunk_size, protocol=None):
  path = posixpath if protocol != 'file' else os.path

  for x,y,z in xyzrange( bbox.minpt, bbox.maxpt, chunk_size ):
    highpt = min2(Vec(x,y,z) + chunk_size, volume_bbox.maxpt)
    filename = "{}-{}_{}-{}_{}-{}".format(
      x, highpt.x,
      y, highpt.y, 
      z, highpt.z
    )
    yield path.join(key, filename)

def shade(dest_img, dest_bbox, src_img, src_bbox):
  """
  Shade dest_img at coordinates dest_bbox using the
  image contained in src_img at coordinates src_bbox.

  The buffer will only be painted in the overlapping
  region of the content.

  Returns: void
  """
  if not Bbox.intersects(dest_bbox, src_bbox):
    return

  spt = max2(src_bbox.minpt, dest_bbox.minpt)
  ept = min2(src_bbox.maxpt, dest_bbox.maxpt)
  dbox = Bbox(spt, ept) - dest_bbox.minpt

  ZERO3 = Vec(0,0,0)
  istart = max2(spt - src_bbox.minpt, ZERO3)
  iend = min2(ept - src_bbox.maxpt, ZERO3) + src_img.shape[:3]
  sbox = Bbox(istart, iend)

  while src_img.ndim < 4:
    src_img = src_img[..., np.newaxis]
  
  dest_img[ dbox.to_slices() ] = src_img[ sbox.to_slices() ]

def content_type(encoding):
  if encoding == 'jpeg':
    return 'image/jpeg'
  elif encoding in ('compressed_segmentation', 'fpzip', 'kempressed'):
    return 'image/x.' + encoding 
  return 'application/octet-stream'

def should_compress(encoding, compress, cache, iscache=False):
  if iscache and cache.compress != None:
    return cache.compress

  if compress is None:
    return 'gzip' if encoding in ('raw', 'compressed_segmentation') else None
  elif compress == True:
    return 'gzip'
  elif compress == False:
    return None
  else:
    return compress

def cdn_cache_control(val):
  """Translate cdn_cache into a Cache-Control HTTP header."""
  if val is None:
    return 'max-age=3600, s-max-age=3600'
  elif type(val) is str:
    return val
  elif type(val) is bool:
    if val:
      return 'max-age=3600, s-max-age=3600'
    else:
      return 'no-cache'
  elif type(val) is int:
    if val < 0:
      raise ValueError(
        'cdn_cache must be a positive integer, boolean, or string. Got: ' + str(val)
      )

    if val == 0:
      return 'no-cache'
    else:
      return 'max-age={}, s-max-age={}'.format(val, val)
  else:
    raise NotImplementedError(type(val) + ' is not a supported cache_control setting.')



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
    
