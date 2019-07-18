import datetime
import os

import posixpath

import numpy as np

from cloudvolume import lib
from cloudvolume.exceptions import (
  SkeletonDecodeError, SkeletonEncodeError, 
  SkeletonUnassignedEdgeError
)
from cloudvolume.lib import red, Bbox
from cloudvolume.storage import Storage, SimpleStorage

from .common import cdn_cache_control

from ...skeleton import Skeleton


# class ShardedPrecomputedSkeletonSource(object):

#   def get(self, segid):

class UnshardedPrecomputedSkeletonSource(object):
  def __init__(self, meta, cache, config):
    self.meta = meta
    self.cache = cache
    self.config = config

  @property
  def path(self):
    return self.meta.path 

  def get(self, segids):
    """
    Retrieve one or more skeletons from the data layer.

    Example: 
      skel = vol.skeleton.get(5)
      skels = vol.skeleton.get([1, 2, 3])

    Raises SkeletonDecodeError on missing files or decoding errors.

    Required:
      segids: list of integers or integer

    Returns: 
      if segids is a list, returns list of Skeletons
      else returns a single Skeleton
    """
    list_return = True
    if type(segids) in (int, float):
      list_return = False
      segids = [ int(segids) ]

    compress = self.config.compress 
    if compress is None:
      compress = True

    results = self.cache.download(
      [ os.path.join(self.meta.path, str(segid)) for segid in segids ],
      compress=compress
    )
    missing = [ filename for filename, content in results.items() if content is None ]

    if len(missing):
      raise SkeletonDecodeError("File(s) do not exist: {}".format(", ".join(missing)))

    skeletons = []
    for filename, content in results.items():
      segid = int(os.path.basename(filename))
      try:
        skel = Skeleton.from_precomputed(content, segid=segid)
      except Exception as err:
        raise SkeletonDecodeError("segid " + str(segid) + ": " + err.message)
      skeletons.append(skel)

    if list_return:
      return skeletons

    if len(skeletons):
      return skeletons[0]

    return None

  def upload_raw(self, segid, vertices, edges, radii=None, vertex_types=None):
    skel = Skeleton(
      vertices, edges, radii, 
      vertex_types, segid=segid
    )
    return self.upload(skel)
    
  def upload(self, skeletons):
    if type(skeletons) == Skeleton:
      skeletons = [ skeletons ]

    files = [ (os.path.join(self.meta.path, str(skel.id)), skel.to_precomputed()) for skel in skeletons ]
    self.cache.upload(
      files=files, 
      subdir=self.meta.path,
      compress='gzip', 
      cache_control=cdn_cache_control(self.config.cdn_cache)
    )

class PrecomputedSkeletonMetadata(object):
  def __init__(self, meta, cache=None, info=None):
    self.meta = meta
    self.cache = cache

    if info:
      self.info = info
    else:
      self.info = self.fetch_info()

  @property
  def path(self):
    if 'skeletons' in self.meta.info:
      return self.meta.info['skeletons']
    return 'skeletons'

  @property
  def full_path(self):
    return self.meta.join(self.meta.cloudpath, self.path)

  def fetch_info(self):
    info = self.cache.download_json(self.meta.join(self.path, 'info'))

    if not info:
      return self.default_info()
    return info

  def refresh_info(self):
    self.info = self.fetch_info()
    return self.info

  def commit_info(self):
    if self.info:
      self.cache.upload_single(
        self.meta.join(self.path, 'info'),
        json.dumps(self.info), 
        content_type='application/json',
        compress=False,
        cache_control='no-cache',
      )

  def default_info(self):
    return {
      '@type': 'neuroglancer_skeletons',
      'transform': [  
        1, 0, 0, 0, # identity
        0, 1, 0, 0,
        0, 0, 1, 0
      ],
      'vertex_attributes': [
        {
          "id": "radius",
          "data_type": "float32",
          "num_components": 1,
        }, 
        {
          "id": "swc_type",
          "data_type": "uint8",
          "num_components": 1,
        }
      ],
      'sharding': None,
    }

  def is_sharded(self):
    if 'sharding' not in self.info:
      return False
    elif self.info['sharding'] is None:
      return False
    else:
      return True

class PrecomputedSkeletonSource(object):
  def __new__(cls, meta, cache, config):
    skel_meta = PrecomputedSkeletonMetadata(meta, cache)

    if skel_meta.is_sharded():
      raise NotImplementedError()

    return UnshardedPrecomputedSkeletonSource(skel_meta, cache, config)


