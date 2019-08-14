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

from ..common import cdn_cache_control

from ....skeleton import Skeleton

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
        raise SkeletonDecodeError("segid " + str(segid) + ": " + str(err))
      skel.transform = self.meta.transform
      skeletons.append(skel.physical_space())

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