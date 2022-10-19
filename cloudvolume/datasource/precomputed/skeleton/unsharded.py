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

from ..common import cdn_cache_control
from ..spatial_index import CachedSpatialIndex
from ... import readonlyguard

from ....skeleton import Skeleton

class UnshardedPrecomputedSkeletonSource(object):
  def __init__(self, meta, cache, config, readonly=False):
    self.meta = meta
    self.cache = cache
    self.config = config

    self.readonly = bool(readonly)

    self.spatial_index = None
    if self.meta.spatial_index:
      mip = self.meta.mip or 0
      self.spatial_index = CachedSpatialIndex(
        self.cache, self.config,
        cloudpath=self.meta.layerpath, 
        bounds=self.meta.meta.bounds(mip),
        resolution=self.meta.info['spatial_index'].get('resolution', self.meta.meta.resolution(mip)),
        chunk_size=self.meta.info['spatial_index']['chunk_size'],
      )

  @property
  def path(self):
    return self.meta.skeleton_path 

  def get(self, segids, allow_missing=False):
    """
    Retrieve one or more skeletons from the data layer.

    Example: 
      skel = vol.skeleton.get(5)
      skels = vol.skeleton.get([1, 2, 3])

    Raises SkeletonDecodeError on missing files or decoding errors.

    Required:
      segids: list of integers or integer
    Optional:
      allow_missing: skip over non-existent files otherwise
        raise cloudvolume.exceptions.SkeletonDecodeError

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
      [ self.meta.join(self.meta.skeleton_path, str(segid)) for segid in segids ],
      compress=compress
    )
    missing = [ filename for filename, content in results.items() if content is None ]
    results = { filename: content for filename, content in results.items() if content is not None }

    if not allow_missing and len(missing):
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

  @readonlyguard
  def upload_raw(self, segid, vertices, edges, radii=None, vertex_types=None):
    skel = Skeleton(
      vertices, edges, radii, 
      vertex_types, segid=segid
    )
    return self.upload(skel)
    
  @readonlyguard
  def upload(self, skeletons):

    compress = self.config.compress 
    if compress is None:
      compress = True

    if type(skeletons) == Skeleton:
      skeletons = [ skeletons ]

    files = [ (self.meta.join(self.meta.skeleton_path, str(skel.id)), skel.to_precomputed()) for skel in skeletons ]
    self.cache.upload(
      files=files, 
      compress=compress, 
      cache_control=cdn_cache_control(self.config.cdn_cache)
    )

  def get_bbox(self, bbox):
    if self.spatial_index is None:
      raise IndexError("A spatial index has not been created.")

    segids = self.spatial_index.query(bbox)
    return self.get(segids)
