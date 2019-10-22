import json
import os 

import numpy as np

from ...exceptions import SpatialIndexGapError
from ...storage import Storage, SimpleStorage
from ... import paths
from ...lib import Bbox, Vec, xyzrange, min2

class SpatialIndex(object):
  """
  Implements the client side reader of the 
  spatial index. During data generation, the
  labels in a given task are enumerated and 
  assigned their bounding box as JSON:

  {
    SEGID: [ x,y,z, x,y,z ],
    ...
  }

  The filename is the physical bounding box of the
  task dot spatial.

  e.g. "0-1024_0-1024_0-500.spatial" where the bbox 
  units are nanometers.

  The info file of the data type can then be augmented
  with:

  {
    "spatial_index": { "chunk_size": [ sx, sy, sz ] }
  }

  Where sx, sy, and sz are given in physical dimensions.
  """
  def __init__(self, cloudpath, bounds, chunk_size, progress=False):
    self.cloudpath = cloudpath
    self.path = paths.extract(cloudpath)
    self.bounds = Bbox.create(bounds)
    self.chunk_size = Vec(*chunk_size)
    self.progress = progress

  def join(self, *paths):
    if self.path.protocol == 'file':
      return os.path.join(*paths)
    else:
      return posixpath.join(*paths)    

  def fetch_index_files(self, index_files):
    with Storage(self.cloudpath, progress=self.progress) as stor:
      results = stor.get_files(index_files)

    for res in results:
      if res['error'] is not None:
        raise SpatialIndexGapError(res['error'])

    return { res['filename']: res['content'] for res in results }

  def query(self, bbox):
    """
    For the specified bounding box (or equivalent representation),
    list all segment ids enclosed within it.

    Returns: set(labels)
    """
    bbox = Bbox.create(bbox, context=self.bounds, autocrop=True)
    original_bbox = bbox.clone()
    bbox = bbox.expand_to_chunk_size(self.chunk_size, offset=self.bounds.minpt)

    if bbox.subvoxel():
      return []

    index_files = []
    for pt in xyzrange(bbox.minpt, bbox.maxpt, self.chunk_size):
      search = Bbox( pt, min2(pt + self.chunk_size, self.bounds.maxpt) )
      index_files.append(search.to_filename() + '.spatial')

    results = self.fetch_index_files(index_files)

    labels = set()
    for filename, content in results.items():
      if content is None:
        raise SpatialIndexGapError(filename + " was not found.")

      res = json.loads(content)

      for label, label_bbx in res.items():
        label = int(label)
        label_bbx = Bbox.from_list(label_bbx)

        if Bbox.intersects(label_bbx, original_bbox):
          labels.add(label)

    return labels

class CachedSpatialIndex(SpatialIndex):
  def __init__(self, cache, cloudpath, bounds, chunk_size, progress=None):
    self.cache = cache
    self.subdir = os.path.relpath(cloudpath, cache.meta.cloudpath)

    super(CachedSpatialIndex, self).__init__(
      cloudpath, bounds, chunk_size, progress
    )

  def fetch_index_files(self, index_files):
    index_files = [ self.cache.meta.join(self.subdir, fname) for fname in index_files ]
    return self.cache.download(index_files, progress=self.progress)
