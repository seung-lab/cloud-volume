from collections import defaultdict
import json
import os 

import numpy as np

from ...exceptions import SpatialIndexGapError
from ...storage import Storage, SimpleStorage
from ... import paths
from ...lib import Bbox, Vec, xyzrange, min2, toiter

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

  def index_file_paths_for_bbox(self, bbox):
    bbox = bbox.expand_to_chunk_size(self.chunk_size, offset=self.bounds.minpt)

    if bbox.subvoxel():
      return []

    index_files = []
    for pt in xyzrange(bbox.minpt, bbox.maxpt, self.chunk_size):
      search = Bbox( pt, min2(pt + self.chunk_size, self.bounds.maxpt) )
      index_files.append(search.to_filename() + '.spatial')
    
    return index_files

  def file_locations_per_label(self, labels=None, allow_missing=False):
    """
    Queries entire dataset to find which spatial index files the 
    given labels are located in. Can be expensive. If labels is not 
    specified, all labels are fetched.

    Returns: { filename: [ labels... ], ... }
    """
    if labels is not None:
      labels = set(toiter(labels))
      
    index_files = self.index_file_paths_for_bbox(self.bounds)
    index_files = self.fetch_index_files(index_files)
    locations = defaultdict(list)
    for filename, content in index_files.items():
      if content is None:
        if allow_missing:
          continue
        else:
          raise SpatialIndexGapError(filename + " was not found.")

      segid_bbox_dict = json.loads(content)
      filename = os.path.basename(filename)

      if labels is None:
        for label in segid_bbox_dict.keys():
          locations[int(label)].append(filename)
      elif len(labels) > len(segid_bbox_dict):
        for label in segid_bbox_dict.keys():
          if int(label) in labels:
            locations[int(label)].append(filename)
      else:
        for label in labels:
          if str(label) in segid_bbox_dict:
            locations[int(label)].append(filename)

    return locations

  def query(self, bbox, allow_missing=False):
    """
    For the specified bounding box (or equivalent representation),
    list all segment ids enclosed within it.

    If allow_missing is set, then don't raise an error if an index
    file is missing.

    Returns: set(labels)
    """
    bbox = Bbox.create(bbox, context=self.bounds, autocrop=True)
    original_bbox = bbox.clone()
    bbox = bbox.expand_to_chunk_size(self.chunk_size, offset=self.bounds.minpt)

    if bbox.subvoxel():
      return []

    index_files = self.index_file_paths_for_bbox(bbox)
    results = self.fetch_index_files(index_files)

    labels = set()
    for filename, content in results.items():
      if content is None:
        if allow_missing:
          continue
        else:
          raise SpatialIndexGapError(filename + " was not found.")

      res = json.loads(content)

      # The bbox test saps performance a lot
      # but we can skip it if we know 100% that
      # the labels are going to be inside. This
      # optimization is important for querying 
      # entire datasets, which is contemplated
      # for shard generation.
      if bbox.contains_bbox(self.bounds):
        for label, label_bbx in res.items():
          labels.add(int(label))
      else:
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
