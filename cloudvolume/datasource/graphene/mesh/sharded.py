import six

from collections import defaultdict
import itertools
import json
import os
import posixpath
import re
import requests

import numpy as np
from tqdm import tqdm

from ....lib import red, toiter, Bbox, Vec
from ....mesh import Mesh
from .... import paths
from ....storage import Storage, GreenStorage

from ..sharding import GrapheneShardReader
from ...precomputed.sharding import ShardingSpecification

from .unsharded import GrapheneUnshardedMeshSource

class GrapheneShardedMeshSource(GrapheneUnshardedMeshSource):
  def __init__(self, mesh_meta, cache, config, readonly):
    super(GrapheneShardedMeshSource, self).__init__(mesh_meta, cache, config, readonly)

    self.readers = {}
    for level, sharding in self.meta.info['sharding'].items(): # { level: std sharding, ... }
      spec = ShardingSpecification.from_dict(sharding)
      self.readers[int(level)] = GrapheneShardReader(self.meta, self.cache, spec)

  def initial_path(self, level):
    return self.meta.join(self.meta.mesh_path, 'initial', str(level))

  def dynamic_path(self, level):
    return self.meta.join(self.meta.mesh_path, 'dynamic')

  # 1. determine if the segid is before or after the shard time point
  # 2. assuming it is sharded, fetch the draco encoded file from the
  #    correct level

  def dynamic_exists(self, labels, progress=None):
    """
    Checks for dynamic mesh existence.
  
    Returns: { label: path or None, ... }
    """
    labels = toiter(labels)

    checks = [ str(label) + ':0' for label in labels ]
    
    cloudpath = self.meta.join(self.meta.meta.cloudpath, self.meta.mesh_path, 'dynamic') 
    StorageClass = GreenStorage if self.config.green else Storage
    progress = progress if progress is not None else self.config.progress

    with StorageClass(cloudpath, progress=progress) as stor:
      results = stor.files_exist(checks)

    output = {}
    for filepath, exists in results.items():
      label = int(os.path.basename(filepath)[:-2]) # strip :0
      output[label] = filepath if exists else None

    return output

  def initial_exists(self, labels, return_byte_range=False, progress=None):
    """
    Checks for initial mesh existence.
  
    Returns: 
      If return_byte_range:
        { label: [ path, byte offset, byte size ] or None, ... }
      Else:
        { label: path or None, ... }
    """
    labels = toiter(labels)
    progress = progress if progress is not None else self.config.progress

    layers = defaultdict(list)
    for label in labels:
      if label == 0:
        continue
      layer = self.meta.meta.decode_layer_id(label)
      layers[layer].append(label)

    all_results = {}
    for layer, layer_labels in layers.items():
      path = self.initial_path(layer)
      results = self.readers[int(layer)].exists(
        layer_labels, 
        path=path, 
        return_byte_range=return_byte_range, 
        progress=progress
      )
      all_results.update(results)

    return all_results

  def exists(self, labels, progress=None):
    """
    Checks dynamic then initial meshes for existence.

    Returns: { label: path or None, ... }
    """
    labels = toiter(labels)
    labels = set(labels)

    dynamic_labels = self.dynamic_exists(labels, progress)
    remainder_labels = set([ label for label, path in dynamic_labels.items() if path ])

    initial_labels = self.initial_exists(remainder_labels, progress=progress, return_byte_range=False)

    dynamic_labels.update(initial_labels)
    return dynamic_labels

  def download_segid(self, seg_id, bounding_box):    
    """See GrapheneUnshardedMeshSource.get for the user facing function."""
    import DracoPy
    
    level = self.meta.meta.decode_layer_id(seg_id)
    if level not in self.readers:
      raise KeyError("There is no shard configuration in the mesh info file for level {}.".format(level))

    subdirectory = self.meta.join(self.meta.mesh_path, 'initial', str(level))
    raw_binary = self.readers[level].get_data(seg_id, path=subdirectory)

    if raw_binary is None:
      raise IndexError('No mesh found for segment {}'.format(seg_id))

    is_draco = False
    mesh = None

    try:
      # Easier to ask forgiveness than permission
      mesh = Mesh.from_draco(raw_binary)
      is_draco = True
    except DracoPy.FileTypeException:
      mesh = Mesh.from_precomputed(raw_binary)
    
    if mesh is None:
      raise IndexError('No mesh found for segment {}'.format(seg_id))

    mesh.segid = seg_id
    return mesh, is_draco

