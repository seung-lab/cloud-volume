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
  def __init__(self, *args, **kwargs):
    super(GrapheneShardedMeshSource, self).__init__(self, *args, **kwargs)

    self.readers = {}
    for level, sharding in self.meta.info['sharding'].items(): # { level: std sharding, ... }
      spec = ShardingSpecification.from_dict(sharding)
      self.readers[int(level)] = GrapheneShardReader(self.meta, self.cache, spec)

  # 1. determine if the segid is before or after the shard time point
  # 2. assuming it is sharded, fetch the draco encoded file from the
  #    correct level

  def download_segid(self, seg_id, bounding_box):
    """See GrapheneUnshardedMeshSource.get for the user facing function."""
    level = self.meta.meta.decode_layer_id(seg_id)
    if level not in self.readers:
      raise KeyError("There is no shard configuration in the mesh info file for level {}.".format(level))

    subdirectory = self.meta.join(self.meta.mesh_path, 'initial', str(level))
    raw_binary = self.reader[level].get_data(segid, path=subdirectory)

    if raw_binary is None:
      raise IndexError('No mesh found for segment {}'.format(seg_id))

    is_draco = False
    mesh = None

    try:
      # Easier to ask forgiveness than permission
      mesh = Mesh.from_draco(frag)
      is_draco = True
    except DracoPy.FileTypeException:
      mesh = Mesh.from_precomputed(frag)
    
    if mesh is None:
      raise IndexError('No mesh found for segment {}'.format(seg_id))

    mesh.segid = seg_id
    return mesh, is_draco

