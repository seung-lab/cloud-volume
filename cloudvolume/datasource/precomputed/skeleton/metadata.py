import copy
import re

from ....lib import jsonify

import numpy as np

SKEL_MIP_REGEXP = re.compile(r'skeletons_mip_(\d+)')

class PrecomputedSkeletonMetadata(object):
  def __init__(self, meta, cache=None, info=None):
    self.meta = meta
    self.cache = cache

    if info:
      self.info = info
    elif 'skeletons' in self.meta.info and self.meta.info['skeletons']:
      self.info = self.fetch_info()
    else:
      self.info = self.default_info()

  @property
  def spatial_index(self):
    if 'spatial_index' in self.info:
      return self.info['spatial_index']
    return None  

  @property
  def skeleton_path(self):
    if 'skeletons' in self.meta.info:
      return self.meta.info['skeletons']
    return 'skeletons'

  @property
  def mip(self):
    if 'mip' in self.info:
      return int(self.info['mip'])
    
    # Igneous has long used skeletons_mip_N to store
    # some information about the skeletonizing job. Let's 
    # exploit that for now.
    matches = re.search(SKEL_MIP_REGEXP, self.skeleton_path)
    if matches is None:
      return None 

    mip, = matches.groups()
    return int(mip)

  def join(self, *paths):
    return self.meta.join(*paths)

  @property
  def transform(self):
    return np.array(self.info['transform'], dtype=np.float32).reshape( (3,4) )

  @transform.setter
  def transform(self, val):
    self.info['transform'] = val

  @property
  def basepath(self):
    return self.meta.basepath

  @property
  def cloudpath(self):
    return self.meta.cloudpath

  @property
  def layerpath(self):
    return self.meta.join(self.meta.cloudpath, self.skeleton_path)

  def fetch_info(self):
    info = self.cache.download_json(self.meta.join(self.skeleton_path, 'info'))
    if not info:
      return self.default_info()
    return info

  def refresh_info(self):
    self.info = self.fetch_info()
    return self.info

  def commit_info(self):
    if self.info is None:
      return 

    info = copy.deepcopy(self.info)
    if info.get("sharding", None) is None:
      del info["sharding"]

    self.cache.upload_single(
      self.meta.join(self.skeleton_path, 'info'),
      jsonify(info), 
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
          "id": "vertex_types",
          "data_type": "uint8",
          "num_components": 1,
        }
      ],
      'sharding': None,
      'spatial_index': None, # { 'chunk_size': physical units }
    }

  def is_sharded(self):
    if 'sharding' not in self.info:
      return False
    elif self.info['sharding'] is None:
      return False
    else:
      return True
