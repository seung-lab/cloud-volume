import json

import numpy as np

class PrecomputedSkeletonMetadata(object):
  def __init__(self, meta, cache=None, info=None):
    self.meta = meta
    self.cache = cache

    if info:
      self.info = info
    else:
      self.info = self.fetch_info()

  @property
  def skeleton_path(self):
    if 'skeletons' in self.meta.info:
      return self.meta.info['skeletons']
    return 'skeletons'

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
          "id": "vertex_types",
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
