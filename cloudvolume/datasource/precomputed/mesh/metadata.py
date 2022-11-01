import re

from ....lib import jsonify

import numpy as np

MESH_MIP_REGEXP = re.compile(r'mesh_mip_(\d+)')

class PrecomputedMeshMetadata(object):
  def __init__(self, meta, cache=None, info=None):
    self.meta = meta
    self.cache = cache

    if info:
      self.info = info
    else:
      self.info = self.fetch_info()

  @property
  def chunk_size(self):
    if 'chunk_size' in self.info:
      return self.info['chunk_size']
    return None

  @property
  def mip(self):
    if 'mip' in self.info:
      return int(self.info['mip'])
    
    # Igneous has long used mesh_mip_N_err_M to store
    # some information about the meshing job. Let's 
    # exploit that for now.
    matches = re.search(MESH_MIP_REGEXP, self.mesh_path)
    if matches is None:
      return None 

    mip, = matches.groups()
    return int(mip)

  @mip.setter
  def mip(self, val):
    self.info["mip"] = int(val)
  
  @property
  def spatial_index(self):
    if 'spatial_index' in self.info:
      return self.info['spatial_index']
    return None

  @property
  def mesh_path(self):
    if 'mesh' in self.meta.info:
      return self.meta.info['mesh']
    return 'mesh'

  def join(self, *paths):
    return self.meta.join(*paths)

  @property
  def basepath(self):
    return self.meta.basepath

  @property
  def cloudpath(self):
    return self.meta.cloudpath

  @property
  def layerpath(self):
    return self.meta.join(self.meta.cloudpath, self.mesh_path)

  def fetch_info(self):
    if 'mesh' not in self.meta.info or not self.meta.info['mesh']:
      return self.default_info()

    info = self.cache.download_json(self.meta.join(self.mesh_path, 'info'))
    if not info:
      return self.default_info()
    return info

  def refresh_info(self):
    self.info = self.fetch_info()
    return self.info

  def commit_info(self):
    if self.info:
      self.cache.upload_single(
        self.meta.join(self.mesh_path, 'info'),
        jsonify(self.info), 
        content_type='application/json',
        compress=False,
        cache_control='no-cache',
      )

  def default_info(self):
    return {
      '@type': 'neuroglancer_legacy_mesh',
      'spatial_index': None, # { 'chunk_size': physical units }
    }

  def is_sharded(self):
    if 'sharding' not in self.info:
      return False
    elif self.info['sharding'] is None:
      return False
    else:
      return True

  def is_multires(self):
    return self.info['@type'] == 'neuroglancer_multilod_draco'

  def is_legacy(self):
    return not self.is_multires()