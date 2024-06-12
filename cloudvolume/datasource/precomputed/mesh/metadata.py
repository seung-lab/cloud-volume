from typing import Optional

import re
import weakref

from ....lib import jsonify
from ..sharding import ShardingSpecification, compute_shard_params_for_hashed

import numpy as np

MESH_MIP_REGEXP = re.compile(r'mesh_mip_(\d+)')

class PrecomputedMeshMetadata(object):
  def __init__(self, meta, cache=None, config=None, info=None, readonly=False):
    self.meta = meta
    self.cache = cache
    self.config = config
    self.readonly = readonly
    self._cv = None

    if info:
      self.info = info
    else:
      self.info = self.fetch_info()

  @property
  def cv(self):
    return self._cv

  @cv.setter
  def cv(self, vol):
    self._cv = weakref.ref(vol)

  @cv.deleter
  def cv(self):
    del self._cv

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

  def compute_sharding_specification(
    self, 
    num_labels:int,
    shard_index_bytes:int = 2**13,
    minishard_index_bytes:int = 2**15,
    min_shards:int = 1,
    minishard_index_encoding:str = 'gzip', 
    data_encoding:str = 'gzip',
    max_labels_per_shard:Optional[int] = None,
  ) -> ShardingSpecification:
    """
    Calculate the shard parameters for this volume given
    the total number of labels in the volume.
    """
    if max_labels_per_shard is not None:
      assert max_labels_per_shard >= 1
      min_shards = max(int(np.ceil(len(all_labels) / max_labels_per_shard)), min_shards)

    (shard_bits, minishard_bits, preshift_bits) = \
      compute_shard_params_for_hashed(
        num_labels=num_labels,
        shard_index_bytes=int(shard_index_bytes),
        minishard_index_bytes=int(minishard_index_bytes),
        min_shards=int(min_shards),
      )

    return ShardingSpecification(
      type='neuroglancer_uint64_sharded_v1',
      preshift_bits=preshift_bits,
      hash='murmurhash3_x86_128',
      minishard_bits=minishard_bits,
      shard_bits=shard_bits,
      minishard_index_encoding=minishard_index_encoding,
      data_encoding=data_encoding,
    )

  def to_sharded(
    self, 
    num_labels:int,
    shard_index_bytes:int = 2**13,
    minishard_index_bytes:int = 2**15,
    min_shards:int = 1,
    minishard_index_encoding:str = 'gzip', 
    data_encoding:str = 'gzip',
    max_labels_per_shard:Optional[int] = None,
  ):
    """Adds a computed sharding property to the info."""
    spec = self.compute_sharding_specification(
      num_labels=num_labels, 
      shard_index_bytes=shard_index_bytes, 
      minishard_index_bytes=minishard_index_bytes, 
      min_shards=min_shards,
      minishard_index_encoding=minishard_index_encoding,
      data_encoding=data_encoding,
      max_labels_per_shard=max_labels_per_shard,
    )
    self.info['sharding'] = spec.to_dict()
    self._refresh_mesh_interface()

  def to_unsharded(self):
    self.info.pop("sharding", None)
    self._refresh_mesh_interface()

  def _refresh_mesh_interface(self):
    from cloudvolume.datasource.precomputed.mesh import PrecomputedMeshSource
    if self.cv:
      mesh_src = PrecomputedMeshSource(self.meta, self.cache, self.config, self.readonly, info=self.info)
      mesh_src.meta.cv = self.cv()
      self.cv().mesh = mesh_src

  def to_multi_resolution(self, vertex_quantization_bits:int):
    if vertex_quantization_bits not in [10, 16]:
      raise ValueError(f"vertex_quantization_bits must by 10 or 16. Got: {vertex_quantization_bits}")

    res = self.meta.resolution(self.mip)

    self.info['@type'] = "neuroglancer_multilod_draco"
    self.info['vertex_quantization_bits'] = vertex_quantization_bits
    self.info['transform'] = [ 
      res[0], 0,      0,      0,
      0,      res[1], 0,      0,
      0,      0,      res[2], 0,
    ]
    self.info['lod_scale_multiplier'] = 1.0
    self._refresh_mesh_interface()

  def to_single_resolution(self):
    self.info['@type'] = "neuroglancer_legacy_mesh"
    self.info.pop("vertex_quantization_bits", None)
    self.info.pop("transform", None)
    self.info.pop("lod_scale_multiplier", None)
    self._refresh_mesh_interface()

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