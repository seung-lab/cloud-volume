from __future__ import annotations

from typing import Any, Optional

import copy
import re
import weakref

from ....lib import jsonify
from ..sharding import ShardingSpecification, compute_shard_params_for_hashed

import numpy as np
import numpy.typing as npt

SKEL_MIP_REGEXP = re.compile(r'skeletons_mip_(\d+)')

class PrecomputedSkeletonMetadata(object):
  def __init__(
    self, meta: Any, cache: Any = None, config: Any = None,
    info: Optional[dict[str, Any]] = None, readonly: bool = False
  ) -> None:
    self.meta = meta
    self.cache = cache
    self.config = config
    self.readonly = readonly
    self._cv: Any = None

    if info:
      self.info = info
    elif 'skeletons' in self.meta.info and self.meta.info['skeletons']:
      self.info = self.fetch_info()
    else:
      self.info = self.default_info()

  @property
  def cv(self) -> Any:
    return self._cv

  @cv.setter
  def cv(self, vol: Any) -> None:
    self._cv = weakref.ref(vol)

  @cv.deleter
  def cv(self) -> None:
    del self._cv

  @property
  def spatial_index(self) -> Optional[dict[str, Any]]:
    if 'spatial_index' in self.info:
      return self.info['spatial_index']
    return None

  @property
  def skeleton_path(self) -> str:
    if 'skeletons' in self.meta.info:
      return self.meta.info['skeletons']
    return 'skeletons'

  @property
  def mip(self) -> Optional[int]:
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

  def join(self, *paths: str) -> str:
    return self.meta.join(*paths)

  @property
  def transform(self) -> npt.NDArray[np.float32]:
    return np.array(self.info['transform'], dtype=np.float32).reshape( (3,4) )

  @transform.setter
  def transform(self, val: Any) -> None:
    self.info['transform'] = val

  @property
  def basepath(self) -> str:
    return self.meta.basepath

  @property
  def cloudpath(self) -> str:
    return self.meta.cloudpath

  @property
  def layerpath(self) -> str:
    return self.meta.join(self.meta.cloudpath, self.skeleton_path)

  def fetch_info(self) -> dict[str, Any]:
    info = self.cache.download_json(self.meta.join(self.skeleton_path, 'info'))
    if not info:
      return self.default_info()
    return info

  def refresh_info(self) -> dict[str, Any]:
    self.info = self.fetch_info()
    return self.info

  def commit_info(self) -> None:
    if self.info is None:
      return

    info = copy.deepcopy(self.info)
    if info.get("sharding", None) is None:
        info.pop("sharding", None)

    self.cache.upload_single(
      self.meta.join(self.skeleton_path, 'info'),
      jsonify(info),
      content_type='application/json',
      compress=False,
      cache_control='no-cache',
    )

  def default_info(self) -> dict[str, Any]:
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

  def compute_sharding_specification(
    self,
    num_labels: int,
    shard_index_bytes: int = 2**13,
    minishard_index_bytes: int = 2**15,
    min_shards: int = 1,
    minishard_index_encoding: str = 'gzip',
    data_encoding: str = 'gzip',
    max_labels_per_shard: Optional[int] = None,
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
    num_labels: int,
    shard_index_bytes: int = 2**13,
    minishard_index_bytes: int = 2**15,
    min_shards: int = 1,
    minishard_index_encoding: str = 'gzip',
    data_encoding: str = 'gzip',
    max_labels_per_shard: Optional[int] = None,
  ) -> None:
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

    self._refresh_skeleton_interface()

  def to_unsharded(self) -> None:
    self.info.pop("sharding", None)
    self._refresh_skeleton_interface()

  def _refresh_skeleton_interface(self) -> None:
    from cloudvolume.datasource.precomputed.skeleton import PrecomputedSkeletonSource
    if self.cv:
      skeleton_src = PrecomputedSkeletonSource(self.meta, self.cache, self.config, self.readonly, info=self.info)
      skeleton_src.meta.cv = self.cv()
      self.cv().skeleton = skeleton_src

  def is_sharded(self) -> bool:
    if 'sharding' not in self.info:
      return False
    elif self.info['sharding'] is None:
      return False
    else:
      return True
