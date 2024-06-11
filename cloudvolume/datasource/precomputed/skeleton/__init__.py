from typing import Optional

import os

from .metadata import PrecomputedSkeletonMetadata
from .sharded import ShardedPrecomputedSkeletonSource
from .unsharded import UnshardedPrecomputedSkeletonSource

from ..metadata import PrecomputedMetadata
from ....cacheservice import CacheService
from ....paths import strict_extract
from ....cloudvolume import SharedConfiguration

class PrecomputedSkeletonSource(object):
  def __new__(cls, meta, cache, config, readonly=False, info=None):
    skel_meta = PrecomputedSkeletonMetadata(meta, cache, config, readonly=readonly, info=info)

    if skel_meta.is_sharded():
      return ShardedPrecomputedSkeletonSource(skel_meta, cache, config, readonly) 

    return UnshardedPrecomputedSkeletonSource(skel_meta, cache, config, readonly)

  @classmethod
  def from_cloudpath(
    cls, 
    cloudpath:str, 
    cache=False, 
    progress=False,
    secrets=None,
    spatial_index_db:Optional[str]=None, 
    cache_locking:bool = True,
  ):
    config = SharedConfiguration(
      cdn_cache=False,
      compress=True,
      compress_level=None,
      green=False,
      mip=0,
      parallel=1,
      progress=progress,
      secrets=secrets,
      spatial_index_db=spatial_index_db,
      cache_locking=cache_locking,
    )
    
    cache = CacheService(
      cloudpath=(cache if type(cache) == str else cloudpath),
      enabled=bool(cache),
      config=config,
      compress=True,
    )

    cloudpath, skel_dir = os.path.split(cloudpath)
    meta = PrecomputedMetadata(
      cloudpath, config, cache, 
      info={ 'skeletons': skel_dir }
    )

    return PrecomputedSkeletonSource(meta, cache, config)