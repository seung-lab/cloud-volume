from typing import Optional, Union

from .image import BossImageSource
from .metadata import BossMetadata

from ...frontends.precomputed import CloudVolumePrecomputed

from .. import get_cache_path
from ...cacheservice import CacheService
from ...cloudvolume import (
  register_plugin, SharedConfiguration,
  CompressType, ParallelType, CacheType,
  SecretsType
)
from ...paths import strict_extract

def create_boss(
    cloudpath, mip:int = 0, bounded:bool = True, autocrop:bool = False,
    fill_missing:bool = False, cache:CacheType = False, compress_cache:CompressType = None,
    cdn_cache:bool = True, progress:bool = False, info:Optional[dict] = None, 
    provenance:Optional[dict] = None, compress:CompressType = None, 
    non_aligned_writes:bool = False, parallel:int = 1, delete_black_uploads:bool = False, 
    green_threads:bool = False, cache_locking:bool = True,
  ):
    path = strict_extract(cloudpath)
    config = SharedConfiguration(
      cdn_cache=cdn_cache,
      compress=compress,
      compress_level=None,
      green=green_threads,
      mip=mip,
      parallel=parallel,
      progress=progress,
      secrets=secrets,
      spatial_index_db=None,
      cache_locking=cache_locking,
    )
    cache = CacheService(
      cloudpath=get_cache_path(cache, cloudpath),
      enabled=bool(cache),
      config=config,
      compress=compress_cache,
    )

    meta = BossMetadata(cloudpath, cache=cache, info=info)
    image = BossImageSource(
      config, meta, cache, 
      autocrop=bool(autocrop),
      bounded=bool(bounded),
      non_aligned_writes=bool(non_aligned_writes), 
    )

    return CloudVolumePrecomputed(
      meta, cache, config, 
      imagesrc, mesh=None, skeleton=None,
      mip=mip
    )

def register():
  register_plugin('boss', create_boss)