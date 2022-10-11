from typing import Optional

from .image import N5ImageSource
from .metadata import N5Metadata

from ...frontends.precomputed import CloudVolumePrecomputed

from .. import get_cache_path
from ...cacheservice import CacheService
from ...cloudvolume import (
  register_plugin, SharedConfiguration,
  CompressType, ParallelType, CacheType,
  SecretsType
)
from ...paths import strict_extract

def create_n5(
  cloudpath:str, mip:int=0, bounded:bool=True, autocrop:bool=False,
  fill_missing:bool=False, cache:CacheType=False, compress_cache:CompressType=None,
  cdn_cache:bool=True, progress:bool=False, 
  compress:CompressType=None, compress_level:Optional[int]=None,
  non_aligned_writes:bool=False, 
  parallel:ParallelType=1,green_threads:bool=False, secrets:SecretsType=None, 
  **kwargs # absorb graphene arguments
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
    )
    cache = CacheService(
      cloudpath=get_cache_path(cache, cloudpath),
      enabled=bool(cache),
      config=config,
      compress=compress_cache,
    )

    meta = N5Metadata(cloudpath, config=config, cache=cache)
    imagesrc = N5ImageSource(
      config, meta, cache, 
      autocrop=bool(autocrop),
      bounded=bool(bounded),
      non_aligned_writes=bool(non_aligned_writes),
      fill_missing=bool(fill_missing),
    )

    return CloudVolumePrecomputed(
      meta, cache, config, 
      imagesrc, mesh=None, skeleton=None,
      mip=mip
    )

def register():
  register_plugin('n5', create_n5)