from typing import Optional

from .image import Zarr2ImageSource
from .metadata import Zarr2Metadata

from ...frontends.precomputed import CloudVolumePrecomputed

from .. import get_cache_path
from ...cacheservice import CacheService
from ...cloudvolume import (
  register_plugin, SharedConfiguration,
  CompressType, ParallelType, CacheType,
  SecretsType
)
from ...paths import strict_extract

def create_zarr2(
  cloudpath:str, mip:int=0, bounded:bool=True, autocrop:bool=False,
  fill_missing:bool=False, cache:CacheType=False, compress_cache:CompressType=None,
  cdn_cache:bool=True, progress:bool=False, info:Optional[dict]=None,
  compress:CompressType=None, compress_level:Optional[int]=None,
  non_aligned_writes:bool=False, delete_black_uploads:bool=False,
  parallel:ParallelType=1,green_threads:bool=False, 
  secrets:SecretsType=None, cache_locking:bool = True,
  codec_threads:ParallelType = 1,
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
      cache_locking=cache_locking,
      codec_threads=codec_threads,
    )
    cache = CacheService(
      cloudpath=get_cache_path(cache, cloudpath),
      enabled=bool(cache),
      config=config,
      compress=compress_cache,
    )

    meta = Zarr2Metadata(cloudpath, config=config, cache=cache, info=info)
    imagesrc = Zarr2ImageSource(
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
  register_plugin('zarr2', create_zarr2)