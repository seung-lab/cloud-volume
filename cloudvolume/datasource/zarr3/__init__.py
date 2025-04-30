from typing import Optional

from ...frontends.precomputed import CloudVolumePrecomputed

from ...cloudvolume import (
  register_plugin, SharedConfiguration,
  CompressType, ParallelType, CacheType,
  SecretsType
)

def create_zarr3(
  cloudpath:str, mip:int=0, bounded:bool=True, autocrop:bool=False,
  fill_missing:bool=False, cache:CacheType=False, compress_cache:CompressType=None,
  cdn_cache:bool=True, progress:bool=False, info:Optional[dict]=None,
  compress:CompressType=None, compress_level:Optional[int]=None,
  non_aligned_writes:bool=False, delete_black_uploads:bool=False,
  parallel:ParallelType=1,green_threads:bool=False, 
  secrets:SecretsType=None, cache_locking:bool = True,
  **kwargs # absorb graphene arguments
):
  raise NotImplementedError("zarr3 is not yet supported.")

def register():
  register_plugin('zarr3', create_zarr3)