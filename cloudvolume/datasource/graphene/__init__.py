from typing import Optional, Union

from .mesh import GrapheneMeshSource
from .metadata import GrapheneMetadata
from ..precomputed.image import PrecomputedImageSource
from ..precomputed.skeleton import PrecomputedSkeletonSource

from .. import get_cache_path
from ...cacheservice import CacheService
from ...cloudvolume import (
  SharedConfiguration, register_plugin,
  CompressType, ParallelType, CacheType,
  SecretsType
)
from ...paths import strict_extract

from requests import HTTPError

def create_graphene(
    cloudpath:str, mip:int=0, bounded:bool=True, autocrop:bool=False,
    fill_missing:bool=False, cache:CacheType=False, compress_cache:CompressType=None,
    cdn_cache:bool=True, progress:bool=False, info:dict=None, provenance:dict=None,
    compress:CompressType=None, parallel:ParallelType=1,
    delete_black_uploads:bool=False, background_color:int=0,
    green_threads:bool=False, use_https:bool=False,
    mesh_dir:Optional[str]=None, skel_dir:Optional[str]=None, 
    agglomerate:bool=False, secrets:SecretsType=None, 
    spatial_index_db:Optional[str]=None, 
    lru_bytes:int = 0,
    **kwargs
  ):
    from ...frontends import CloudVolumeGraphene
    
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
      spatial_index_db=spatial_index_db,
    )

    def mkcache(cloudpath):
      return CacheService(
        cloudpath=get_cache_path(cache, cloudpath),
        enabled=bool(cache),
        config=config,
        compress=compress_cache,
      )
    meta = GrapheneMetadata(
      cloudpath, config=config, cache=mkcache(cloudpath),
      info=info, provenance=provenance, 
      use_https=use_https, agglomerate=agglomerate,
      auth_token=config.secrets,
    )
    # Resetting the cache is necessary because
    # graphene retrieves a data_dir from the info file
    # that reflects the real cache location.
    cache_service = mkcache(meta.cloudpath) 
    meta.cache = cache_service
    cache_service.meta = meta

    if mesh_dir:
      meta.info['mesh'] = str(mesh_dir)
    if skel_dir:
      meta.info['skeletons'] = str(skel_dir)

    image = PrecomputedImageSource(
      config, meta, cache_service,
      autocrop=bool(autocrop),
      bounded=bool(bounded),
      non_aligned_writes=False,
      fill_missing=bool(fill_missing),
      delete_black_uploads=bool(delete_black_uploads),
      background_color=background_color,
      lru_bytes=lru_bytes,
    )

    mesh = GrapheneMeshSource(meta, cache_service, config)
    skeleton = PrecomputedSkeletonSource(meta, cache_service, config)

    return CloudVolumeGraphene(
      meta, cache_service, config, 
      image, mesh, skeleton,
      mip=mip
    )

def register():
  register_plugin('graphene', create_graphene)