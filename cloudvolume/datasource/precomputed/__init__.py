from typing import Optional, Union

from .image import PrecomputedImageSource
from .metadata import PrecomputedMetadata
from .mesh import PrecomputedMeshSource
from .skeleton import PrecomputedSkeletonSource

from .. import get_cache_path
from ...cloudvolume import (
  register_plugin, SharedConfiguration,
  CompressType, ParallelType, CacheType,
  SecretsType
)
from ...cacheservice import CacheService
from ...frontends import CloudVolumePrecomputed
from ...lib import yellow
from ...paths import strict_extract
from ...secrets import CLOUD_VOLUME_CACHE_DIR

def create_precomputed(
    cloudpath:str, mip:int=0, bounded:bool=True, autocrop:bool=False,
    fill_missing:bool=False, cache:CacheType=False, compress_cache:CompressType=None,
    cdn_cache:bool=True, progress:bool=False, info:Optional[dict]=None, 
    provenance:Optional[dict]=None, compress:CompressType=None, 
    compress_level:Optional[int]=None, non_aligned_writes:bool=False, 
    parallel:ParallelType=1, delete_black_uploads:bool=False, background_color:int=0, 
    green_threads:bool=False, use_https:bool=False,
    max_redirects:int=10, mesh_dir:Optional[str]=None, skel_dir:Optional[str]=None,
    secrets:SecretsType=None, spatial_index_db:Optional[str]=None, 
    lru_bytes:int = 0, cache_locking:bool = True,
    **kwargs # absorb graphene arguments
  ):
    path = strict_extract(cloudpath)
    config = SharedConfiguration(
      cdn_cache=cdn_cache,
      compress=compress,
      compress_level=compress_level,
      green=green_threads,
      mip=mip,
      parallel=parallel,
      progress=progress,
      secrets=secrets,
      spatial_index_db=spatial_index_db,
      cache_locking=cache_locking,
    )

    cache_service = CacheService(
      cloudpath=get_cache_path(cache, cloudpath),
      enabled=bool(cache),
      config=config,
      compress=compress_cache,
    )

    meta = PrecomputedMetadata(
      cloudpath, config=config, cache=cache_service,
      info=info, provenance=provenance,
      max_redirects=max_redirects,
      use_https=use_https # for parsing redirects
    )

    if mesh_dir:
      meta.info['mesh'] = str(mesh_dir)
    if skel_dir:
      meta.info['skeletons'] = str(skel_dir)

    readonly = bool(meta.redirected_from)

    if readonly:
      print(yellow("""
        Redirected 

        From: {} 
        To:   {}
        Hops: {}

        Volume set to readonly to mitigate accidental
        redirection resulting in writing data to the wrong 
        location.

        To set the data to writable, access the destination
        location directly or set:

        vol.image.readonly = False
        vol.mesh.readonly = False 
        vol.skeleton.readonly = False
      """.format(
        cloudpath, meta.cloudpath, len(meta.redirected_from)
      )))

    image = PrecomputedImageSource(
      config, meta, cache_service,
      autocrop=bool(autocrop),
      bounded=bool(bounded),
      non_aligned_writes=bool(non_aligned_writes),
      fill_missing=bool(fill_missing),
      delete_black_uploads=bool(delete_black_uploads),
      background_color=background_color,
      readonly=readonly,
      lru_bytes=lru_bytes,
    )

    mesh = PrecomputedMeshSource(meta, cache_service, config, readonly)
    skeleton = PrecomputedSkeletonSource(meta, cache_service, config, readonly)

    cv = CloudVolumePrecomputed(
      meta, cache_service, config, 
      image, mesh, skeleton,
      mip
    )

    skeleton.meta.cv = cv # assigned as a weakref
    mesh.meta.cv = cv # assigned as a weakref

    return cv


def register():
  register_plugin('precomputed', create_precomputed)