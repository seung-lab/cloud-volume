from .mesh import GrapheneMeshSource
from .metadata import GrapheneMetadata
from ..precomputed.image import PrecomputedImageSource
from ..precomputed.skeleton import PrecomputedSkeletonSource

from ...cacheservice import CacheService
from ...cloudvolume import SharedConfiguration, register_plugin
from ...paths import strict_extract

from ...frontends import CloudVolumeGraphene
from requests import HTTPError

def create_graphene(
    cloudpath, mip=0, bounded=True, autocrop=False,
    fill_missing=False, cache=False, compress_cache=None,
    cdn_cache=True, progress=False, info=None, provenance=None,
    compress=None, parallel=1,
    delete_black_uploads=False, background_color=0,
    green_threads=False, use_https=False,
    **kwargs
  ):
    
    path = strict_extract(cloudpath)
    config = SharedConfiguration(
      cdn_cache=cdn_cache,
      compress=compress,
      green=green_threads,
      mip=mip,
      parallel=parallel,
      progress=progress,
    )
    cache = CacheService(
      cloudpath=(cache if type(cache) == str else cloudpath),
      enabled=bool(cache),
      config=config,
      compress=compress_cache,
    )

    meta = GrapheneMetadata(
      cloudpath, cache=cache,
      info=info, provenance=provenance, use_https=use_https
    )

    image = PrecomputedImageSource(
      config, meta, cache,
      autocrop=bool(autocrop),
      bounded=bool(bounded),
      non_aligned_writes=False,
      fill_missing=bool(fill_missing),
      delete_black_uploads=bool(delete_black_uploads),
      background_color=background_color,
    )

    mesh = GrapheneMeshSource(meta, cache, config)
    skeleton = PrecomputedSkeletonSource(meta, cache, config)

    return CloudVolumeGraphene(
      meta, cache, config, 
      image, mesh, 
      mip=mip
    )

def register():
  register_plugin('graphene', create_graphene)