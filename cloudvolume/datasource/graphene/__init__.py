from .mesh import GrapheneMeshSource
from .metadata import GrapheneMetadata
from ..precomputed.image import PrecomputedImageSource
from ..precomputed.skeleton import PrecomputedSkeletonSource

from .. import get_cache_path
from ...cacheservice import CacheService
from ...cloudvolume import SharedConfiguration, register_plugin
from ...paths import strict_extract

from requests import HTTPError

def create_graphene(
    cloudpath, mip=0, bounded=True, autocrop=False,
    fill_missing=False, cache=False, compress_cache=None,
    cdn_cache=True, progress=False, info=None, provenance=None,
    compress=None, parallel=1,
    delete_black_uploads=False, background_color=0,
    green_threads=False, use_https=False,
    mesh_dir=None, skel_dir=None, agglomerate=False, 
    secrets=None, **kwargs
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
    cache = mkcache(meta.cloudpath) 
    meta.cache = cache
    cache.meta = meta

    if mesh_dir:
      meta.info['mesh'] = str(mesh_dir)
    if skel_dir:
      meta.info['skeletons'] = str(skel_dir)

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
      image, mesh, skeleton,
      mip=mip
    )

def register():
  register_plugin('graphene', create_graphene)