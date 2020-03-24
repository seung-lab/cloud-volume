from .image import PrecomputedImageSource
from .metadata import PrecomputedMetadata
from .mesh import PrecomputedMeshSource
from .skeleton import PrecomputedSkeletonSource

from ...cloudvolume import register_plugin, SharedConfiguration
from ...cacheservice import CacheService
from ...frontends import CloudVolumePrecomputed
from ...lib import yellow
from ...paths import strict_extract

def create_precomputed(
    cloudpath, mip=0, bounded=True, autocrop=False,
    fill_missing=False, cache=False, compress_cache=None,
    cdn_cache=True, progress=False, info=None, provenance=None,
    compress=None, compress_level=None, non_aligned_writes=False, parallel=1,
    delete_black_uploads=False, background_color=0, 
    green_threads=False, use_https=False,
    max_redirects=10
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
    )

    def cachecrt(cloudpath):
      return CacheService(
        cloudpath=cloudpath,
        enabled=bool(cache),
        config=config,
        compress=compress_cache,
      )

    cache = cachecrt(cache if type(cache) == str else cloudpath)

    meta = PrecomputedMetadata(
      cloudpath, cache=cache,
      info=info, provenance=provenance,
      max_redirects=max_redirects
    )

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
      config, meta, cache,
      autocrop=bool(autocrop),
      bounded=bool(bounded),
      non_aligned_writes=bool(non_aligned_writes),
      fill_missing=bool(fill_missing),
      delete_black_uploads=bool(delete_black_uploads),
      background_color=background_color,
      readonly=readonly,
    )

    mesh_cache = cache
    if 'mesh' in meta.info:
      mesh_cache = cachecrt(meta.join(cache.cloudpath, meta.info['mesh']))

    mesh = PrecomputedMeshSource(meta, mesh_cache, config, readonly)

    skel_cache = cache
    if 'skeletons' in meta.info:
      skel_cache = cachecrt(meta.join(cache.cloudpath, meta.info['skeletons']))

    skeleton = PrecomputedSkeletonSource(meta, skel_cache, config, readonly)

    return CloudVolumePrecomputed(
      meta, cache, config, 
      image, mesh, skeleton,
      mip
    )

def register():
  register_plugin('precomputed', create_precomputed)