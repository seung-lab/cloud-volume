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
    max_redirects=10, mesh_dir=None, skel_dir=None,
    secrets=None, **kwargs # absorb graphene arguments
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
    )

    cache = CacheService(
      cloudpath=(cache if type(cache) == str else cloudpath),
      enabled=bool(cache),
      config=config,
      compress=compress_cache,
    )

    meta = PrecomputedMetadata(
      cloudpath, config=config, cache=cache,
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
      config, meta, cache,
      autocrop=bool(autocrop),
      bounded=bool(bounded),
      non_aligned_writes=bool(non_aligned_writes),
      fill_missing=bool(fill_missing),
      delete_black_uploads=bool(delete_black_uploads),
      background_color=background_color,
      readonly=readonly,
    )

    mesh = PrecomputedMeshSource(meta, cache, config, readonly)
    skeleton = PrecomputedSkeletonSource(meta, cache, config, readonly)

    return CloudVolumePrecomputed(
      meta, cache, config, 
      image, mesh, skeleton,
      mip
    )

def register():
  register_plugin('precomputed', create_precomputed)