from .image import PrecomputedImageSource
from .metadata import PrecomputedMetadata
from .mesh import PrecomputedMeshSource
from .skeleton import PrecomputedSkeletonSource

from ...cloudvolume import register_plugin, SharedConfiguration
from ...cacheservice import CacheService
from ...frontends import CloudVolumePrecomputed
from ...paths import strict_extract

def create_precomputed(
    cloudpath, mip=0, bounded=True, autocrop=False,
    fill_missing=False, cache=False, compress_cache=None,
    cdn_cache=True, progress=False, info=None, provenance=None,
    compress=None, non_aligned_writes=False, parallel=1,
    delete_black_uploads=False, green_threads=False,
    use_https=False
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

    meta = PrecomputedMetadata(
      cloudpath, cache=cache,
      info=info, provenance=provenance,
    )

    image = PrecomputedImageSource(
      config, meta, cache,
      autocrop=bool(autocrop),
      bounded=bool(bounded),
      non_aligned_writes=bool(non_aligned_writes),
      fill_missing=bool(fill_missing),
      delete_black_uploads=bool(delete_black_uploads),
    )

    mesh = PrecomputedMeshSource(meta, cache, config)
    skeleton = PrecomputedSkeletonSource(meta, cache, config)

    return CloudVolumePrecomputed(
      meta, cache, config, 
      image, mesh, skeleton,
      mip
    )

def register():
  register_plugin('precomputed', create_precomputed)