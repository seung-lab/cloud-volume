from .image import BossImageSource
from .metadata import BossMetadata

from ...frontends.precomputed import CloudVolumePrecomputed

from .. import get_cache_path
from ...cacheservice import CacheService
from ...cloudvolume import SharedConfiguration, register_plugin
from ...paths import strict_extract

def create_boss(
    cloudpath, mip=0, bounded=True, autocrop=False,
    fill_missing=False, cache=False, compress_cache=None,
    cdn_cache=True, progress=False, info=None, provenance=None,
    compress=None, non_aligned_writes=False, parallel=1,
    delete_black_uploads=False, green_threads=False
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
    )
    cache = CacheService(
      cloudpath=get_cache_path(cache, cloudpath),
      enabled=bool(cache),
      config=config,
      compress=compress_cache,
    )

    meta = BossMetadata(cloudpath, cache=cache, info=info)
    image = BossImageSource(
      config, meta, cache, 
      autocrop=bool(autocrop),
      bounded=bool(bounded),
      non_aligned_writes=bool(non_aligned_writes), 
    )

    return CloudVolumePrecomputed(
      meta, cache, config, 
      imagesrc, mesh=None, skeleton=None,
      mip=mip
    )

def register():
  register_plugin('boss', create_boss)