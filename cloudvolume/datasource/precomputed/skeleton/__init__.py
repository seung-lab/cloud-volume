from .metadata import PrecomputedSkeletonMetadata
from .sharded import ShardedPrecomputedSkeletonSource
from .unsharded import UnshardedPrecomputedSkeletonSource

class PrecomputedSkeletonSource(object):
  def __new__(cls, meta, cache, config):
    skel_meta = PrecomputedSkeletonMetadata(meta, cache)

    if skel_meta.is_sharded():
      return ShardedPrecomputedSkeletonSource(skel_meta, cache, config)      

    return UnshardedPrecomputedSkeletonSource(skel_meta, cache, config)
