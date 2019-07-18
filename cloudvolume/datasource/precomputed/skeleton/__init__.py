from .metadata import PrecomputedSkeletonMetadata
from .unsharded import UnshardedPrecomputedSkeletonSource

class PrecomputedSkeletonSource(object):
  def __new__(cls, meta, cache, config):
    skel_meta = PrecomputedSkeletonMetadata(meta, cache)

    if skel_meta.is_sharded():
      raise NotImplementedError()

    return UnshardedPrecomputedSkeletonSource(skel_meta, cache, config)
