from ..sharding import ShardingSpecification
from ....skeleton import Skeleton


class ShardedPrecomputedSkeletonSource(object):
  def __init__(self, meta, cache, config):
    self.meta = meta
    self.cache = cache
    self.config = config

    self.spec = ShardingSpecification.from_dict(self.meta.info)

  @property
  def path(self):
    return self.meta.path 

  def get(self, segid):
    pass

  def upload(self):
    pass