from ..sharding import ShardingSpecification, ShardReader
from ....skeleton import Skeleton


class ShardedPrecomputedSkeletonSource(object):
  def __init__(self, meta, cache, config):
    self.meta = meta
    self.cache = cache
    self.config = config

    spec = ShardingSpecification.from_dict(self.meta.info['sharding'])
    self.reader = ShardReader(meta, cache, spec)

  @property
  def path(self):
    return self.meta.path 

  def get(self, segids):
    list_return = True
    if type(segids) in (int, float):
      list_return = False
      segids = [ int(segids) ]

    # compress = self.config.compress 
    # if compress is None:
    #   compress = True

    results = []
    for segid in segids:
      binary = self.reader.get_data(segid)
      skeleton = Skeleton.from_precomputed(
        binary, segid=segid, 
        vertex_attributes=self.meta.info['vertex_attributes']
      )
      skeleton.transform = self.meta.transform
      results.append(skeleton.physical_space())

    if list_return:
      return results
    else:
      return results[0]

  def upload(self):
    raise NotImplementedError()

  def raw_upload(self):
    raise NotImplementedError()