from ..sharding import ShardingSpecification, ShardReader
from ....skeleton import Skeleton
from ..spatial_index import CachedSpatialIndex

class ShardedPrecomputedSkeletonSource(object):
  def __init__(self, meta, cache, config, readonly=False):
    self.meta = meta
    self.cache = cache
    self.config = config
    self.readonly = bool(readonly)

    spec = ShardingSpecification.from_dict(self.meta.info['sharding'])
    self.reader = ShardReader(meta, cache, spec)

    self.spatial_index = None
    if self.meta.spatial_index:
      self.spatial_index = CachedSpatialIndex(
        self.cache,
        cloudpath=self.meta.layerpath, 
        bounds=self.meta.meta.bounds(0) * self.meta.meta.resolution(0),
        chunk_size=self.meta.info['spatial_index']['chunk_size'],
      )

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
      binary = self.reader.get_data(segid, self.meta.skeleton_path)
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

  def get_bbox(self, bbox):
    if self.spatial_index is None:
      raise IndexError("A spatial index has not been created.")

    segids = self.spatial_index.query(bbox)
    return self.get(segids)