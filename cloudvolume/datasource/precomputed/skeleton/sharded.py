from ..sharding import ShardingSpecification, ShardReader
from ....skeleton import Skeleton
from ..spatial_index import CachedSpatialIndex
from ....exceptions import EmptyFileException

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
      mip = self.meta.mip or 0
      self.spatial_index = CachedSpatialIndex(
        self.cache, self.config,
        cloudpath=self.meta.layerpath, 
        bounds=self.meta.meta.bounds(mip),
        resolution=self.meta.info['spatial_index'].get('resolution', self.meta.meta.resolution(mip)),
        chunk_size=self.meta.info['spatial_index']['chunk_size'],
      )

  @property
  def path(self):
    return self.meta.skeleton_path

  def get(self, segids):
    list_return = True
    if type(segids) in (int, float):
      list_return = False
      segids = [ int(segids) ]

    # compress = self.config.compress 
    # if compress is None:
    #   compress = True

    results = []
    binaries = self.reader.get_data(
      segids, self.meta.skeleton_path, 
      progress=self.config.progress
    )

    for segid in segids:
      binary = binaries[segid]
      del binaries[segid]

      if binary is None:
        raise EmptyFileException("segid {} is missing.".format(segid))

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

  def upload(self, *args, **kwargs):
    raise NotImplementedError()

  def raw_upload(self, *args, **kwargs):
    raise NotImplementedError()

  def get_bbox(self, bbox):
    if self.spatial_index is None:
      raise IndexError("A spatial index has not been created.")

    segids = self.spatial_index.query(bbox)
    return self.get(segids)