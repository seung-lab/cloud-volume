from typing import Optional

import numpy as np

from ..sharding import ShardingSpecification, ShardReader, synthesize_shard_files
from ....skeleton import Skeleton
from ..spatial_index import CachedSpatialIndex
from ....exceptions import EmptyFileException

from cloudfiles import CloudFiles

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
    if isinstance(segids, (int,float,np.integer)):
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

  def raw_upload(self, *args, **kwargs):
    raise NotImplementedError()

  def upload(self, skeletons):
    if type(skeletons) == Skeleton:
      skeletons = [ skeletons ]

    skeletons = { skel.id: skel.to_precomputed() for skel in skeletons }
    shard_files = synthesize_shard_files(self.reader.spec, skeletons)

    if len(shard_files) != 1:
      raise ValueError(
        f"Only one shard file should be generated per task. "
        f"Expected: {self.shard_no} Got: {', '.join(shard_files.keys())} "
      )

    cf = CloudFiles(self.meta.layerpath, progress=self.config.progress)
    cf.puts( 
      ( (fname, data) for fname, data in shard_files.items() ),
      compress=False,
      content_type='application/octet-stream',
      cache_control='no-cache',      
    )

  def to_sharded(
    self,
    num_labels:int,
    shard_index_bytes:int = 2**13,
    minishard_index_bytes:int = 2**15,
    min_shards:int = 1,
    minishard_index_encoding:str = 'gzip', 
    data_encoding:str = 'gzip',
    max_labels_per_shard:Optional[int] = None,
  ):
    return self.meta.to_sharded(
      num_labels=num_labels,
      shard_index_bytes=shard_index_bytes,
      minishard_index_bytes=minishard_index_bytes,
      min_shards=min_shards,
      minishard_index_encoding=minishard_index_encoding,
      data_encoding=data_encoding,
      max_labels_per_shard=max_labels_per_shard,
    )

  def to_unsharded(self):
    return self.meta.to_unsharded()

  # harmonize interface with mesh sources
  def put(self, *args, **kwargs):
    return self.upload(*args, **kwargs)

  def get_bbox(self, bbox):
    if self.spatial_index is None:
      raise IndexError("A spatial index has not been created.")

    segids = self.spatial_index.query(bbox)
    return self.get(segids)