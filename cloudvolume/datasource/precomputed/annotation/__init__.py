from typing import Optional, Union

import os
import posixpath

from cloudfiles import CloudFiles
import numpy as np

from .metadata import PrecomputedAnnotationMetadata

from .. import get_cache_path
from ....cacheservice import CacheService
from ....paths import strict_extract
from ....cloudvolume import SharedConfiguration

from ....lib import Bbox, BboxLikeType
from ....paths import strict_extract, to_https_protocol, ascloudpath
from ..common import compressed_morton_code
from ..sharding import ShardReader, ShardingSpecification


from .metadata import PrecomputedAnnotationMetadata

# sharded example 
# https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B6.4e-8%2C%22m%22%5D%2C%22y%22:%5B6.4e-8%2C%22m%22%5D%2C%22z%22:%5B6.6e-8%2C%22m%22%5D%7D%2C%22position%22:%5B34724.5%2C23270.5%2C584.5%5D%2C%22crossSectionScale%22:1%2C%22projectionOrientation%22:%5B-0.09734609723091125%2C-0.26029738783836365%2C-0.0020852696616202593%2C0.9606063961982727%5D%2C%22projectionScale%22:20037.381619627573%2C%22layers%22:%5B%7B%22type%22:%22segmentation%22%2C%22source%22:%22gs://h01-release/data/20210601/c2/subcompartments/%7Cneuroglancer-precomputed:%22%2C%22tab%22:%22source%22%2C%22segments%22:%5B%22%21103%22%5D%2C%22name%22:%22subcompartments%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22annotation%22%2C%22source%22:%22gs://h01-release/data/20210601/c2/subcompartments/annotations/%7Cneuroglancer-precomputed:%22%2C%22tab%22:%22source%22%2C%22name%22:%22new%20layer%22%7D%5D%2C%22selectedLayer%22:%7B%22visible%22:true%2C%22layer%22:%22new%20layer%22%7D%2C%22layout%22:%22xy-3d%22%7D

class PrecomputedAnnotationSource:
  def __init__(
    self,
    cloudpath:str,
    cache:Optional[str] = None,
    config:Optional[SharedConfiguration] = None,
    readonly:bool = False,
    info:Optional[dict] = None,
    use_https:bool = False,
  ):
    path = strict_extract(cloudpath)
    if use_https:
      path = to_https_protocol(path)
      cloudpath = ascloudpath(path)
    
    if config is None:
      config = SharedConfiguration(
        cdn_cache=False,
        compress=False, 
        compress_level=5,
        green=False,
        mip=0,
        parallel=1,
        progress=False,
        secrets={},
        spatial_index_db=None,
        cache_locking=False,
        codec_threads=1,
      )

    self.path = path
    self.meta = PrecomputedAnnotationMetadata(cloudpath, cache, config, info=info, readonly=readonly)
    self.cloudpath = cloudpath
    self.cache = CacheService(
      cloudpath=get_cache_path(cache, cloudpath),
      enabled=bool(cache),
      config=config,
      compress=True,
    )
    self.config = None
    self.readonly = bool(readonly)

  def join(self, *paths):
    if self.path.protocol == 'file':
      return os.path.join(*paths)
    else:
      return posixpath.join(*paths)

  def get_by_id(self, segids:Union[int, list[int]]) -> dict[int, np.ndarray]:
    
    if self.meta.is_id_index_sharded():
      spec = self.meta.info["by_id"]["sharding"]
      reader = ShardReader(self.cloudpath, self.cache, spec)
      annotations = reader.get_data(segids)
    else:
      cf = CloudFiles(self.cloudpath)
      annotations = cf.get([ f"{segid}" for segid in segids ])

    N = len(self.meta.info["dimensions"])

    pts = {}
    for segid, binary in annotations:
      pts[segid] = np.frombuffer(binary, dtype=np.float32).reshape((N, len(binary) // N) )

    return pts

  def get_by_bbox(self, bbox:BboxLikeType, mip:int = 0) -> dict[int, np.ndarray]:
    spatial = self.meta.info["spatial"][mip]
    key = spatial["key"]

    spatial_path = self.join(self.cloudpath, key)

    orig_bbox = bbox.clone()
    bbox = Bbox.create(bbox, self.meta.bounds())
    bbox = bbox.expand_to_chunk_size(
      self.meta.chunk_size(mip),
      offset=self.meta.bounds().minpt,
    )
    bbox = Bbox.clamp(bbox, self.meta.bounds())
    bbox /= self.meta.chunk_size(mip)

    grid = np.mgrid[bbox.to_slices()][...,0,0].T

    if spatial.get("sharding", None) is not None:
      codes = compressed_morton_code(grid, self.meta.grid_shape(mip))
      spec = ShardingSpecification.from_dict(spatial["sharding"])
      reader = ShardReader(spatial_path, self.cache, spec)
      annotations = reader.get_data(codes)
    else:
      filenames = [
        "_".join([ str(x) for x in pt ])
        for pt in grid
      ]
      cf = CloudFiles(spatial_path)
      annotations = cf.get(filenames)

    N = self.meta.ndim

    pts = []
    for file in annotations:
      binary = file["content"]
      pts.append(
        np.frombuffer(binary, dtype=np.float32).reshape((len(binary) // N, 3), order="C")
      )

    return np.concatenate(*pts, axis=0)

  @classmethod
  def from_cloudpath(
    cls, 
    cloudpath:str, 
    cache=False, 
    progress=False,
    secrets=None,
    spatial_index_db:Optional[str]=None, 
    cache_locking:bool = True,
  ):
    config = SharedConfiguration(
      cdn_cache=False,
      compress=True,
      compress_level=None,
      green=False,
      mip=0,
      parallel=1,
      progress=progress,
      secrets=secrets,
      spatial_index_db=spatial_index_db,
      cache_locking=cache_locking,
    )
    
    cache = CacheService(
      cloudpath=(cache if type(cache) == str else cloudpath),
      enabled=bool(cache),
      config=config,
      compress=True,
    )

    return PrecomputedAnnotationSource(cloudpath, cache, config)