from typing import Optional, Union
from dataclasses import dataclass

import os
import posixpath

from cloudfiles import CloudFiles
import numpy as np
import numpy.typing as npt

from .metadata import PrecomputedAnnotationMetadata

from .. import get_cache_path
from ....cacheservice import CacheService
from ....paths import strict_extract
from ....cloudvolume import SharedConfiguration

from ....types import SecretsType
from ....lib import Bbox, BboxLikeType
from ....paths import strict_extract, to_https_protocol, ascloudpath
from ..common import compressed_morton_code
from ..sharding import ShardReader, ShardingSpecification

from .metadata import PrecomputedAnnotationMetadata, AnnotationType

# sharded example 
# https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B6.4e-8%2C%22m%22%5D%2C%22y%22:%5B6.4e-8%2C%22m%22%5D%2C%22z%22:%5B6.6e-8%2C%22m%22%5D%7D%2C%22position%22:%5B34724.5%2C23270.5%2C584.5%5D%2C%22crossSectionScale%22:1%2C%22projectionOrientation%22:%5B-0.09734609723091125%2C-0.26029738783836365%2C-0.0020852696616202593%2C0.9606063961982727%5D%2C%22projectionScale%22:20037.381619627573%2C%22layers%22:%5B%7B%22type%22:%22segmentation%22%2C%22source%22:%22gs://h01-release/data/20210601/c2/subcompartments/%7Cneuroglancer-precomputed:%22%2C%22tab%22:%22source%22%2C%22segments%22:%5B%22%21103%22%5D%2C%22name%22:%22subcompartments%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22annotation%22%2C%22source%22:%22gs://h01-release/data/20210601/c2/subcompartments/annotations/%7Cneuroglancer-precomputed:%22%2C%22tab%22:%22source%22%2C%22name%22:%22new%20layer%22%7D%5D%2C%22selectedLayer%22:%7B%22visible%22:true%2C%22layer%22:%22new%20layer%22%7D%2C%22layout%22:%22xy-3d%22%7D

@dataclass
class LabelAnnotation:
  id: int
  type: AnnotationType
  geometry: npt.NDArray[np.float32]
  properties: dict[str, np.ndarray]

@dataclass
class SpatialAnnotation:
  type: AnnotationType
  geometry: npt.NDArray[np.float32]
  ids: npt.NDArray[np.uint64]
  properties: dict[str, np.ndarray]

class PrecomputedAnnotationSource:
  def __init__(
    self,
    cloudpath:str,
    cache:Optional[str] = None,
    config:Optional[SharedConfiguration] = None,
    readonly:bool = False,
    secrets:SecretsType = None,
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
        secrets=secrets,
        spatial_index_db=None,
        cache_locking=False,
        codec_threads=1,
      )

    self.path = path
    self.meta = PrecomputedAnnotationMetadata(
      cloudpath, cache, config,
      info=info, readonly=readonly
    )
    self.cloudpath = cloudpath
    self.cache = CacheService(
      cloudpath=get_cache_path(cache, cloudpath),
      enabled=bool(cache),
      config=config,
      compress=True,
      meta=self.meta,
    )
    self.config = config
    self.readonly = bool(readonly)

  def join(self, *paths):
    if self.path.protocol == 'file':
      return os.path.join(*paths)
    else:
      return posixpath.join(*paths)

  def _annotation_dtype(self):
    ndim = self.meta.ndim
    prop_dtypes = [ (prop["id"], prop["type"]) for prop in self.meta.properties ]
    return [('_geometry', 'f4', ndim)] + prop_dtypes

  def decode_annotations(self, binary:bytes) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    ndim = self.meta.ndim
    num_points = int.from_bytes(binary[:8], 'little')
    offset = 8

    decoded = np.frombuffer(
      binary,
      offset=offset,
      count=num_points,
      dtype=self._annotation_dtype(),
    )
    geometry = decoded["_geometry"]
    offset += decoded.nbytes

    properties = {}
    for prop in self.meta.properties:
      properties[prop["id"]] = decoded[prop["id"]]

    ids = np.frombuffer(
      binary,
      offset=offset,
      count=num_points,
      dtype="<u8",
    )
    offset += ids.nbytes
    assert offset == len(binary)

    return (geometry, ids, properties)

  def get_by_id(self, segids:Union[int, list[int]]) -> dict[Union[str,int], LabelAnnotation]:
    annos = {}

    cloudpath = self.join(self.cloudpath, self.meta.info["by_id"]["key"])

    if self.meta.is_id_index_sharded():
      spec = ShardingSpecification.from_dict(self.meta.info["by_id"]["sharding"])
      reader = ShardReader(cloudpath, self.cache, spec)
      annotations = reader.get_data(segids)
    else:
      cf = CloudFiles(cloudpath, secrets=self.config.secrets)
      annotations = cf.get([ f"{segid}" for segid in segids ])
      for file in annotations:
        binary = file["content"]
        segid = int(file["path"])
        annos[segid] = self.decode_annotations(binary)

    return LabelAnnotation(
      segids,
      self.meta.annotation_type,
      all_geo,
      properties,
    )

  def get_by_bbox(self, bbox:BboxLikeType, mip:int = 0) -> SpatialAnnotation:
    spatial = self.meta.info["spatial"][mip]
    key = spatial["key"]

    spatial_path = self.join(self.cloudpath, key)

    realized_bbox = Bbox.create(bbox, self.meta.bounds())
    realized_bbox = realized_bbox.expand_to_chunk_size(
      self.meta.chunk_size(mip),
      offset=self.meta.bounds().minpt,
    )
    realized_bbox = Bbox.clamp(realized_bbox, self.meta.bounds())
    realized_bbox /= self.meta.chunk_size(mip)

    grid = np.mgrid[realized_bbox.to_slices()][...,0,0].T
    if spatial.get("sharding", None) is not None:
      codes = compressed_morton_code(grid, self.meta.grid_shape(mip))
      spec = ShardingSpecification.from_dict(spatial["sharding"])
      reader = ShardReader(spatial_path, self.cache, spec)
      annotations = reader.get_data(codes)
      annotations = [ f["content"] for f in annotations.values() if f is not None ]
    else:
      filenames = [
        "_".join([ str(x) for x in pt ])
        for pt in grid
      ]
      cf = CloudFiles(spatial_path, secrets=self.config.secrets)
      annotations = cf.get(filenames)
      annotations = [ f["content"] for f in annotations ]

    all_geo = []
    ids = []
    properties = {}
    for binary in annotations:
      geometry, annotation_ids, props = self.decode_annotations(binary)
      all_geo.append(geometry)
      ids.append(annotation_ids)

      for prop in props.keys():
        if prop not in properties:
          properties[prop] = props[prop]
        else:
          properties[prop] = np.concatenate([properties[prop], props[prop]])

      if len(all_geo):
        all_geo = np.concatenate(all_geo, axis=0)
      else:
        all_geo = np.zeros([0,N], dtype=np.float32)

      if len(ids):
        ids = np.concatenate(ids)
      else:
        ids = np.zeros([0,], dtype=np.uint64)

    return SpatialAnnotation(
      self.meta.annotation_type,
      all_geo,
      ids,
      properties,
    )
