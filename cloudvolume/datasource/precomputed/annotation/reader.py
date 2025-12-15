from typing import Optional, Union, Iterable
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
from ....lib import Bbox, BboxLikeType, toiter
from ....paths import strict_extract, to_https_protocol, ascloudpath
from ..common import compressed_morton_code
from ..sharding import ShardReader, ShardingSpecification

from .metadata import PrecomputedAnnotationMetadata, AnnotationType

# sharded example 
# https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B6.4e-8%2C%22m%22%5D%2C%22y%22:%5B6.4e-8%2C%22m%22%5D%2C%22z%22:%5B6.6e-8%2C%22m%22%5D%7D%2C%22position%22:%5B34724.5%2C23270.5%2C584.5%5D%2C%22crossSectionScale%22:1%2C%22projectionOrientation%22:%5B-0.09734609723091125%2C-0.26029738783836365%2C-0.0020852696616202593%2C0.9606063961982727%5D%2C%22projectionScale%22:20037.381619627573%2C%22layers%22:%5B%7B%22type%22:%22segmentation%22%2C%22source%22:%22gs://h01-release/data/20210601/c2/subcompartments/%7Cneuroglancer-precomputed:%22%2C%22tab%22:%22source%22%2C%22segments%22:%5B%22%21103%22%5D%2C%22name%22:%22subcompartments%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22annotation%22%2C%22source%22:%22gs://h01-release/data/20210601/c2/subcompartments/annotations/%7Cneuroglancer-precomputed:%22%2C%22tab%22:%22source%22%2C%22name%22:%22new%20layer%22%7D%5D%2C%22selectedLayer%22:%7B%22visible%22:true%2C%22layer%22:%22new%20layer%22%7D%2C%22layout%22:%22xy-3d%22%7D

def crop_mask(type:AnnotationType, geometry:np.ndarray, bbox:Bbox) -> npt.NDArray[bool]:
  lower_bound = bbox.minpt
  upper_bound = bbox.maxpt

  if type == AnnotationType.POINT:
    points = geometry
    mask = np.all(points >= lower_bound, axis=1) & np.all(
      points <= upper_bound, axis=1
    )
  elif type == "line":
    # Combine point_a and point_b into a single 2D array for vectorized comparison
    points_a = geometry[:,:,0]
    points_b = geometry[:,:,1]
    mask = (
      np.all(points_a >= lower_bound, axis=1)
      & np.all(points_a <= upper_bound, axis=1)
    ) | (
      np.all(points_b >= lower_bound, axis=1)
      & np.all(points_b <= upper_bound, axis=1)
    )
  elif type == AnnotationType.AXIS_ALIGNED_BOUNDING_BOX:
    # Combine point_a and point_b into a single 2D array for vectorized comparison
    points_a = geometry[:,:,0]
    points_b = geometry[:,:,1]
    mask = (
      (
        np.all(points_a >= lower_bound, axis=1)
        & np.all(points_a <= upper_bound, axis=1)
      )
      | (
        np.all(points_b >= lower_bound, axis=1)
        & np.all(points_b <= upper_bound, axis=1)
      )
      | (
        np.all(points_a <= lower_bound, axis=1)
        & np.all(points_b >= upper_bound, axis=1)
      )
      | (
        np.all(points_b <= lower_bound, axis=1)
        & np.all(points_a >= upper_bound, axis=1)
      )
    )
  elif type == AnnotationType.ELLIPSOID:
    # Combine center into a single 2D array for vectorized comparison
    center = geometry[:,:,0]
    radius = geometry[:,:,1]
    mask = np.all(centers >= lower_bound, axis=1) & np.all(
      centers <= upper_bound, axis=1
    )
  else:
    raise TypeError(f"{type} is not supported by crop.")

  return mask

@dataclass
class LabelAnnotation:
  id: int
  type: AnnotationType
  geometry: npt.NDArray[np.float32]
  properties: dict[str, np.ndarray]
  relationships: dict[str, npt.NDArray[np.uint64]]

  def pandas(self):
    import pandas as pd
    data = {}
    data.update(self.properties)
    data["point"] = self.geometry
    df = pd.DataFrame(data)
    return df

  def crop(self, bbox:Bbox) -> "LabelAnnotation":
    mask = crop_mask(self.type, self.geometry, bbox)
    return LabelAnnotation(
      id=self.id,
      type=self.type,
      geometry=self.geometry[mask],
      properties={
        k: v[mask]
        for k,v in self.properties.items()
      },
      relationships=self.relationships,
    )

@dataclass
class MultiLabelAnnotation:
  type: AnnotationType
  geometry: npt.NDArray[np.float32]
  ids: npt.NDArray[np.uint64]
  properties: dict[str, np.ndarray]

  def pandas(self):
    import pandas as pd
    data = { "ids": self.ids }
    data.update(self.properties)
    data["point"] = self.geometry
    df = pd.DataFrame(data)
    df.set_index("ID", inplace=True)
    return df

  def crop(self, bbox:Bbox) -> "MultiLabelAnnotation":
    mask = crop_mask(self.type, self.geometry, bbox)
    return MultiLabelAnnotation(
      type=self.type,
      geometry=self.geometry[mask],
      ids=self.geometry[mask],
      properties={
        k: v[mask]
        for k,v in self.properties.items()
      },
    )

class PrecomputedAnnotationReader:
  def __init__(
    self,
    meta:PrecomputedAnnotationMetadata,
    cache:CacheService,
    config:SharedConfiguration,
    readonly:bool = False,
    secrets:SecretsType = None,
    info:Optional[dict] = None,
    use_https:bool = False,
  ):
    self.meta = meta
    self.cache = cache
    self.config = config
    self.readonly = bool(readonly)

  def ids(self) -> Iterable[int]:
    """Get all annotation IDs from the kv store.

    Returns:
      np.array: An array of all annotation IDs.
    """
    if "by_id" not in self.meta.info:
      raise ValueError("No by_id information found in the info file.")
    
    by_id = self.meta.info["by_id"]
    cf = CloudFiles(self.cloudpath)
    if self.meta.is_id_index_sharded():
      shard_filenames = ( x for x in cf.list(flat=True) if x.endswith('.shard') )
      spec = ShardingSpecification.from_dict(by_id["sharding"])
      reader = ShardReader(self.cloudpath, self.cache, spec)

      all_ids = []
      for fname in shard_filenames:
        all_ids.append(
          reader.list_labels(fname, path=by_id["key"])
        )
      all_ids = np.concatenate(all_ids, dtype=np.uint64)
      all_ids.sort()
      return all_ids
    else:
      cf = CloudFiles(self.cloudpath)
      all_ids = ( int(x) for x in cf.list(prefix=by_id["key"]) )
      return np.fromiter(all_ids, dtype=np.uint64)

  def _annotation_dtype(self):
    ndim = self.meta.ndim

    two_point_types = (
      AnnotationType.LINE,
      AnnotationType.AXIS_ALIGNED_BOUNDING_BOX,
      AnnotationType.ELLIPSOID,
    )

    geometry_dtype = [('_pt1', 'f4', ndim)]

    if self.meta.annotation_type in two_point_types:
      geometry_dtype += [('_pt2', 'f4', ndim)]

    prop_dtypes = [ (prop["id"], prop["type"]) for prop in self.meta.properties ]
    return geometry_dtype + prop_dtypes

  def decode_single_annotation(self, binary:bytes):
    ndim = self.meta.ndim
    offset = 0

    decoded = np.frombuffer(
      binary,
      offset=offset,
      count=1,
      dtype=self._annotation_dtype(),
    )
    geometry = decoded["_pt1"]
    if "_pt2" in decoded.dtype.names:
      geometry = np.hstack(geometry, decoded["_pt2"])

    offset += decoded.nbytes

    properties = {}
    for prop in self.meta.properties:
      properties[prop["id"]] = decoded[prop["id"]]

    relationships = {}
    for relation in self.meta.info["relationships"]:
      num_obj = int.from_bytes(binary[offset:offset+4], 'little')
      offset += 4
      object_ids = np.frombuffer(binary, offset=offset, count=num_obj, dtype=np.uint64)
      offset += object_ids.nbytes
      relationships[relation["key"]] = object_ids

    assert offset == len(binary)

    return (geometry, properties, relationships)

  def decode_label_annotation(self, segid:int, binary:bytes) -> LabelAnnotation:
    (geometry, properties, relationships) = self.decode_single_annotation(binary)
    return LabelAnnotation(
      segid,
      self.meta.annotation_type,
      geometry,
      properties,
      relationships,
    )

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
    geometry = decoded["_pt1"]
    if "_pt2" in decoded.dtype.names:
      geometry = np.hstack(geometry, decoded["_pt2"])
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

  def get_by_id(self, label:Union[int, list[int]]) -> Union[LabelAnnotation, dict[int, LabelAnnotation]]:
    label, return_multiple = toiter(label, is_iter=True)

    annos = {}

    by_id = self.meta.info["by_id"]

    if self.meta.is_id_index_sharded():
      spec = ShardingSpecification.from_dict(by_id["sharding"])
      reader = ShardReader(self.meta.cloudpath, self.cache, spec)
      result = reader.get_data(label, path=by_id["key"])
      annos = {
        segid: self.decode_label_annotation(segid, binary)
        for segid, binary in result.items()
      }
    else:
      cloudpath = self.meta.join(self.meta.cloudpath, by_id["key"])
      cf = CloudFiles(cloudpath, secrets=self.config.secrets)
      annotations = cf.get([ f"{segid}" for segid in label ])
      for file in annotations:
        binary = file["content"]
        segid = int(file["path"])
        annos[segid] = self.decode_label_annotation(segid, binary)

    if return_multiple:
      return annos
    return annos[label[0]]

  def get_by_bbox(self, bbox:BboxLikeType, mip:int = 0) -> MultiLabelAnnotation:
    spatial = self.meta.info["spatial"][mip]
    key = spatial["key"]

    spatial_path = self.meta.join(self.meta.cloudpath, key)

    realized_bbox = Bbox.create(bbox, self.meta.bounds)
    orig_bbox = realized_bbox.clone()
    realized_bbox = realized_bbox.expand_to_chunk_size(
      self.meta.chunk_size(mip),
      offset=self.meta.bounds.minpt,
    )
    realized_bbox = Bbox.clamp(realized_bbox, self.meta.bounds)
    realized_bbox /= self.meta.chunk_size(mip)

    grid_box = Bbox([0,0,0], self.meta.grid_shape(mip))
    realized_bbox = Bbox.clamp(realized_bbox, grid_box)

    grid = np.mgrid[realized_bbox.to_slices()][...,0,0].T
    if spatial.get("sharding", None) is not None:
      codes = compressed_morton_code(grid, self.meta.grid_shape(mip))
      spec = ShardingSpecification.from_dict(spatial["sharding"])
      reader = ShardReader(self.meta.cloudpath, self.cache, spec)
      annotations = reader.get_data(codes, path=key)
      annotations = [ binary for binary in annotations.values() if binary is not None ]
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

    return MultiLabelAnnotation(
      self.meta.annotation_type,
      all_geo,
      ids,
      properties,
    ).crop(orig_bbox)

