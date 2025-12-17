"""
Precomputed Annotation Reader code adapted from
a mixture of these two repositories:

https://github.com/fcollman/precomputed_python
https://github.com/google/neuroglancer/
"""

from typing import Optional, Union, Iterable

import os
import posixpath

from cloudfiles import CloudFiles
import numpy as np
import numpy.typing as npt

from tqdm import tqdm

from .metadata import (
  PrecomputedAnnotationMetadata, 
  LabelAnnotation, 
  MultiLabelAnnotation,
)

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

  def ids(self) -> npt.NDArray[np.uint64]:
    """Get all annotation IDs from the kv store.

    Returns:
      np.array: An array of all annotation IDs.
    """
    if "by_id" not in self.meta.info:
      raise ValueError("No by_id information found in the info file.")
    
    cf = CloudFiles(self.meta.cloudpath)

    by_id = self.meta.info["by_id"]

    key = by_id["key"]
    if not key.endswith(cf.sep):
      key += cf.sep

    cache_key = os.path.join(key, "_ids.npy")
    cache_path = os.path.join(self.cache.path.replace("file://", ""), cache_key)

    if self.cache.enabled:
      if self.cache.has(cache_key):
        return np.load(cache_path, allow_pickle=False)

    if self.meta.is_id_index_sharded():
      shard_filenames = [ 
        x for x in cf.list(prefix=key, flat=True) 
        if x.endswith('.shard')
      ]
      spec = ShardingSpecification.from_dict(by_id["sharding"])
      reader = ShardReader(self.meta.cloudpath, self.cache, spec)

      all_ids = []
      for fname in tqdm(shard_filenames, disable=(not self.config.progress), desc="Downloading Labels"):
        all_ids.append(
          reader.list_labels(fname, path=key)
        )
      all_ids = np.concatenate(all_ids, dtype=np.uint64)
      all_ids.sort()
    else:
      cf = CloudFiles(self.cloudpath)
      all_ids = ( int(x) for x in cf.list(prefix=key, flat=True) )
      all_ids = np.fromiter(all_ids, dtype=np.uint64)

    if self.cache.enabled:
      np.save(cache_path, all_ids)

    return all_ids

  def _annotation_dtype(self, binary:bytes):
    ndim = self.meta.ndim

    prop_dtypes = [ (prop["id"], prop["type"]) for prop in self.meta.properties ]

    # Derived from Neuroglancer Python code
    if self.meta.annotation_type == AnnotationType.POLYLINE:
      num_pts = np.frombuffer(encoded, dtype="<u4", count=1)[0]
      num_points = ("num_points", "<u4")
      geometry = (
        "_pt1",
        "<f4",
        (num_points_value * self.coordinate_space.rank,),
      )
      return [num_points, geometry] + prop_dtypes

    two_point_types = (
      AnnotationType.LINE,
      AnnotationType.AXIS_ALIGNED_BOUNDING_BOX,
      AnnotationType.ELLIPSOID,
    )

    geometry_dtype = [('_pt1', 'f4', ndim)]

    if self.meta.annotation_type in two_point_types:
      geometry_dtype += [('_pt2', 'f4', ndim)]

    return geometry_dtype + prop_dtypes

  def _decode_single_annotation(self, binary:bytes):
    ndim = self.meta.ndim
    offset = 0

    decoded = np.frombuffer(
      binary,
      offset=offset,
      count=1,
      dtype=self._annotation_dtype(binary),
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

  def _decode_label_annotation(self, segid:int, binary:bytes) -> LabelAnnotation:
    (geometry, properties, relationships) = self._decode_single_annotation(binary)
    return LabelAnnotation(
      segid,
      self.meta.annotation_type,
      geometry,
      properties,
      relationships,
    )

  def _decode_annotations(self, binary:bytes) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    ndim = self.meta.ndim
    num_points = int.from_bytes(binary[:8], 'little')
    offset = 8

    decoded = np.frombuffer(
      binary,
      offset=offset,
      count=num_points,
      dtype=self._annotation_dtype(binary),
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

  def get_all(self) -> dict[int, LabelAnnotation]:
    """Retrieve all annotations."""
    if self.meta.has_spatial_index():
      slcs = tuple([ slice(None) for i in range(self.meta.ndim) ])
      return self.get_by_bbox(slcs)
    else:
      # This branch could be radically sped up if needed
      # by pulling the shards and disassembling them directly
      return self.get_by_id(self.ids())

  def get_by_id(self, labels:Union[int, list[int]]) -> Union[LabelAnnotation, dict[int, LabelAnnotation]]:
    """
    Retrieve annotations by one or more IDs.
    """
    labels, return_multiple = toiter(labels, is_iter=True)
    by_id = self.meta.info["by_id"]

    if self.meta.is_id_index_sharded():
      spec = ShardingSpecification.from_dict(by_id["sharding"])
      reader = ShardReader(self.meta.cloudpath, self.cache, spec)
      result = reader.get_data(labels, path=by_id["key"])
      annos = {
        segid: self._decode_label_annotation(segid, binary)
        for segid, binary in result.items()
      }
    else:
      filenames = [
        self.meta.join(by_id["key"], str(segid))
        for segid in labels
      ]
      annotations = self.cache.download(filenames)

      annos = {}
      for path, binary in annotations.items():
        segid = int(os.path.basename(path))
        annos[segid] = self._decode_label_annotation(segid, binary)

    if return_multiple:
      return annos
    return annos[labels[0]]

  def get_by_bbox(self, bbox:BboxLikeType, mip:int = 0) -> MultiLabelAnnotation:
    """
    Query for all annotations in the given bounding box.
    Bounds are inclusive on both sides.
    """
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
        self.meta.join(
          key, 
          "_".join([ str(x) for x in pt ])
        )
        for pt in grid
      ]
      annotations = self.cache.download(filenames)
      annotations = [ binary for binary in annotations.values() ]

    all_geo = []
    ids = []
    properties = {}
    for binary in annotations:
      geometry, annotation_ids, props = self._decode_annotations(binary)
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
      type=self.meta.annotation_type,
      geometry=all_geo,
      ids=ids,
      properties=properties,
    ).crop(orig_bbox)

  def get_by_relationship(self, relationship:str, labels:Union[int, Iterable[int]]) -> npt.NDArray[np.uint64]:
    """
    Get the annotations corresponding to the relationship type.
    """
    labels, return_multiple = toiter(labels, is_iter=True)

    rels = self.meta.relationships

    if relationship not in rels:
      raise ValueError(f"Relationship {relationship} not found. Available: {','.join(rels.keys())}")

    rel = rels[relationship]

    if 'sharding' in rel and rel.get("sharding", None) is not None:
      spec = ShardingSpecification.from_dict(rel["sharding"])
      reader = ShardReader(self.meta.cloudpath, self.cache, spec)
      binaries = reader.get_data(labels, path=rel["key"])
    else:
      filenames = [ self.meta.join(rel["key"], str(segid)) for segid in labels ]
      binaries = self.cache.download(filenames)
      binaries = { 
        int(os.path.basename(path)): binary 
        for path, binary in binaries.items() 
      }
      
    ret = {}

    for label, binary in binaries.items():
      geometry, ids, properties = self._decode_annotations(binary)
      ret[label] = MultiLabelAnnotation(
        type=self.meta.annotation_type,
        geometry=geometry,
        ids=ids,
        properties=properties,
      )

    if return_multiple:
      return ret

    return ret[next(iter(ret.keys()))]
