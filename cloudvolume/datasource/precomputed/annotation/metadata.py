from typing import Literal, NamedTuple, Optional, Union, Any, cast, Iterable

from collections import OrderedDict
from dataclasses import dataclass

import posixpath
import os

from cloudfiles import CloudFiles

import numpy as np
import numpy.typing as npt

from ....types import CacheType, StrEnum
from ....lib import Bbox
from ....paths import strict_extract, to_https_protocol

ANNOTATION_INFO_TYPE = "neuroglancer_annotations_v1"

class AnnotationType(StrEnum):
  POINT = "POINT"
  LINE = "LINE"
  AXIS_ALIGNED_BOUNDING_BOX = "AXIS_ALIGNED_BOUNDING_BOX"
  ELLIPSOID = "ELLIPSOID"
  POLYLINE = "POLYLINE"

def crop_mask(type:AnnotationType, geometry:np.ndarray, bbox:Bbox) -> npt.NDArray[bool]:
  lower_bound = np.array(bbox.minpt)
  upper_bound = np.array(bbox.maxpt)

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

_PROPERTY_DTYPES: dict[
  str, tuple[Union[tuple[str], tuple[str, tuple[int, ...]]], int]
] = {
  "uint8": (("|u1",), 1),
  "uint16": (("<u2",), 2),
  "uint32": (("<u4",), 4),
  "int8": (("|i1",), 1),
  "int16": (("<i2",), 2),
  "int32": (("<i4",), 4),
  "float32": (("<f4",), 4),
  "rgb": (("|u1", (3,)), 1),
  "rgba": (("|u1", (4,)), 1),
}

def _get_dtype_for_geometry(annotation_type:AnnotationType, rank:int):
  geometry_size = rank if annotation_type == AnnotationType.POINT else 2 * rank
  return [("geometry", "<f4", geometry_size)]

def _get_dtype_for_properties(properties:Iterable[dict[str, Any]]):
  dtype = []
  offset = 0
  for i, p in enumerate(properties):
    dtype_entry, alignment = _PROPERTY_DTYPES[p["type"]]
    # if offset % alignment:
    #     padded_offset = (offset + alignment - 1) // alignment * alignment
    #     padding = padded_offset - offset
    #     dtype.append((f"padding{offset}", "|u1", (padding,)))
    #     offset += padding
    dtype.append((f"{p['id']}", *dtype_entry))  # type: ignore[arg-type]
    size = np.dtype(dtype[-1:]).itemsize
    offset += size
  alignment = 4
  if offset % alignment:
    padded_offset = (offset + alignment - 1) // alignment * alignment
    padding = padded_offset - offset
    dtype.append((f"padding{offset}", "|u1", (padding,)))
    offset += padding
  return dtype

class PrecomputedAnnotationMetadata:
  def __init__(
    self, 
    cloudpath:str,
    cache:CacheType, 
    config:Optional["SharedConfiguration"] = None, 
    info:Optional[dict] = None, 
    readonly:bool = False,
    use_https:bool = False,
  ):

    path = strict_extract(cloudpath)
    if use_https:
      cloudpath = to_https_protocol(path)

    self.path = path
    self.cloudpath = cloudpath
    self.cache = cache
    self.config = config
    self.readonly = readonly

    if info:
      self.info = info
    else:
      self.info = self.fetch_info()

    typ = self.info.get("@type", "")

    if typ != ANNOTATION_INFO_TYPE:
      raise ValueError(f"info @type must be {ANNOTATION_INFO_TYPE}. Got: {typ}")

  def join(self, *paths):
    if self.path.protocol == 'file':
      return os.path.join(*paths)
    else:
      return posixpath.join(*paths)

  def fetch_info(self):
    return CloudFiles(self.cloudpath, secrets=self.config.secrets).get_json('info')

  def default_info(self):
    return {
      "@type": ANNOTATION_INFO_TYPE,
      "dimensions": [], # e.g. [[ 32.0, "nm" ], ... ]
      "lower_bound": [],
      "upper_bound": [],
      "annotation_type": AnnotationType.POINT, 
      "properties": [{
        # "id": "class_label"
        # "type": "int32"
        # "description": ""
        # "enum_labels": [0,1,2,3,4,5000],
        # "enum_values": ["axon", "dendrite", "astrocyte", "soma", ...]
        # 
      }],
      # e.g. [ { "id": , "key": ..., "sharding": ...  } ]
      "relationships": [],
      # e.g. [ { "key": ..., "sharding": ...  } ]
      "by_id": [], 
      # e.g. [ { "key", "sharding", "grid_shape", "chunk_size", "limit" } ] 
    }

  @property
  def ndim(self) -> int:
    return len(self.info["dimensions"])

  @property
  def properties(self) -> dict:
    return self.info.get("properties", [])

  def properties_enum(self) -> dict[str, dict[int, str]]:
    enums = {}
    for p in self.properties:
      if "enum_labels" in p:
        self._enum_dict[p['id']] = {
          k: v for k, v in zip(p["enum_values"], p["enum_labels"])
        }
    return enums

  @property
  def relationship_names(self) -> list[str]:
    if "relationships" not in self.info.keys():
      raise ValueError("No relationships found in the info file.")
    return [ r["key"] for r in self.info["relationships"] ]

  @property
  def property_names(self) -> list[str]:
    """Get the properties of the annotations.

    Returns:
      list: A list of property names.
    """
    if "properties" not in self.info.keys():
      raise ValueError("No properties found in the info file.")
    return [ p["id"] for p in self.info["properties"] ]

  @property
  def annotation_dtype(self) -> np.dtype:
    return (
      _get_dtype_for_geometry(self.annotation_type, self.ndim)
      + _get_dtype_for_properties(self.properties)
    )

  @property
  def bounds(self) -> Bbox:
    return Bbox(self.info["lower_bound"], self.info["upper_bound"])

  def chunk_size(self, mip:int) -> np.ndarray:
    return np.array(self.info["spatial"][mip]["chunk_size"], dtype=int)

  def grid_shape(self, mip:int) -> np.ndarray:
    return np.array(self.info["spatial"][mip]["grid_shape"], dtype=int)

  @property
  def annotation_type(self) -> AnnotationType:
    return AnnotationType(self.info["annotation_type"])

  def is_id_index_sharded(self) -> bool:
    if "by_id" not in self.info:
      return False

    index = self.info["by_id"]
    return index.get("sharding", None) is not None


