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

class AnnotationType(StrEnum):
  POINT = "POINT"
  LINE = "LINE"
  AXIS_ALIGNED_BOUNDING_BOX = "AXIS_ALIGNED_BOUNDING_BOX"
  ELLIPSOID = "ELLIPSOID"
  POLYLINE = "POLYLINE"

ANNOTATION_INFO_TYPE = "neuroglancer_annotations_v1"

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

def _crop_mask(type:AnnotationType, geometry:np.ndarray, bbox:Bbox) -> npt.NDArray[bool]:
  lower_bound = np.array(bbox.minpt)
  upper_bound = np.array(bbox.maxpt)

  if type == AnnotationType.POINT:
    points = geometry
    return np.all(points >= lower_bound, axis=1) & np.all(
      points <= upper_bound, axis=1
    )
  elif type == "line":
    # Combine point_a and point_b into a single 2D array for vectorized comparison
    points_a = geometry[:,:,0]
    points_b = geometry[:,:,1]
    return (
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
    return (
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
    return np.all(center >= lower_bound, axis=1) & np.all(
      center <= upper_bound, axis=1
    )
  else:
    raise TypeError(f"{type} is not supported by crop.")

@dataclass
class LabelAnnotation:
  id: int
  type: AnnotationType
  geometry: npt.NDArray[np.float32]
  properties: dict[str, np.ndarray]
  relationships: dict[str, npt.NDArray[np.uint64]]
  properties_enum: Optional[dict[str, dict[int,str]]]
  dimensions: Optional[list[str]]

  def __len__(self) -> int:
    return self.geometry.shape[0]

  def tobytes(self) -> bytes:
    raise NotImplementedError()

  def pandas(self):
    import pandas as pd
    data = {}
    data.update(self.properties)
    for i in range(self.geometry.shape[1]):
      axis = f"axis_{i}"
      if i < len(self.dimensions):
        axis = self.dimensions[i]
      data[axis] = self.geometry[:,i]

    df = pd.DataFrame(data)

    if isinstance(self.properties_enum, dict):
      for name, enum_dict in self.properties_enum.items():
        df[name] = df[name].map(enum_dict).astype('category')

    return df

  def crop(self, bbox:Bbox) -> "LabelAnnotation":
    mask = _crop_mask(self.type, self.geometry, bbox)
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

class SpecificLabelAnnotation(LabelAnnotation):
  type: AnnotationType = AnnotationType.POINT
  def __init__(self, id, geometry, properties, relationships, dimensions):
    super().__init__(id, self.type, geometry, properties, relationships, dimensions)

class PointAnnotation(SpecificLabelAnnotation):
  type: AnnotationType = AnnotationType.POINT
  @property
  def points(self) -> npt.NDArray[np.float32]:
    return self.geometry

  def viewer(self):
    """View as point cloud."""
    import microviewer
    microviewer.objects([ self.points ])

class LineAnnotation(SpecificLabelAnnotation):
  type: AnnotationType = AnnotationType.LINE

class AxisAlignedBoundingBoxAnnotation(SpecificLabelAnnotation):
  type: AnnotationType = AnnotationType.AXIS_ALIGNED_BOUNDING_BOX
  
  def bbox(self, i:int) -> Bbox:
    return Bbox.from_list(*self.geometry[i,:])

  def bboxes(self) -> list[Bbox]:
    return [
      Bbox.from_list(self.geometry[i,:])
      for i in range(len(self.geometry))
    ]

  def viewer(self):
    """View as point cloud."""
    import microviewer
    microviewer.objects(self.bboxes())


class EllipsoidAnnotation(SpecificLabelAnnotation):
  type: AnnotationType = AnnotationType.ELLIPSOID
  @property
  def radii(self):
    return self.geometry[:,:,1]
  @property
  def centers(self):
    return self.geometry[:,:,0]

class PolyLineAnnotation(SpecificLabelAnnotation):
  type: AnnotationType = AnnotationType.POLYLINE

ANNOTATION_CLASS = {
  AnnotationType.POINT: PointAnnotation,
  AnnotationType.LINE: LineAnnotation,
  AnnotationType.ELLIPSOID: EllipsoidAnnotation,
  AnnotationType.AXIS_ALIGNED_BOUNDING_BOX: AxisAlignedBoundingBoxAnnotation,
  AnnotationType.POLYLINE: PolyLineAnnotation,
}

def get_annotation_class(type:AnnotationType):
  return ANNOTATION_CLASS[type]

@dataclass
class MultiLabelAnnotation:
  type: AnnotationType
  geometry: npt.NDArray[np.float32]
  ids: npt.NDArray[np.uint64]
  properties: dict[str, np.ndarray]
  properties_enum: Optional[dict[str, dict[int,str]]]
  dimensions: Optional[list[str]]

  def __len__(self) -> int:
    return len(self.geometry)

  def pandas(self):
    import pandas as pd
    data = { "ID": self.ids }
    data.update(self.properties)

    if len(self.geometry.shape) > 2 and self.geometry.shape[2] > 1:
      for j in range(self.geometry.shape[2]):
        for i in range(self.geometry.shape[1]):
          axis = f"axis_{j}_{i}"
          if i < len(self.dimensions):
            axis = f"axis{j}_{self.dimensions[i]}"
          data[axis] = self.geometry[:,i,j]
    else:
      for i in range(self.geometry.shape[1]):
        axis = f"axis_{i}"
        if i < len(self.dimensions):
          axis = self.dimensions[i]
        data[axis] = self.geometry[:,i]
    
    df = pd.DataFrame(data)

    if isinstance(self.properties_enum, dict):
      for name, enum_dict in self.properties_enum.items():
        df[name] = df[name].map(enum_dict).astype('category')

    df.set_index("ID", inplace=True)
    return df

  def split_by_id(self) -> dict[int,LabelAnnotation]:
    all_labels = np.unique(self.ids)

    AnnotationClass = get_annotation_class(self.type)
    dims = list(self.dimensions.keys())

    out = {}
    for label in all_labels:
      mask = self.ids == label
      properties = { 
        name: arr[mask]
        for name, arr in self.properties.items()
      }
      label = int(label)
      out[label] = AnnotationClass(
        label,
        self.geometry[mask],
        properties,
        relationships={},
        properties_enum=self.properties_enum,
        dimensions=dims,
      )
    return out

  def crop(self, bbox:Bbox) -> "MultiLabelAnnotation":
    mask = _crop_mask(self.type, self.geometry, bbox)
    return MultiLabelAnnotation(
      type=self.type,
      geometry=self.geometry[mask],
      ids=self.ids[mask],
      properties={
        k: v[mask]
        for k,v in self.properties.items()
      },
      properties_enum=self.properties_enum,
      dimensions=self.dimensions,
    )

  def viewer(self):
    if self.type != AnnotationType.POINT:
      raise ValueError(f"Type {self.type} not supported.")

    import microviewer
    microviewer.objects([ self.geometry ])

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

  def has_id_index(self) -> bool:
    return self.info.get("by_id", None) is not None

  def has_spatial_index(self) -> bool:
    return self.info.get("spatial", None) is not None

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
      }],
      # e.g. [ { "id": , "key": ..., "sharding": ...  } ]
      "relationships": [],
      # e.g. [ { "key": ..., "sharding": ...  } ]
      "by_id": [], 
      # e.g. [ { "key", "sharding", "grid_shape", "chunk_size", "limit" } ] 
    }

  @property
  def dimensions(self) -> list[list]:
    return OrderedDict(self.info["dimensions"])

  @property
  def ndim(self) -> int:
    return len(self.info["dimensions"])

  @property
  def rank(self) -> int:
    """Alias for ndim."""
    return self.ndim

  @property
  def properties(self) -> dict:
    return self.info.get("properties", [])

  @property
  def properties_enum(self) -> dict[str, dict[int, str]]:
    enums = {}
    for p in self.properties:
      if "enum_labels" in p:
        enums[p['id']] = {
          k: v for k, v in zip(p["enum_values"], p["enum_labels"])
        }
    
    return enums

  @property
  def properties_summary(self) -> dict[str, dict[int, str]]:
    enums = {}
    for p in self.properties:
      if "enum_labels" in p:
        enums[p['id']] = {
          k: v for k, v in zip(p["enum_values"], p["enum_labels"])
        }
      elif "enum_values" in p:
        enums[p['id']] = np.asarray(p["enum_values"])
      else:
        enums[p['id']] = p["type"]
    
    return enums

  @property
  def relationships(self) -> dict[str, dict[str, Any]]:
    if "relationships" not in self.info.keys():
      raise ValueError("No relationships found in the info file.")
    return { r["id"]: r for r in self.info["relationships"] }

  @property
  def property_names(self) -> list[str]:
    """Get the properties of the annotations.

    Returns:
      list: A list of property names.
    """
    if "properties" not in self.info.keys():
      raise ValueError("No properties found in the info file.")
    return [ p["id"] for p in self.info["properties"] ]

  def annotation_dtype(self, binary:bytes) -> np.dtype:
    prop_dtypes = _get_dtype_for_properties(self.properties)

    # Derived from Neuroglancer Python code
    if self.annotation_type == AnnotationType.POLYLINE:
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

    geometry_dtype = [('_pt1', 'f4', self.ndim)]

    if self.annotation_type in two_point_types:
      geometry_dtype += [('_pt2', 'f4', self.ndim)]

    return geometry_dtype + prop_dtypes

  @property
  def bounds(self) -> Bbox:
    # assumes all dimensions are 1 unit and same
    unit = next(iter(self.dimensions.values()))[1] 
    return Bbox(self.info["lower_bound"], self.info["upper_bound"], unit=unit)

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
