from typing import Union, Optional

from collections import OrderedDict
from enum import StrEnum

from cloudfiles import CloudFiles

import numpy as np

from ....lib import Bbox
from ....paths import strict_extract, to_https_protocol

class AnnotationType(StrEnum):
  POINT = "POINT"
  LINE = "LINE"
  BBOX = "AXIS_ALIGNED_BOUNDING_BOX"
  ELLIPSOID = "ELLIPSOID"
  POLYLINE = "POLYLINE"


ANNOTATION_INFO_TYPE = "neuroglancer_annotations_v1"

class PrecomputedAnnotationMetadata:
  def __init__(
    self, 
    cloudpath:str,
    cache, 
    config:Optional["SharedConfiguration"] = None, 
    info:Optional[dict] = None, 
    readonly:bool = False,
    use_https:bool = False,
  ):

    path = strict_extract(cloudpath)
    if use_https:
      cloudpath = to_https_protocol(path)

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

  def fetch_info(self):
    return CloudFiles(self.cloudpath, secrets=self.config.secrets).get_json('info')

  def default_info(self):
    return {
      "@type": ANNOTATION_INFO_TYPE,
      "dimensions": [], # e.g. [[ 32.0, "nm" ], ... ]
      "lower_bound": [],
      "upper_bound": [],
      "annotation_type": AnnotationType.POINT, 
      "properties": {
        # "id": "class_label"
        # "type": "int32"
        # "description": ""
        # "enum_labels": [0,1,2,3,4,5000],
        # "enum_values": ["axon", "dendrite", "astrocyte", "soma", ...]
        # 
      },
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
    return self.info["properties"]

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


