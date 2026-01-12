from typing import Optional, Union, Iterable
from dataclasses import dataclass

import os
import posixpath

from cloudfiles import CloudFiles
import numpy as np
import numpy.typing as npt

from .metadata import PrecomputedAnnotationMetadata

from ....cacheservice import CacheService
from ....paths import strict_extract
from ....cloudvolume import SharedConfiguration

from ....types import SecretsType
from ....lib import Bbox, BboxLikeType, toiter
from ....paths import strict_extract, to_https_protocol, ascloudpath
from ..common import compressed_morton_code
from ..sharding import ShardReader, ShardingSpecification

from .metadata import (
  PrecomputedAnnotationMetadata, 
  AnnotationType,
  MultiLabelAnnotation,
  LabelAnnotation,
)
from .reader import PrecomputedAnnotationReader

class PrecomputedAnnotationSource:
  def __init__(
    self,
    cloudpath:str,
    cache:Optional[str] = None,
    cache_locking:bool = True,
    info:Optional[dict] = None,
    progress:bool = False,
    readonly:bool = False,
    secrets:SecretsType = None,
    use_https:bool = False,
    mip:int = -1,
  ):
    from .. import get_cache_path

    path = strict_extract(cloudpath)
    if use_https:
      path = to_https_protocol(path)
      cloudpath = ascloudpath(path)
    
    config = SharedConfiguration(
      cdn_cache=False,
      compress=False, 
      compress_level=5,
      green=False,
      mip=mip,
      parallel=1,
      progress=progress,
      secrets=secrets,
      spatial_index_db=None,
      cache_locking=cache_locking,
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
    self.reader = PrecomputedAnnotationReader(self.meta, self.cache, self.config)

  def ids(self) -> npt.NDArray[np.uint64]:
    """Get all annotation IDs."""
    return self.reader.ids()

  def get_by_bbox(self, query:BboxLikeType, mip:Optional[int] = None) -> MultiLabelAnnotation:
    """Get all annotations within a bounding box."""
    if mip is None:
      mip = self.config.mip
    return self.reader.get_by_bbox(query, mip=mip)

  def get_by_id(self, query:list[int]) -> LabelAnnotation:
    """Get all annotations by ID."""
    return self.reader.get_by_id(query)

  def get_by_relationship(self, relationship:str, labels:Union[int, Iterable[int]]) -> MultiLabelAnnotation:
    """Get annotations by relationship."""
    return self.reader.get_by_relationship(relationship, labels)

  def get_all(self) -> MultiLabelAnnotation:
    """Get all annotations using the most efficient method available."""
    return self.reader.get_all()

  def summary(self) -> dict:
    return {
      "type": self.reader.meta.annotation_type,
      "bounds": self.reader.meta.bounds,
      "path": self.reader.meta.cloudpath,
      "by_id": self.reader.meta.info["by_id"] is not None,
      "spatial_query": self.reader.meta.info["spatial"] is not None,
      "relationships": list(self.reader.meta.relationships.keys()),
      "properties": self.reader.meta.properties_enum,
    }

  def __getitem__(self, slcs):
    return self.get_by_bbox(slcs, mip=self.config.mip)


