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
from ....lib import Bbox, BboxLikeType, toiter
from ....paths import strict_extract, to_https_protocol, ascloudpath
from ..common import compressed_morton_code
from ..sharding import ShardReader, ShardingSpecification

from .metadata import PrecomputedAnnotationMetadata, AnnotationType
from .reader import PrecomputedAnnotationReader
# from .writer import AnnotationWriter

# sharded example 
# https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B6.4e-8%2C%22m%22%5D%2C%22y%22:%5B6.4e-8%2C%22m%22%5D%2C%22z%22:%5B6.6e-8%2C%22m%22%5D%7D%2C%22position%22:%5B34724.5%2C23270.5%2C584.5%5D%2C%22crossSectionScale%22:1%2C%22projectionOrientation%22:%5B-0.09734609723091125%2C-0.26029738783836365%2C-0.0020852696616202593%2C0.9606063961982727%5D%2C%22projectionScale%22:20037.381619627573%2C%22layers%22:%5B%7B%22type%22:%22segmentation%22%2C%22source%22:%22gs://h01-release/data/20210601/c2/subcompartments/%7Cneuroglancer-precomputed:%22%2C%22tab%22:%22source%22%2C%22segments%22:%5B%22%21103%22%5D%2C%22name%22:%22subcompartments%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22annotation%22%2C%22source%22:%22gs://h01-release/data/20210601/c2/subcompartments/annotations/%7Cneuroglancer-precomputed:%22%2C%22tab%22:%22source%22%2C%22name%22:%22new%20layer%22%7D%5D%2C%22selectedLayer%22:%7B%22visible%22:true%2C%22layer%22:%22new%20layer%22%7D%2C%22layout%22:%22xy-3d%22%7D

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
    self.reader = PrecomputedAnnotationReader(self.meta, self.cache, self.config)
    # self.writer = AnnotationWriter(cloudpath, meta, cache, config)

  def get_by_bbox(self, query:BboxLikeType, mip:int = 0):
    return self.reader.get_by_bbox(query, mip=mip)

  def get_by_ids(self, query:list[int]):
    return self.reader.get_by_ids(query)

  def get(self, query, mip:int = 0):
    if isinstance(query, Bbox):
      return self.get_by_bbox(query, mip=mip)
    elif isinstance(query, (int, np.integer)):
      return self.reader.get_by_ids(query)
    elif isinstance(query, list):
      return self.reader.get_by_ids(query)
    elif isinstance(query, tuple):
      if len(query) == 0:
        return []
      elif isinstance(query[0], slice):
        return self.get_by_bbox(query, mip=mip)
      else:
        return self.get_by_ids(query)
    else:
      raise ValueError(f"{query} is not a valid query type.")



