from typing import Union, Optional

from ....lib import Bbox, BboxLikeType
from ..common import compressed_morton_code
from ..sharding import ShardReader

from cloudfiles import CloudFiles

import numpy as np

# sharded example 
# https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B6.4e-8%2C%22m%22%5D%2C%22y%22:%5B6.4e-8%2C%22m%22%5D%2C%22z%22:%5B6.6e-8%2C%22m%22%5D%7D%2C%22position%22:%5B34724.5%2C23270.5%2C584.5%5D%2C%22crossSectionScale%22:1%2C%22projectionOrientation%22:%5B-0.09734609723091125%2C-0.26029738783836365%2C-0.0020852696616202593%2C0.9606063961982727%5D%2C%22projectionScale%22:20037.381619627573%2C%22layers%22:%5B%7B%22type%22:%22segmentation%22%2C%22source%22:%22gs://h01-release/data/20210601/c2/subcompartments/%7Cneuroglancer-precomputed:%22%2C%22tab%22:%22source%22%2C%22segments%22:%5B%22%21103%22%5D%2C%22name%22:%22subcompartments%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22annotation%22%2C%22source%22:%22gs://h01-release/data/20210601/c2/subcompartments/annotations/%7Cneuroglancer-precomputed:%22%2C%22tab%22:%22source%22%2C%22name%22:%22new%20layer%22%7D%5D%2C%22selectedLayer%22:%7B%22visible%22:true%2C%22layer%22:%22new%20layer%22%7D%2C%22layout%22:%22xy-3d%22%7D

class PrecomputedAnnotationSource:
  def __init__(self, meta, cache, config, readonly=False):
    self.meta = meta
    self.cache = cache
    self.config = config
    self.readonly = bool(readonly)

    spec = ShardingSpecification.from_dict(self.meta.info['sharding'])
    self.reader = ShardReader(meta.cloudpath, cache, spec)

  def get_by_id(self, segids:Union[int, list[int]]) -> dict[int, np.ndarray]:
    
    if self.meta.is_id_index_sharded():
      spec = self.meta.info["by_id"]["sharding"]
      reader = ShardReader(self.)

    cf = CloudFiles(self.meta.annotations_cloudpath)
    annotations = cf.get([ f"{segid}" for segid in segids ])

    N = len(self.meta.info["dimensions"])

    pts = {}
    for segid, binary in annotations:
      pts[segid] = np.frombuffer(binary, dtype=np.float32).reshape((N, len(binary) // N) )

    return pts

  def get_by_bbox(self, bbox:BboxLikeType, mip:int = 0) -> dict[int, np.ndarray]:
    spatial = self.meta.info["spatial"]
    key = spatial["key"]

    cf = CloudFiles(self.meta.annotations_cloudpath)
    spatial_path = cf.join(self.meta.annotations_cloudpath, key)
    cf = CloudFiles(spatial_path)

    bbox = Bbox.create(bbox, self.meta.bounds())
    bbox = bbox.expand_to_chunk_size(
      self.meta.chunk_size(mip),
      offset=self.meta.bounds().minpt,
    )
    bbox = Bbox.clamp(bbox, self.meta.bounds())
    bbox /= self.meta.chunk_size(mip)

    grid = np.mgrid[bbox.to_slices()]

    codes = compressed_morton_code(grid, self.meta.grid_shape(mip))

    files = self.reader.get(codes)



