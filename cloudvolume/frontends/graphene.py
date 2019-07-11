import json
import os
import posixpath
import re
import requests
import sys

import fastremap
import numpy as np

from .. import exceptions
from ..cacheservice import CacheService
from ..lib import Bbox, toiter
from ..storage import SimpleStorage, Storage, reset_connection_pools

from .precomputed import CloudVolumePrecomputed

def warn(text):
  print(colorize('yellow', text))

class CloudVolumeGraphene(CloudVolumePrecomputed):

  @property
  def manifest_endpoint(self):
    return self.meta.manifest_endpoint

  @property
  def graph_chunk_size(self):
    return self.meta.graph_chunk_size 

  @property
  def mesh_chunk_size(self):
    # TODO: add this as new parameter to the info as it can be different from the chunkedgraph chunksize
    return self.meta.mesh_chunk_size
  
  def download(
    self, bbox, mip=None, 
    parallel=None, root_ids=None,
    mask_base=True
  ):
    """
    Graphene slicing is distinguished from Precomputed in two ways:

    1) All input bounding boxes are in mip 0 coordinates.
    2) The final position of the input slices may optionally be
      a list of root ids that describe how to remap the base
      watershed coloring.

    If mask_base is set, segids that are not remapped

    """
    bbox = Bbox.create(bbox, context=self.meta.bounds(0), bounded=self.bounded)

    if bbox.subvoxel():
      raise exceptions.EmptyRequestException("Requested {} is smaller than a voxel.".format(bbox))

    if mip is None:
      mip = self.mip

    bbox = self.bbox_to_mip(bbox, mip=0, to_mip=mip)

    if root_ids is None and mask_base:
      return np.zeros( bbox.size(), dtype=self.dtype )

    img = super(CloudVolumeGraphene, self).download(bbox, mip=mip, parallel=parallel)

    if root_ids is None:
      if mask_base:
        img[:] = 0
      return img

    root_ids = toiter(root_ids)

    remapping = {}
    for root_id in root_ids:
      leaves = self._get_leaves(root_id, bbox, mip)
      remapping.update({ leaf: root_id for leaf in leaves })
    
    img = fastremap.remap(img, remapping, preserve_missing_labels=True, in_place=True)
    if mask_base:
      img = fastremap.mask_except(img, root_ids, in_place=True)

    return img

  def __getitem__(self, slices):
    """
    Graphene slicing is distinguished from Precomputed in two ways:

    1) All input bounding boxes are in mip 0 coordinates.
    2) The final position of the input slices may optionally be
      a list of root ids that describe how to remap the base
      watershed coloring.
    """
    try:
      iter(slices[-1])
      root_ids = list(slices.pop())
      return self.download(
        slices, mip=self.mip, 
        parallel=self.parallel, root_ids=root_ids
      )
    except TypeError: 
      # The end of the array was not iterable, 
      # and thus not the root ids.
      return self.download(
        slices, mip=self.mip, 
        parallel=self.parallel, root_ids=None
      )

  def _get_leaves(self, root_id, bbox, mip):
    """
    get the supervoxels for this root_id

    params
    ------
    root_id: uint64 root id to find supervoxels for
    bbox: cloudvolume.lib.Bbox 3d bounding box for segmentation
    """
    root_id = int(root_id)
    url = posixpath.join(self.meta.server_url, "segment", str(root_id), "leaves")
    bbox = Bbox.create(bbox, context=self.meta.bounds(mip), bounded=self.bounded)

    response = requests.post(url, json=[ root_id ], params={
      'bounds': bbox.to_filename(),
    })
    response.raise_for_status()

    return np.frombuffer(response.content, dtype=np.uint64)
