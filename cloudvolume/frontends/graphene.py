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
from ..lib import Bbox, Vec, toiter
from ..storage import SimpleStorage, Storage, reset_connection_pools
from ..volumecutout import VolumeCutout

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
  
  def download_point(self, pt, size=256, mip=None, parallel=None, **kwargs):
    """
    Download to the right of point given in mip 0 coords.
    Useful for quickly visualizing a neuroglancer coordinate
    at an arbitary mip level.

    pt: (x,y,z)
    size: int or (sx,sy,sz)
    mip: int representing resolution level
    parallel: number of processes to launch (0 means all cores)

    Also accepts the arguments for download such as root_ids and mask_base.

    Return: image
    """
    if isinstance(size, int) or isinstance(size, float):
      size = Vec(size, size, size)
    else:
      size = Vec(*size)

    if mip is None:
      mip = self.mip

    mip = self.meta.to_mip(mip)
    size2 = size // 2

    pt = self.point_to_mip(pt, mip=0, to_mip=mip)
    bbox = Bbox(pt - size2, pt + size2).astype(np.int64)

    if parallel is None:
      parallel = self.parallel

    return self.download(bbox, mip, parallel=parallel, **kwargs)

  def download(
    self, bbox, mip=None, 
    parallel=None, root_ids=None,
    mask_base=False
  ):
    """
    Graphene slicing is distinguished from Precomputed in two ways:

    Expects inputs of the form:

    img = cv[ slice, slice, slice, [ root_ids ] ]

    e.g. 

    img = cv[:,:,:]
    img = cv[:,:,:, [ 720575940615625227 ]]

    The final position of the input slices may optionally be
      a list of root ids that describe how to remap the base
      watershed coloring.

    If mask_base is set, segids that are not remapped are blacked out.

    Returns: img
    """
    bbox = Bbox.create(
      bbox, context=self.bounds, 
      bounded=self.bounded, 
      autocrop=self.autocrop
    )
    
    if bbox.subvoxel():
      raise exceptions.EmptyRequestException("Requested {} is smaller than a voxel.".format(bbox))

    if mip is None:
      mip = self.mip

    mip0_bbox = self.bbox_to_mip(bbox, mip=mip, to_mip=0)
    # Only ever necessary to make requests within the bounding box
    # to the server. We can fill black in other situations.
    mip0_bbox = bbox.intersection(self.meta.bounds(0), mip0_bbox)

    if root_ids is None and mask_base:
      return np.zeros( bbox.size(), dtype=self.dtype )

    img = super(CloudVolumeGraphene, self).download(bbox, mip=mip, parallel=parallel)

    if root_ids is None:
      if mask_base:
        img[:] = 0
      return img

    root_ids = list(toiter(root_ids))

    remapping = {}
    for root_id in root_ids:
      leaves = self.get_leaves(root_id, mip0_bbox, 0)
      remapping.update({ leaf: root_id for leaf in leaves })
    
    img = fastremap.remap(img, remapping, preserve_missing_labels=True, in_place=True)
    if mask_base:
      img = fastremap.mask_except(img, root_ids, in_place=True)

    return VolumeCutout.from_volume(
      self.meta, mip, img, bbox 
    )

  def __getitem__(self, slices):
    return self.download(
      slices, mip=self.mip,
      mask_base=False,
      parallel=self.parallel, 
    )

  def get_root(self, segid):
    """
    Get the root id of this label.
    """
    url = posixpath.join(self.meta.server_url, "graph/root")
    response = requests.post(url, json=[ int(segid) ])
    response.raise_for_status()
    return np.frombuffer(response.content, dtype=np.uint64)[0]

  def get_leaves(self, root_id, bbox, mip):
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
