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
  
  def download_point(
    self, pt, size=256, 
    mip=None, parallel=None, 
    coord_resolution=None,
    **kwargs
  ):
    """
    Download to the right of point given in mip 0 coords.
    Useful for quickly visualizing a neuroglancer coordinate
    at an arbitary mip level.

    pt: (x,y,z)
    size: int or (sx,sy,sz)
    mip: int representing resolution level
    parallel: number of processes to launch (0 means all cores)
    coord_resolution: (rx,ry,rz) the coordinate resolution of the input point.
      Sometimes Neuroglancer is working in the resolution of another
      higher res layer and this can help correct that.

    Also accepts the arguments for download such as segids and preserve_zeros.

    Return: image as VolumeCutout(ndarray)
    """
    if isinstance(size, int) or isinstance(size, float):
      size = Vec(size, size, size)
    else:
      size = Vec(*size)

    if mip is None:
      mip = self.mip

    mip = self.meta.to_mip(mip)
    size2 = size // 2

    if coord_resolution is not None:
      factor = self.meta.resolution(0) / Vec(*coord_resolution)
      pt = Vec(*pt) / factor

    pt = self.point_to_mip(pt, mip=0, to_mip=mip)
    bbox = Bbox(pt - size2, pt + size2).astype(np.int64)

    if parallel is None:
      parallel = self.parallel

    return self.download(bbox, mip, parallel=parallel, **kwargs)

  def download(
    self, bbox, mip=None, 
    parallel=None, segids=None,
    preserve_zeros=False
  ):
    """
    Downloads base segmentation and optionally agglomerates
    labels based on information in the graph server.

    bbox: specifies cutout to fetch
    mip: which resolution level to get (default self.mip)
    parallel: what parallel level to use (default self.parallel)

    segids: agglomerate the leaves of these segids from the graph 
      server and label them with the given segid.
    preserve_zeros: If segids is not None:
      False: mask other segids with zero
      True: mask other segids with the largest integer value
        contained by the image data type and leave zero as is.

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

    img = super(CloudVolumeGraphene, self).download(bbox, mip=mip, parallel=parallel)

    if segids is None:
      return img

    segids = list(toiter(segids))

    remapping = {}
    for segid in segids:
      leaves = self.get_leaves(segid, mip0_bbox, 0)
      remapping.update({ leaf: segid for leaf in leaves })
    
    img = fastremap.remap(img, remapping, preserve_missing_labels=True, in_place=True)

    mask_value = 0
    if preserve_zeros:
      mask_value = np.inf
      if np.issubdtype(self.dtype, np.integer):
        mask_value = np.iinfo(self.dtype).max

      segids.append(0)

    img = fastremap.mask_except(img, segids, in_place=True, value=mask_value)

    return VolumeCutout.from_volume(
      self.meta, mip, img, bbox 
    )

  def __getitem__(self, slices):
    return self.download(
      slices, mip=self.mip,
      preserve_zeros=True,
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
    }, headers=self.meta.auth_header)
    response.raise_for_status()

    return np.frombuffer(response.content, dtype=np.uint64)
