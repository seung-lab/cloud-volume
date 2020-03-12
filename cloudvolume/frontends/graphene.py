import json
import os
import posixpath
import re
import requests
import sys

import fastremap
import numpy as np

from .. import compression
from .. import exceptions
from ..cacheservice import CacheService
from ..lib import Bbox, Vec, toiter
from ..storage import SimpleStorage, Storage, reset_connection_pools
from ..volumecutout import VolumeCutout
from ..datasource.graphene.metadata import GrapheneApiVersion

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

    if all(size == 1):
      bbox = Bbox(pt, pt + 1).astype(np.int64)
    else:
      bbox = Bbox(pt - size2, pt + size2).astype(np.int64)

    if parallel is None:
      parallel = self.parallel

    return self.download(bbox, mip, parallel=parallel, **kwargs)

  def download(
    self, bbox, mip=None, 
    parallel=None, segids=None,
    preserve_zeros=False,
    agglomerate=False, timestamp=None
  ):
    """
    Downloads base segmentation and optionally agglomerates
    labels based on information in the graph server.

    bbox: specifies cutout to fetch
    mip: which resolution level to get (default self.mip)
    parallel: what parallel level to use (default self.parallel)

    agglomerate: if true, remap all watershed ids in the volume
      and return a flat segmentation.

    if agglomerate is false, these other options come into play:

    segids: agglomerate the leaves of these segids from the graph 
      server and label them with the given segid.
    preserve_zeros: If segids is not None:
      False: mask other segids with zero
      True: mask other segids with the largest integer value
        contained by the image data type and leave zero as is.

    Returns: img as a VolumeCutout
    """
    if type(bbox) is Vec:
      bbox = Bbox(bbox, bbox+1)
    
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

    if agglomerate:
      img = self.agglomerate_cutout(img, timestamp=timestamp)
      return VolumeCutout.from_volume(self.meta, mip, img, bbox)

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
  
  def agglomerate_cutout(self, img, timestamp=None):
    """Remap a graphene volume to its latest root ids. This creates a flat segmentation."""
    labels = fastremap.unique(img)
    roots = self.get_roots(labels, timestamp=timestamp, binary=True)
    mapping = { segid: root for segid, root in zip(labels, roots) }
    return fastremap.remap(img, mapping, preserve_missing_labels=True, in_place=True)

  def __getitem__(self, slices):
    return self.download(
      slices, mip=self.mip,
      preserve_zeros=True,
      parallel=self.parallel, 
    )

  def get_root(self, segid, *args, **kwargs):
    """Deprecated. Get a single root id for a single segid."""
    return get_roots(segid, *args, **kwargs)[0]

  def get_roots(self, segids, timestamp=None, binary=False):
    """
    Get the root ids for these labels.
    """
    segids = toiter(segids)
    if isinstance(segids, np.ndarray):
      segids = segids.tolist()

    if self.meta.supports_api('v1'):
      roots = self._get_roots_v1(segids, timestamp, binary)
    elif self.meta.supports_api('1.0'):
      roots = self._get_roots_legacy(segids, timestamp)
    else:
      raise exceptions.UnsupportedGrapheneAPIVersionError(
        "{} is not a supported API version. Supported versions: ".format(self.meta.api_version) \
        + ", ".join([ str(_) for _ in self.meta.supported_api_versions ])
      )

    return np.array(roots, dtype=self.meta.dtype)

  def _get_roots_v1(self, segids, timestamp, binary=False):
    args = {}
    if timestamp is not None:
      args['timestamp'] = timestamp

    headers = {}
    headers.update(self.meta.auth_header)

    gzip_condition = len(segids) * 8 > 1e6

    params = {}
    if gzip_condition:
      headers['Content-Encoding'] = 'gzip'
      headers['Accept-Encoding'] = 'gzip;q=1, identity;q=0.1'
      params['gzip'] = 1
    else:
      headers['Accept-Encoding'] = 'identity'

    version = GrapheneApiVersion('v1')
    path = version.path(self.meta.server_path)
    url = posixpath.join(self.meta.base_path, path, "roots")
    args['node_ids'] = segids

    if binary:
      params['as_binary'] = 'root_ids'

    data = json.dumps(args)
    if gzip_condition:
      data = compression.compress(data.encode('utf8'), method='gzip')

    response = requests.post(url, data=data, headers=headers, params=params)
    response.raise_for_status()

    if binary:
      return np.frombuffer(response.content, dtype=np.uint64)
    else:
      return json.loads(response.content)['root_ids']

  def _get_roots_legacy(self, segids, timestamp):
    args = {}
    if timestamp is not None:
      args['timestamp'] = timestamp

    version = GrapheneApiVersion('1.0')
    path = version.path(self.meta.server_path)
    roots = []
    for segid in segids:
      url = posixpath.join(self.meta.base_path, path, "graph/{:d}/root".format(int(segid)))
      response = requests.get(url, json=args, headers=self.meta.auth_header)
      response.raise_for_status()
      root = np.frombuffer(response.content, dtype=np.uint64)[0]
      roots.append(root)
    return roots

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
