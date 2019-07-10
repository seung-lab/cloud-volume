import json
import os
import re
import requests
import sys

import fastremap
import numpy as np

from .precomputed import CloudVolumePrecomputed

from ..cacheservice import CacheService
from ..lib import Bbox
from ..storage import SimpleStorage, Storage, reset_connection_pools

def warn(text):
  print(colorize('yellow', text))

class CloudVolumeGraphene(CloudVolumePrecomputed):
  @property
  def cloud_url(self):
    return self.cloudpath

  @property
  def info_endpoint(self):
    return "%s/info"%self.cloud_url

  @property
  def manifest_endpoint(self):
    return "%s/manifest"%self.cloud_url.replace('segmentation', 'meshing')

  @property
  def cloudpath(self):
    return self._info_dict["data_dir"]

  @property
  def graph_chunk_size(self):
    return self._info_dict["graph"]["chunk_size"]

  @property
  def mesh_chunk_size(self):
    #TODO: add this as new parameter to the info as it can be different from the chunkedgraph chunksize
    return self.graph_chunk_size

  @property
  def _storage(self):
    return self._cv._storage

  @property
  def dataset_name(self):
    return self.info_endpoint.split("/")[-1]

  @staticmethod
  def _convert_root_id_list(root_ids):
    if isinstance(root_ids, int):
      return [root_ids]
    if isinstance(root_ids, list):
      return np.array(root_ids, dtype=np.uint64)
    if isinstance(root_ids, (np.ndarray, np.generic)):
      return np.array(root_ids.ravel(), dtype=np.uint64)
    return root_ids
    
  def __getitem__(self, slices):
    assert(len(slices) == 4)
    seg_cutout = self._cv.__getitem__(slices[:-1])
    root_ids = slices[-1]
    root_ids = self._convert_root_id_list(root_ids)
    bbox, steps, channel_slice = self.__interpret_slices(slices[:-1])
    bbox = bbox * [2**self.mip, 2**self.mip, 1]
    print(bbox, self.bounds)
    bbox=bbox.intersection(self.bounds* [2**self.mip, 2**self.mip, 1], bbox)
    print(bbox)
    remap_d = {}
    for root_id in root_ids:
      leaves = self._get_leaves(root_id, bbox)
      remap_d.update(dict(zip(leaves, [root_id]*len(leaves))))
    seg_cutout = fastremap.remap(seg_cutout, remap_d, preserve_missing_labels=True)
      # seg_cutout[np.isin(sup_voxels_cutout, leaves)] = root_id
    return seg_cutout

  def _get_leaves(self, root_id, bbox):
    """
    get the supervoxels for this root_id

    params
    ------
    root_id: uint64 root id to find supervoxels for
    bbox: cloudvolume.lib.Bbox 3d bounding box for segmentation
    """
    url = "%s/segment/%d/leaves" % (self.cloud_url, root_id)
    bbox = Bbox.create(bbox, context=self.bounds, bounded=self.bounded)

    query_d = {
      'bounds': bbox.to_filename(),
    }

    response = requests.post(url, json=[int(root_id)], params=query_d)
    response.raise_for_status()

    return np.frombuffer(response.content, dtype=np.uint64)
