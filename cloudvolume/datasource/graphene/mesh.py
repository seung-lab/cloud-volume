import six

from collections import defaultdict
import itertools
import json
import os
import posixpath
import re
import requests

import numpy as np
from tqdm import tqdm

from ...lib import red, toiter, Bbox
from ...mesh import Mesh
from ... import paths
from ...storage import Storage, GreenStorage

from ..precomputed.mesh import PrecomputedMeshSource

class GrapheneMeshSource(PrecomputedMeshSource):
  def _get_fragment_filenames(self, seg_id, lod=0, level=2, bbox=None):
    # TODO: add lod to endpoint
    query_d = {
      'verify': True,
    }

    if bbox is not None:
      bbox = Bbox.create(bbox)
      query_d['bounds'] = bbox.to_filename()

    url = "%s/%s:%s" % (self.meta.manifest_endpoint, seg_id, lod)
    
    if level is not None:
      res = requests.get(
        url,
        data=json.dumps({ "start_layer": level }),
        params=query_d,
        headers=self.meta.auth_header
      )
    else:
      res = requests.get(url, params=query_d, headers=self.meta.auth_header)

    res.raise_for_status()

    return json.loads(res.content)["fragments"]

  def get(
      self, segids, 
      remove_duplicate_vertices=False, 
      fuse=False, level=2, 
      bounding_box=None
    ):
    """
    Merge fragments derived from these segids into a single vertex and face list.

    Why merge multiple segids into one mesh? For example, if you have a set of
    segids that belong to the same neuron.

    segid: (iterable or int) segids to render into a single mesh

    Optional:
      remove_duplicate_vertices: bool, fuse exactly matching vertices within a chunk
      fuse: bool, merge all downloaded meshes into a single mesh
      level: int, level of mesh to return. None to return highest available (default 2) 
      bounding_box: Bbox, bounding box to restrict mesh download to
    
    Returns: Mesh object if fused, else { segid: Mesh, ... }
    """
    import DracoPy

    segids = list(set([ int(segid) for segid in toiter(segids) ]))

    meshes = []
    for seg_id in tqdm(segids, disable=(not self.config.progress), desc="Downloading Meshes"):
      fragment_filenames = self._get_fragment_filenames(
        seg_id, level=level, bbox=bounding_box
      )
      fragments = self._get_mesh_fragments(fragment_filenames)
      fragments = sorted(fragments, key=lambda frag: frag[0])  # make decoding deterministic

      fragiter = tqdm(fragments, disable=(not self.config.progress), desc="Decoding Mesh Buffer")
      is_draco = False
      for i, (filename, frag) in enumerate(fragiter):
        mesh = None
        
        if frag is not None:
          try:
            # Easier to ask forgiveness than permission
            mesh = Mesh.from_draco(frag)
            is_draco = True
          except DracoPy.FileTypeException:
            mesh = Mesh.from_precomputed(frag)
            
        fragments[i] = mesh
      
      fragments = [ f for f in fragments if f is not None ] 
      if len(fragments) == 0:
        raise IndexError('No mesh fragments found for segment {}'.format(seg_id))

      mesh = Mesh.concatenate(*fragments)
      mesh.segid = seg_id
      resolution = self.meta.resolution(self.config.mip)
      mesh = mesh.deduplicate_chunk_boundaries(self.meta.mesh_chunk_size * resolution, is_draco=is_draco)
      meshes.append(mesh.consolidate())

    if not fuse:
      return { m.segid: m for m in meshes }

    return Mesh.concatenate(*meshes).consolidate()

