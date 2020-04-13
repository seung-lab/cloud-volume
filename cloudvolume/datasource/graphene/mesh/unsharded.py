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

from ....lib import red, toiter, Bbox, Vec, jsonify
from ....mesh import Mesh
from .... import paths
from ....storage import Storage, GreenStorage

from ...precomputed.mesh import UnshardedLegacyPrecomputedMeshSource, PrecomputedMeshMetadata


class GrapheneUnshardedMeshSource(UnshardedLegacyPrecomputedMeshSource):

  def _get_fragment_filenames(self, seg_id, lod=0, level=2, bbox=None):
    # TODO: add lod to endpoint
    query_d = {
      'verify': True,
    }

    if bbox is not None:
      bbox = Bbox.create(bbox)
      query_d['bounds'] = bbox.to_filename()

    url = "%s/%s:%s" % (self.meta.meta.manifest_endpoint, seg_id, lod)
    
    if level is not None:
      res = requests.get(
        url,
        data=jsonify({ "start_layer": level }),
        params=query_d,
        headers=self.meta.meta.auth_header
      )
    else:
      res = requests.get(url, params=query_d, headers=self.meta.meta.auth_header)

    res.raise_for_status()

    return json.loads(res.content.decode('utf8'))["fragments"]

  def download_segid(self, seg_id, bounding_box):
    level = self.meta.meta.decode_layer_id(seg_id)
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
    return mesh, is_draco

  def get(
      self, segids, 
      remove_duplicate_vertices=False, 
      fuse=False, bounding_box=None
    ):
    """
    Merge fragments derived from these segids into a single vertex and face list.

    Why merge multiple segids into one mesh? For example, if you have a set of
    segids that belong to the same neuron.

    segid: (iterable or int) segids to render into a single mesh

    Optional:
      remove_duplicate_vertices: bool, fuse exactly matching vertices within a chunk
      fuse: bool, merge all downloaded meshes into a single mesh
      bounding_box: Bbox, bounding box to restrict mesh download to
    
    Returns: Mesh object if fused, else { segid: Mesh, ... }
    """
    import DracoPy

    segids = list(set([ int(segid) for segid in toiter(segids) ]))

    meta = self.meta.meta

    meshes = []
    for seg_id in tqdm(segids, disable=(not self.config.progress), desc="Downloading Meshes"):
      level = meta.decode_layer_id(seg_id)
      mesh, is_draco = self.download_segid(seg_id, bounding_box)
      resolution = meta.resolution(self.config.mip)
      if meta.chunks_start_at_voxel_offset:
        offset = meta.voxel_offset(self.config.mip)
      else:
        offset = Vec(0,0,0)

      if remove_duplicate_vertices:
        mesh = mesh.consolidate()
      elif is_draco:
        if level == 2:
          # Deduplicate at quantized lvl2 chunk borders
          draco_grid_size = meta.get_draco_grid_size(level)
          mesh = mesh.deduplicate_chunk_boundaries(
            meta.mesh_chunk_size * resolution,
            offset=offset * resolution,
            is_draco=True,
            draco_grid_size=draco_grid_size,
          )
        else:
          # TODO: cyclic draco quantization to properly
          # stitch and deduplicate draco meshes at variable
          # levels (see github issue #299)
          print('Warning: deduplication not currently supported for this layer\'s variable layered draco meshes')
      else:
        mesh = mesh.deduplicate_chunk_boundaries(
            meta.mesh_chunk_size * resolution,
            offset=offset * resolution,
            is_draco=False,
          )
      
      meshes.append(mesh)

    if not fuse:
      return { m.segid: m for m in meshes }

    return Mesh.concatenate(*meshes).consolidate()

