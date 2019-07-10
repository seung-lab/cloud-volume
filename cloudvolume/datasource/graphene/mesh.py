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
        params=query_d
      )
    else:
      res = requests.get(url, params=query_d)

    res.raise_for_status()

    return json.loads(res.content)["fragments"]

  def get(self, seg_id, remove_duplicate_vertices=False, level=2, bounding_box=None):
    """
    Merge fragments derived from these segids into a single vertex and face list.

    Why merge multiple segids into one mesh? For example, if you have a set of
    segids that belong to the same neuron.

    segid: (iterable or int) segids to render into a single mesh

    Optional:
      remove_duplicate_vertices: bool, fuse exactly matching vertices within a chunk
      level: int, level of mesh to return. None to return highest available (default 2) 
      bounding_box: Bbox, bounding box to restrict mesh download to
    Returns: {
      num_vertices: int,
      vertices: [ (x,y,z), ... ]  # floats
      faces: [ int, int, int, ... ] # int = vertex_index, 3 to a face
    }
    """
    import DracoPy

    if isinstance(seg_id, list) or isinstance(seg_id, np.ndarray):
      if len(seg_id) != 1:
        raise IndexError("GrapheneMeshSource.get accepts at most one segid. Got: " + str(seg_id))
      seg_id = seg_id[0]

    segid = int(seg_id)

    fragment_filenames = self._get_fragment_filenames(
      seg_id, level=level, bbox=bounding_box
    )
    fragments = self._get_mesh_fragments(fragment_filenames)
    # fragments = sorted(fragments, key=lambda frag: frag['filename'])  # make decoding deterministic

    # decode all the fragments
    fragiter = tqdm(fragments, disable=(not self.config.progress), desc="Decoding Mesh Buffer")
    for i, (filename, frag) in enumerate(fragiter):
      try:
        # Easier to ask forgiveness than permission
        mesh = Mesh.from_draco(frag)
      except DracoPy.FileTypeException:
        mesh = Mesh.from_precomputed(frag)
      fragments[i] = mesh
    
    if len(fragments) == 0:
      raise IndexError('No mesh fragments found for segment {}'.format(seg_id))

    mesh = Mesh.concatenate(*fragments)
    mesh.segid = seg_id

    return mesh.deduplicate_chunk_boundaries(self.meta.mesh_chunk_size)