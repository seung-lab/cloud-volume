import six

from collections import defaultdict
import itertools
import json
import re
import requests
import os

import struct
import numpy as np
from tqdm import tqdm

from ...lib import red, toiter, Bbox
from ...mesh import Mesh
from ...storage import Storage

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

  def _get_mesh_fragments(self, filenames):
    mesh_dir = self.meta.info['mesh']
    paths = [ "%s/%s" % (mesh_dir, filename) for filename in filenames ]
    with Storage(self.meta.cloudpath, progress=self.config.progress) as stor:
      return stor.get_files(paths)

  def _produce_output(self, mdata, remove_duplicate_vertices_in_chunk):
    vertexct = np.zeros(len(mdata) + 1, np.uint32)
    vertexct[1:] = np.cumsum([x['num_vertices'] for x in mdata])
    vertices = np.concatenate([x['vertices'] for x in mdata])
    faces = np.concatenate([
      mesh['faces'] + vertexct[i] for i, mesh in enumerate(mdata)
    ])
    if len(faces.shape) == 1:
      faces = faces.reshape(-1, 3)
      
    if remove_duplicate_vertices_in_chunk:
      vertices, faces = np.unique(vertices[faces.reshape(-1)],
                    return_inverse=True, axis=0)
      faces = faces.reshape(-1,3).astype(np.uint32)
    else:
      vertices, faces = remove_duplicate_vertices_cross_chunks(
        vertices, faces, self.vol.mesh_chunk_size)
      
    return {
      'num_vertices': len(vertices),
      'vertices': vertices,
      'faces': faces,
    }

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
    meshdata = []
    fragments = tqdm(fragments, disable=(not self.config.progress), desc="Decoding Mesh Buffer")
    for frag in fragments:
      try:
        # Easier to ask forgiveness than permission
        mesh = decode_draco_mesh_buffer(frag["content"])
        # FIXME: Current cross chunk logic does not support Draco, so must check all vertices for duplicates
        remove_duplicate_vertices = True
      except DracoPy.FileTypeException:
        mesh = decode_mesh_buffer(frag["content"])
      meshdata.append(mesh)
    
    if len(meshdata) == 0:
      raise IndexError('No mesh fragments found for segment {}'.format(seg_id))

    return self._produce_output(meshdata, remove_duplicate_vertices)
