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

from .lib import red, toiter
from .storage import Storage
from cloudvolume import meshservice

import DracoPy

SEGIDRE = re.compile(r'/(\d+):0.*?$')

def remove_duplicate_vertices_cross_chunks(verts, faces, chunk_size):
    # find all vertices that are exactly on chunk_size boundaries
    is_chunk_aligned = np.any(np.mod(verts, chunk_size) == 0, axis=1)
    # # uniq_vertices, uniq_faces, vert_face_counts = np.unique(vertices[faces],
    #                                                         return_inverse=True,
    #                                                         return_counts=True,
    #                                                         axis=0)
    # find all vertices that have exactly 2 duplicates
    unique_vertices, unique_inverse, counts = np.unique(verts,
                                                        return_inverse=True,
                                                        return_counts=True,
                                                        axis=0)
    only_double = np.where(counts == 2)[0]
    is_doubled = np.isin(unique_inverse, only_double)
    # this stores whether each vertex should be merged or not
    do_merge = np.array(is_doubled & is_chunk_aligned)

    # setup an artificial 4th coordinate for vertex positions
    # which will be unique in general, 
    # but then repeated for those that are merged
    new_vertices = np.hstack((verts, np.arange(verts.shape[0])[:, np.newaxis]))
    new_vertices[do_merge, 3] = -1
    fa = np.array(faces)
    n_faces = fa.shape[0]
    n_dim = fa.shape[1]

    # use unique to make the artificial vertex list unique and reindex faces
    vertices, newfaces = np.unique(new_vertices[faces.ravel(),:], return_inverse=True, axis=0)
    faces = newfaces.reshape((n_faces, n_dim))
    faces = faces.astype(np.uint32)

    return vertices[:,0:3], faces

def decode_mesh_buffer(fragment):
  num_vertices = struct.unpack("=I", fragment[0:4])[0]
  try:
    # count=-1 means all data in buffer
    vertices = np.frombuffer(fragment, dtype=np.float32, count=3*num_vertices, offset=4)
    faces = np.frombuffer(fragment, dtype=np.uint32, count=-1, offset=(4 + 12 * num_vertices))
  except ValueError:
    raise ValueError("""Unable to process fragment. Violation: Input buffer too small.
        Minimum size: Buffer Length: {}, Actual Size: {}
      """.format(4 + 4*num_vertices, len(fragment)))

  return {
    'num_vertices': num_vertices,
    'vertices': vertices.reshape( num_vertices, 3 ),
    'faces': faces,
    'encoding_type': 'precomputed'
  }

def decode_draco_mesh_buffer(fragment):
    try:
        mesh_object = DracoPy.decode_buffer_to_mesh(fragment)
        vertices = np.array(mesh_object.points)
        faces = np.array(mesh_object.faces)
    except ValueError:
        raise ValueError("Not a valid draco mesh")

    assert len(vertices) % 3 == 0, "Draco mesh vertices not 3-D"
    num_vertices = len(vertices) // 3

    # For now, just return this dict until we figure out
    # how exactly to deal with Draco's lossiness/duplicate vertices
    return {
        'num_vertices': num_vertices,
        'vertices': vertices.reshape(num_vertices, 3),
        'faces': faces,
        'encoding_type': 'draco'
    }


class GrapheneMeshService(object):
    def __init__(self, vol):
        self.vol = vol

    def _get_fragment_filenames(self, seg_id, lod=0):
        #TODO: add lod to endpoint

        url = "%s/%s:%s?verify=True" % (self.vol.manifest_endpoint, seg_id, lod)
        r = requests.get(url)
        if (r.status_code != 200):
            raise Exception(f'manifest endpoint {url} not responding')

        filenames = json.loads(r.content)["fragments"]

        return filenames

    def _get_mesh_fragments(self, filenames):
        mesh_dir = self.vol.info['mesh']
        paths = ["%s/%s" % (mesh_dir, filename) for filename in filenames]
        with Storage(self.vol.layer_cloudpath,
                     progress=self.vol.progress) as stor:
            fragments = stor.get_files(paths)

        return fragments

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

    def get(self, seg_id, remove_duplicate_vertices=False):
        """
        Merge fragments derived from these segids into a single vertex and face list.

        Why merge multiple segids into one mesh? For example, if you have a set of
        segids that belong to the same neuron.

        segid: (iterable or int) segids to render into a single mesh

        Optional:
          remove_duplicate_vertices: bool, fuse exactly matching vertices within a chunk
        Returns: {
          num_vertices: int,
          vertices: [ (x,y,z), ... ]  # floats
          faces: [ int, int, int, ... ] # int = vertex_index, 3 to a face
        }

        """
        if isinstance(seg_id, list) or isinstance(seg_id, np.ndarray):
            assert len(seg_id) == 1
            seg_id = seg_id[0]

        fragment_filenames = self._get_fragment_filenames(seg_id)
        fragments = self._get_mesh_fragments(fragment_filenames)
        # fragments = sorted(fragments, key=lambda frag: frag['filename'])  # make decoding deterministic

        # decode all the fragments
        meshdata = []
        for frag in tqdm(fragments, disable=(not self.vol.progress),
                         desc="Decoding Mesh Buffer"):
            try:
                # Easier to ask forgiveness than permission
                mesh = decode_draco_mesh_buffer(frag["content"])
            except DracoPy.FileTypeException:
                mesh = decode_mesh_buffer(frag["content"])
            meshdata.append(mesh)

        return self._produce_output(meshdata,
                                    remove_duplicate_vertices)
