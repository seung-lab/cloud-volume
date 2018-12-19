import json
import re
import os

import struct
import numpy as np
from tqdm import tqdm

from .lib import red, toiter
from .storage import Storage

class PrecomputedMeshService(object):
  def __init__(self, vol):
    self.vol = vol

  def _manifest_path(self, segid):
    mesh_dir = self.vol.info['mesh']
    mesh_json_file_name = str(segid) + ':0'
    return os.path.join(mesh_dir, mesh_json_file_name)

  def _get_manifests(self, segids):
    mesh_dir = self.vol.info['mesh']
    
    paths = [ self._manifest_path(segid) for segid in segids ]

    with Storage(self.vol.layer_cloudpath, progress=self.vol.progress) as stor:
      fragments = stor.get_files(paths)

    contents = []
    for frag in fragments:
      content = frag['content'].decode('utf8')
      content = json.loads(content)
      contents.extend(content['fragments'])

    return contents

  def _get_mesh_fragments(self, paths):
    mesh_dir = self.vol.info['mesh']
    paths = [ os.path.join(mesh_dir, path) for path in paths ]
    with Storage(self.vol.layer_cloudpath, progress=self.vol.progress) as stor:
      fragments = stor.get_files(paths)

    return fragments

  def get(self, segids, remove_duplicate_vertices=True):
    """
    Merge fragments derived from these segids into a single vertex and face list.

    Why merge multiple segids into one mesh? For example, if you have a set of
    segids that belong to the same neuron.

    segids: (list or int) segids to render into a single mesh

    remove_duplicate_vertices: bool, fuse exactly matching vertices

    Returns: {
      num_vertices: int,
      vertices: [ (x,y,z), ... ]  # floats
      faces: [ int, int, int, ... ] # int = vertex_index, 3 to a face
    }

    """
    segids = toiter(segids)
    dne = self._check_missing_manifests(segids)

    if dne:
      missing = ', '.join([ str(segid) for segid in dne ])
      raise ValueError(red(
        'Segment ID(s) {} are missing corresponding mesh manifests.\nAborted.' \
        .format(missing)
      ))

    fragments = self._get_manifests(segids)
    fragments = self._get_mesh_fragments(fragments)
    
    # decode all the fragments
    meshdata = []
    for frag in tqdm(fragments, disable=(not self.vol.progress), desc="Decoding Mesh Buffer"):
      mesh = decode_mesh_buffer(frag['filename'], frag['content'])
      meshdata.append(mesh)

    vertexct = np.zeros(len(meshdata) + 1, np.uint32)
    vertexct[1:] = np.cumsum([x['num_vertices'] for x in meshdata])
    vertices = np.concatenate([x['vertices'] for x in meshdata])
    faces = np.concatenate([ 
      mesh['faces'] + vertexct[i] for i, mesh in enumerate(meshdata) 
    ])

    if remove_duplicate_vertices:
      vertices, faces = np.unique(vertices[faces], return_inverse=True, axis=0)
      faces = faces.astype(np.uint32)

    return {
      'num_vertices': len(vertices),
      'vertices': vertices,
      'faces': faces,
    }

  def _check_missing_manifests(self, segids):
    """Check if there are any missing mesh manifests prior to downloading."""
    manifest_paths = [ self._manifest_path(segid) for segid in segids ]
    with Storage(self.vol.layer_cloudpath, progress=self.vol.progress) as stor:
      exists = stor.files_exist(manifest_paths)

    dne = []
    for path, there in exists.items():
      if not there:
        (segid,) = re.search(r'(\d+):0$', path).groups()
        dne.append(segid)
    return dne

  def save(self, segids, filepath=None, file_format='ply'):
    """
    Save one or more segids into a common mesh format as a single file.

    segids: int, string, or list thereof
    filepath: string or None (optional)
    file_format: string (optional)

    Supported Formats: 'obj', 'ply'
    """
    if type(segids) != list:
      segids = [segids]

    meshdata = self.get(segids)

    if not filepath:
      filepath = str(segids[0])
      if len(segids) > 1:
        filepath = "{}_{}.{}".format(segids[0], segids[-1], file_format)

    if file_format == 'obj':
      objdata = mesh_to_obj(meshdata, progress=self.vol.progress)
      objdata = '\n'.join(objdata) + '\n'
      data = objdata.encode('utf8')
    elif file_format == 'ply':
      data = mesh_to_ply(meshdata)
    else:
      raise NotImplementedError('Only .obj and .ply is currently supported.')

    with open(filepath, 'wb') as f:
      f.write(data)

def decode_mesh_buffer(filename, fragment):
  num_vertices = struct.unpack("=I", fragment[0:4])[0]
  try:
    vertices = np.frombuffer(fragment, 'float32, float32, float32', num_vertices, 4)
    faces = np.frombuffer(fragment, np.uint32, -1, 4 + 12*num_vertices)
  except ValueError:
    raise ValueError("""Unable to process fragment {}. Violation: Input buffer too small.
        Minimum size: Buffer Length: {}, Actual Size: {}
      """.format(filename, 4 + 4*num_vertices, len(fragment)))

  return {
      'filename': filename,
      'num_vertices': num_vertices,
      'vertices': vertices,
      'faces': faces
  }

def mesh_to_obj(mesh, progress=False):
  objdata = []

  for vertex in tqdm(mesh['vertices'], disable=(not progress), desc='Vertex Representation'):
    objdata.append('v %s %s %s' % (vertex[0], vertex[1], vertex[2]))

  faces = [face + 1 for face in mesh['faces']] # obj counts from 1 not 0 as in python
  for i in tqdm(range(0, len(faces), 3), disable=(not progress), desc='Face Representation'):
    objdata.append('f %s %s %s' % (faces[i], faces[i+1], faces[i+2]))

  return objdata

def mesh_to_ply(mesh):
  # FIXME: Storing vertices per face (3) as uchar would save a bit storage
  #        but I can't figure out how to mix uint8 with uint32 efficiently.
  vertexct = mesh['num_vertices']
  trianglect = len(mesh['faces']) // 3

  # Header
  plydata = bytearray("""ply
format binary_little_endian 1.0
element vertex {}
property float x
property float y
property float z
element face {}
property list int int vertex_indices
end_header
""".format(vertexct, trianglect).encode('utf8'))

  # Vertex data (x y z)
  plydata.extend(mesh['vertices'].tobytes())

  # Faces (3 f1 f2 f3)
  plydata.extend(
      np.insert(mesh['faces'].reshape(-1, 3), 0, 3, axis=1).tobytes())

  return plydata
