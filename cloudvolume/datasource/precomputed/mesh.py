import six

from collections import defaultdict
import itertools
import json
import re
import os

import struct
import numpy as np
from tqdm import tqdm

from ...lib import yellow, red, toiter
from ...storage import Storage

SEGIDRE = re.compile(r'\b(\d+):0.*?$')

def filename_to_segid(filename):
  matches = SEGIDRE.search(filename)
  if matches is None:
    raise ValueError("There was an issue with the fragment filename: " + filename)

  segid, = matches.groups()
  return int(segid)

NOTICE = {
  'vertices': 0,
  'num_vertices': 0,
  'faces': 0,
}

def deprecation_notice(key):
  if NOTICE[key] < 1:
    print(yellow("""
  Deprecation Notice: Meshes, formerly dicts, are now PrecomputedMesh objects
  as of CloudVolume 0.51.0. 

  Please change mesh['{}'] to mesh.{}
  """.format(key, key)))
    NOTICE[key] += 1

class PrecomputedMesh(object):
  """
  Represents the vertices, faces, and normals of a mesh
  as numpy arrays.

  class PrecomputedMesh:
    ndarray[float32, ndim=2] self.vertices: [ [x,y,z], ... ]
    ndarray[uint32,  ndim=2] self.faces:    [ [v1,v2,v3], ... ]
    ndarray[float32, ndim=2] self.normals:  [ [nx,ny,nz], ... ]

  """
  def __init__(self, vertices, faces, normals=None, segid=None):
    self.vertices = np.array(vertices, dtype=np.float32)
    self.faces = np.array(faces, dtype=np.uint32)

    if normals is None:
      self.normals = np.array([], dtype=np.float32).reshape((0,3))
    else:
      self.normals = np.array(normals, dtype=np.float32)

    self.segid = segid

  def __len__(self):
    return self.vertices.shape[0]

  def __eq__(self, other):
    """Tests strict equality between two meshes."""

    no_self_normals = self.normals is None or self.normals.size == 0
    no_other_normals = other.normals is None or other.normals.size == 0

    if no_self_normals != no_other_normals:
      return False
       
    equality = np.all(self.vertices == other.vertices) \
      and np.all(self.faces == other.faces)

    if no_self_normals:
      return equality

    return (equality and np.all(self.normals == other.normals))

  def __repr__(self):
    return "PrecomputedMesh(vertices<{}>, faces<{}>, normals<{}>)".format(
      self.vertices.shape[0], self.faces.shape[0], self.normals.shape[0]
    )

  def __getitem__(self, key):
    val = None 
    if key == 'vertices':
      val = self.vertices
    elif key == 'num_vertices':
      val = len(self)
    elif key == 'faces':
      val = self.faces
    else:
      raise KeyError("{} not found.".format(key))

    deprecation_notice(key)
    return val

  def empty(self):
    return self.vertices.size == 0 or self.faces.size == 0

  def clone(self):
    return PrecomputedMesh(np.copy(self.vertices), np.copy(self.faces), np.copy(self.normals))

  @classmethod
  def concatenate(cls, *meshes):
    vertex_ct = np.zeros(len(meshes) + 1, np.uint32)
    vertex_ct[1:] = np.cumsum([ len(mesh) for mesh in meshes ])

    vertices = np.concatenate([ mesh.vertices for mesh in meshes ])
    
    faces = np.concatenate([ 
      mesh.faces + vertex_ct[i] for i, mesh in enumerate(meshes) 
    ])

    normals = np.concatenate([ mesh.normals for mesh in meshes ])

    return PrecomputedMesh(vertices, faces, normals)

  def consolidate(self):
    """Remove duplicate vertices and faces. Returns a new mesh object."""
    if self.empty():
      return PrecomputedMesh([], [], normals=None)

    vertices = self.vertices
    faces = self.faces
    normals = self.normals

    eff_verts, uniq_idx, idx_representative = np.unique(
      vertices, axis=0, return_index=True, return_inverse=True
    )

    face_vector_map = np.vectorize(lambda x: idx_representative[x])
    eff_faces = face_vector_map(faces)
    eff_faces = np.unique(eff_faces, axis=0)

    # normal_vector_map = np.vectorize(lambda idx: normals[idx])
    # eff_normals = normal_vector_map(uniq_idx)

    return PrecomputedMesh(eff_verts, eff_faces, None, segid=self.segid)

  @classmethod
  def from_precomputed(self, binary):
    """
    PrecomputedMesh from_precomputed(self, binary)

    Decode a Precomputed format mesh from a byte string.
    
    Format:
      uint32        Nv * float32 * 3   uint32 * 3 until end
      Nv            (x,y,z)            (v1,v2,v2)
      N Vertices    Vertices           Faces
    """
    num_vertices = struct.unpack("=I", binary[0:4])[0]
    try:
      # count=-1 means all data in buffer
      vertices = np.frombuffer(binary, dtype=np.float32, count=3*num_vertices, offset=4)
      faces = np.frombuffer(binary, dtype=np.uint32, count=-1, offset=(4 + 12 * num_vertices)) 
    except ValueError:
      raise ValueError("""
        The input buffer is too small for the Precomputed format.
        Minimum Bytes: {} 
        Actual Bytes: {}
      """.format(4 + 4 * num_vertices, len(binary)))

    vertices = vertices.reshape(num_vertices, 3)
    faces = faces.reshape(faces.size // 3, 3)

    return PrecomputedMesh(vertices, faces, normals=None)

  def to_precomputed(self):
    """
    bytes to_precomputed(self)

    Convert mesh into binary format compatible with Neuroglancer.
    Does not preserve normals.
    """
    vertex_index_format = [
      np.uint32(self.vertices.shape[0]), # Number of vertices (3 coordinates)
      self.vertices,
      self.faces
    ]
    return b''.join([ array.tobytes('C') for array in vertex_index_format ])

  @classmethod
  def from_obj(self, text):
    """Given a string representing a Wavefront OBJ file, decode to a PrecomputedMesh."""

    vertices = []
    faces = []
    normals = []

    if type(text) is bytes:
      text = text.decode('utf8')

    for line in text.split('\n'):
      line = line.strip()
      if len(line) == 0:
        continue
      elif line[0] == '#':
        continue
      elif line[0] == 'f':
        if line.find('/') != -1:
          # e.g. f 6092/2095/6079 6087/2092/6075 6088/2097/6081
          (v1, vt1, vn1, v2, vt2, vn2, v3, vt3, vn3) = re.match(r'f\s+(\d+)/(\d*)/(\d+)\s+(\d+)/(\d*)/(\d+)\s+(\d+)/(\d*)/(\d+)', line).groups()
        else:
          (v1, v2, v3) = re.match(r'f\s+(\d+)\s+(\d+)\s+(\d+)', line).groups()
        faces.append( (int(v1), int(v2), int(v3)) )
      elif line[0] == 'v':
        if line[1] == 't': # vertex textures not supported
          # e.g. vt 0.351192 0.337058
          continue 
        elif line[1] == 'n': # vertex normals
          # e.g. vn 0.992266 -0.033290 -0.119585
          (n1, n2, n3) = re.match(r'vn\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)', line).groups()
          normals.append( (float(n1), float(n2), float(n3)) )
        else:
          # e.g. v -0.317868 -0.000526 -0.251834
          (v1, v2, v3) = re.match(r'v\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)', line).groups()
          vertices.append( (float(v1), float(v2), float(v3)) )

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.uint32)
    normals = np.array(normals, dtype=np.float32)

    return PrecomputedMesh(vertices, faces - 1, normals)

  def to_obj(self):
    """Return a string representing a .obj file."""
    objdata = []
    objdata += [ 'v {:.5f} {:.5f} {:.5f}'.format(*vertex) for vertex in self.vertices ]
    objdata += [ 'f {} {} {}'.format(*face) for face in (self.faces+1) ] # obj is 1 indexed
    objdata = '\n'.join(objdata) + '\n'
    return objdata.encode('utf8')

  def to_ply(self):
    """
    Return a bytearray in .ply format, 
    a more compact format than .obj that's still widely readable.
    """
    vertexct = self.vertices.shape[0]
    trianglect = self.faces.shape[0]

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

    # Vertex data (x y z): "fff" 
    plydata.extend(self.vertices.tobytes('C'))

    # Faces (3 f1 f2 f3): "3iii" 
    plydata.extend(
      np.insert(self.faces, 0, 3, axis=1).tobytes('C')
    )

    return plydata

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
  
    # use unique to make the artificial vertex list unique and reindex faces
    vertices, newfaces = np.unique(new_vertices[faces], return_inverse=True, axis=0)
    #faces = newfaces.reshape((n_faces, n_dim))
    newfaces = newfaces.astype(np.uint32)

    return vertices[:,0:3], newfaces

class PrecomputedMeshSource(object):
  def __init__(self, meta, cache, config):
    self.meta = meta
    self.cache = cache
    self.config = config

  @property
  def path(self):
    return self.meta.info['mesh']

  def manifest_path(self, segid):
    mesh_json_file_name = str(segid) + ':0'
    return os.path.join(self.path, mesh_json_file_name)

  def _get_manifests(self, segids):
    segids = toiter(segids)    
    paths = [ self.manifest_path(segid) for segid in segids ]
    fragments = self.cache.download(paths)

    contents = {}
    for filename, content in fragments.items():
      content = content.decode('utf8')
      content = json.loads(content)
      segid = filename_to_segid(filename)
      contents[segid] = content['fragments']

    return contents

  def _get_mesh_fragments(self, paths):
    paths = [ os.path.join(self.path, path) for path in paths ]

    compress = self.config.compress
    if compress is None:
      compress = True

    fragments = self.cache.download(paths, compress=compress)
    fragments = [ (filename, content) for filename, content in fragments.items() ]
    fragments = sorted(fragments, key=lambda frag: frag[0]) # make decoding deterministic
    return fragments

  def _check_missing_manifests(self, segids):
    """Check if there are any missing mesh manifests prior to downloading."""
    manifest_paths = [ self.manifest_path(segid) for segid in segids ]
    with Storage(self.meta.cloudpath, progress=self.config.progress) as stor:
      exists = stor.files_exist(manifest_paths)

    dne = []
    for path, there in exists.items():
      if not there:
        (segid,) = re.search(r'(\d+):0$', path).groups()
        dne.append(segid)
    return dne

  def get(
      self, segids, 
      remove_duplicate_vertices=True, 
      fuse=True,
      chunk_size=None
    ):
    """
    Merge fragments derived from these segids into a single vertex and face list.

    Why merge multiple segids into one mesh? For example, if you have a set of
    segids that belong to the same neuron.

    segids: (iterable or int) segids to render into a single mesh

    Optional:
      remove_duplicate_vertices: bool, fuse exactly matching vertices
      fuse: bool, merge all downloaded meshes into a single mesh
      chunk_size: [chunk_x, chunk_y, chunk_z] if pass only merge at chunk boundaries
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
    fragments = fragments.values()
    fragments = list(itertools.chain.from_iterable(fragments)) # flatten
    fragments = self._get_mesh_fragments(fragments)

    # decode all the fragments
    meshdata = defaultdict(list)
    for frag in tqdm(fragments, disable=(not self.config.progress), desc="Decoding Mesh Buffer"):
      segid = filename_to_segid(frag[0])
      try:
        mesh = PrecomputedMesh.from_precomputed(frag[1])
      except Exception:
        print(frag[0], 'had a problem.')
        raise
      meshdata[segid].append(mesh)

    if not fuse:
      return { segid: PrecomputedMesh.concatenate(*meshes) for segid, meshes in six.iteritems(meshdata) }

    meshdata = [ (segid, mesh) for segid, mesh in six.iteritems(meshdata) ]
    meshdata = sorted(meshdata, key=lambda sm: sm[0])
    meshdata = [ mesh for segid, mesh in meshdata ]
    meshdata = list(itertools.chain.from_iterable(meshdata)) # flatten
    mesh = PrecomputedMesh.concatenate(*meshdata)

    if not remove_duplicate_vertices:
      return mesh 

    if not chunk_size:
      return mesh.consolidate()

    vertices, faces = remove_duplicate_vertices_cross_chunks(
      mesh.vertices, mesh.faces, chunk_size
    )
    return PrecomputedMesh(vertices, faces, normals=None)

  def save(self, segids, filepath=None, file_format='ply'):
    """
    Save one or more segids into a common mesh format as a single file.

    segids: int, string, or list thereof
    filepath: string, file-like, or None (optional)
    file_format: string (optional)
    
    Supported Formats: 'obj', 'ply', 'precomputed'
    """
    if type(segids) != list:
      segids = [segids]

    mesh = self.get(segids, remove_duplicate_vertices=True)

    if file_format == 'obj':
      data = mesh.to_obj()
    elif file_format == 'ply':
      data = mesh.to_ply()
    elif file_format == 'precomputed':
      data = mesh.to_precomputed()
    else:
      raise NotImplementedError('Only .obj, .ply, and precomputed are currently supported.')

    if not filepath:
      filepath = str(segids[0]) + "." + file_format
      if len(segids) > 1:
        filepath = "{}_{}.{}".format(segids[0], segids[-1], file_format)

    try:
      filepath.write(data)
    except AttributeError:
      with open(filepath, 'wb') as f:
        f.write(data)
