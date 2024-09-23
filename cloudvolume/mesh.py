import copy
import re
import struct
import sys

import numpy as np

from .exceptions import MeshDecodeError
from .lib import Vec, Bbox

def is_draco_chunk_aligned(verts, chunk_size, draco_grid_size):
  """
  Return a mask that for each vertex is true iff it is within
  half a draco_grid_size from a chunk border.
  """
  dist_to_chunk_behind = np.mod(verts, chunk_size)
  dist_to_chunk_ahead = chunk_size - dist_to_chunk_behind
  # Draco rounds up
  is_on_chunk_behind = np.any(
    dist_to_chunk_behind < (draco_grid_size / 2),
    axis=1,
  )
  is_on_chunk_ahead = np.any(
    dist_to_chunk_ahead <= (draco_grid_size / 2),
    axis=1,
  )
  return np.logical_or(is_on_chunk_behind, is_on_chunk_ahead)
    

class Mesh(object):
  """
  Represents the vertices, faces, and normals of a mesh
  as numpy arrays.

  class Mesh:
    ndarray[float32, ndim=2] self.vertices: [ [x,y,z], ... ]
    ndarray[uint32,  ndim=2] self.faces:    [ [v1,v2,v3], ... ]
    ndarray[float32, ndim=2] self.normals:  [ [nx,ny,nz], ... ]
  """
  def __init__(
    self, vertices, faces, normals=None, 
    segid=None, encoding_type=None, encoding_options=None
  ):
    self.vertices = np.array(vertices, dtype=np.float32)
    self.faces = np.array(faces, dtype=np.uint32)

    if normals is None:
      self.normals = np.array([], dtype=np.float32).reshape((0,3))
    else:
      self.normals = np.array(normals, dtype=np.float32)

    self.segid = segid
    self.encoding_type = encoding_type
    self.encoding_options = encoding_options 

  def __len__(self):
    return self.vertices.shape[0]

  def __eq__(self, other):
    """Tests strict equality between two meshes."""

    no_self_normals = self.normals is None or self.normals.size == 0
    no_other_normals = other.normals is None or other.normals.size == 0

    if no_self_normals != no_other_normals:
      return False

    if self.vertices.shape[0] != other.vertices.shape[0]:
      return False
       
    equality = np.all(self.vertices == other.vertices) \
      and np.all(self.faces == other.faces)

    if no_self_normals:
      return equality

    return (equality and np.all(self.normals == other.normals))

  def __sizeof__(self):
    attr_bytes = sum(( 
      sys.getsizeof(x)
      for x in [
        self.segid, self.encoding_type, self.encoding_options
      ]
    ))
    npy_bytes = sum([
      (x.nbytes if isinstance(x, np.ndarray) else sys.getsizeof(x))
      for x in [ self.vertices, self.faces, self.normals ]
    ])
    return attr_bytes + npy_bytes


  def __repr__(self):
    return "Mesh(vertices<{}>, faces<{}>, normals<{}>, segid={}, encoding_type=<{}>)".format(
      self.vertices.shape[0], self.faces.shape[0], self.normals.shape[0],
      self.segid, self.encoding_type
    )

  def empty(self):
    return self.vertices.size == 0 or self.faces.size == 0

  def clone(self):
    return Mesh(
      np.copy(self.vertices), np.copy(self.faces), np.copy(self.normals),
      self.segid, 
      encoding_type=copy.deepcopy(self.encoding_type),
      encoding_options=copy.deepcopy(self.encoding_options),
    )

  def edges(self):
    """
    Generate an edge list from the faces. 
    edges are not guaranteed to be unique.
    """
    srt = lambda x,y: (x,y) if x < y else (y,x)
    for face in self.faces:
      yield srt(face[0], face[1])
      yield srt(face[1], face[2])
      yield srt(face[0], face[2])

  def triangles(self):
    """
    Faces are numbered using the index of vertices,
    but sometimes it is convenient to have a list 
    of triangles in their proper coordinate space.
    """
    Nf = self.faces.shape[0]
    tris = np.zeros( (Nf, 3, 3), dtype=np.float32, order='C' ) # triangle, vertices, (x,y,z)

    for i in range(Nf):
      for j in range(3):
        tris[i,j,:] = self.vertices[ self.faces[i,j] ]

    return tris

  @classmethod
  def concatenate(cls, *meshes, segid=None):
    vertex_ct = np.zeros(len(meshes) + 1, np.uint32)
    vertex_ct[1:] = np.cumsum([ len(mesh) for mesh in meshes ])

    vertices = np.concatenate([ mesh.vertices for mesh in meshes ])
    
    faces = np.concatenate([ 
      mesh.faces + vertex_ct[i] for i, mesh in enumerate(meshes) 
    ])

    normals = np.concatenate([ mesh.normals for mesh in meshes ])

    encoding_type = list(set([ mesh.encoding_type for mesh in meshes ]))
    if len(encoding_type) == 1:
      encoding_type = encoding_type[0]

    return Mesh(vertices, faces, normals, encoding_type=encoding_type, segid=segid)

  def consolidate(self):
    """Remove duplicate vertices and faces. Returns a new mesh object."""
    if self.empty():
      return Mesh([], [], normals=None)

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

    return Mesh(eff_verts, eff_faces, None, 
      segid=self.segid,
      encoding_type=copy.deepcopy(self.encoding_type),
      encoding_options=copy.deepcopy(self.encoding_options),
    )

  @classmethod
  def from_precomputed(self, binary, segid=None):
    """
    Mesh from_precomputed(self, binary)

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
      raise MeshDecodeError("""
        The input buffer is too small for the Precomputed format.
        Minimum Bytes: {} 
        Actual Bytes: {}
      """.format(4 + 4 * num_vertices, len(binary)))

    vertices = vertices.reshape(num_vertices, 3)
    faces = faces.reshape(faces.size // 3, 3)

    return Mesh(
      vertices, faces, 
      segid=segid, 
      normals=None, 
      encoding_type='precomputed'
    )

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
  def from_obj(self, text, segid=None):
    """Given a string representing a Wavefront OBJ file, decode to a Mesh."""

    vertices = []
    faces = []
    normals = []

    if type(text) is bytes:
      text = text.decode('utf8')

    face_re_cplx = re.compile(r'f\s+(\d+)/(\d*)?/(\d+)?\s+(\d+)/(\d*)?/(\d+)?\s+(\d+)/(\d*)?/(\d+)?')
    face_re_simple = re.compile(r'f\s+(\d+)\s+(\d+)\s+(\d+)')
    vertex_re = re.compile(r'v\s+([-\d\.e]+)\s+([-\d\.e]+)\s+([-\d\.e]+)')
    vn_re = re.compile(r'vn\s+([-\d\.e]+)\s+([-\d\.e]+)\s+([-\d\.e]+)')

    for line in text.split('\n'):
      line = line.strip()
      if len(line) == 0:
        continue
      elif line[0] == '#':
        continue
      elif line[0] == 'f':
        if line.find('/') != -1:
          # e.g. f 6092/2095/6079 6087/2092/6075 6088/2097/6081
          # i.e. f vertex_1/texture_1/normal_1 etc
          (v1, vt1, vn1, v2, vt2, vn2, v3, vt3, vn3) = re.match(face_re_cplx, line).groups()
        else:
          (v1, v2, v3) = re.match(face_re_simple, line).groups()
        faces.append( (int(v1), int(v2), int(v3)) )
      elif line[0] == 'v':
        if line[1] == 't': # vertex textures not supported
          # e.g. vt 0.351192 0.337058
          continue 
        elif line[1] == 'n': # vertex normals
          # e.g. vn 0.992266 -0.033290 -0.119585
          (n1, n2, n3) = re.match(vn_re, line).groups()
          normals.append( (float(n1), float(n2), float(n3)) )
        else:
          # e.g. v -0.317868 -0.000526 -0.251834
          (v1, v2, v3) = re.match(vertex_re, line).groups()
          vertices.append( (float(v1), float(v2), float(v3)) )

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.uint32)
    normals = np.array(normals, dtype=np.float32)

    return Mesh(
      vertices, 
      faces - 1, 
      normals, 
      segid=segid, 
      encoding_type='precomputed'
    )

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

  @classmethod
  def from_draco(cls, binary, segid=None):
    import DracoPy

    try:
      mesh = DracoPy.decode(binary)
    except ValueError:
      raise MeshDecodeError("Not a valid draco mesh.")

    return Mesh(
      mesh.points, mesh.faces, 
      segid=segid,
      normals=None,
      encoding_type='draco', 
      encoding_options=mesh.encoding_options
    )

  def deduplicate_vertices(self, is_chunk_aligned):
    faces = self.faces
    verts = self.vertices
    # find all vertices that have exactly 2 duplicates
    unique_vertices, unique_inverse, counts = np.unique(
      verts, return_inverse=True, return_counts=True, axis=0
    )
    
    only_double = np.where(counts == 2)[0]
    is_doubled = np.isin(unique_inverse, only_double)
    # this stores whether each vertex should be merged or not
    do_merge = np.array(is_doubled & is_chunk_aligned)

    # setup an artificial 4th coordinate for vertex positions
    # which will be unique in general, 
    # but then repeated for those that are merged
    new_vertices = np.hstack((verts, np.arange(verts.shape[0])[:, np.newaxis]))
    new_vertices[do_merge, 3] = -1
  
    faces = faces.flatten()

    # use unique to make the artificial vertex list unique and reindex faces
    vertices, newfaces = np.unique(new_vertices[faces], return_inverse=True, axis=0)
    newfaces = newfaces.astype(np.uint32).reshape( (len(newfaces) // 3, 3) )

    return Mesh(vertices[:,0:3], newfaces, None, segid=self.segid, 
      encoding_type=self.encoding_type, encoding_options=self.encoding_options
    )

  def deduplicate_chunk_boundaries(self, chunk_size, is_draco=False, draco_grid_size=None, offset=(0,0,0)):
    offset = Vec(*offset)
    verts = self.vertices - offset
    # find all vertices that are exactly on chunk_size boundaries
    if is_draco:
      if draco_grid_size is None:
        raise ValueError('Must specify draco grid size to dedup draco meshes')
      is_chunk_aligned = is_draco_chunk_aligned(verts, chunk_size, draco_grid_size=draco_grid_size)
    else:
      is_chunk_aligned = np.any(np.mod(verts, chunk_size) == 0, axis=1)

    return self.deduplicate_vertices(is_chunk_aligned)

  def crop(self, bbox:Bbox):
    """
    Create a cropped version of the mesh.
    """
    if self.empty():
      return Mesh([], [], normals=None)

    vert_idx = []
    mapping = {}
    discard = set()
    j = 0
    for i, vert in enumerate(self.vertices):
      if bbox.contains(vert):
        vert_idx.append(i)
        mapping[i] = j
        j += 1
      else:
        discard.add(i)

    vert_idx = np.array(vert_idx, dtype=int)

    cropped_faces_idx = []

    for i, (f1, f2, f3) in enumerate(self.faces):
      if f1 in discard or f2 in discard or f3 in discard:
        continue
      cropped_faces_idx.append(i)

    cropped_faces_idx = np.array(cropped_faces_idx, dtype=int)

    cropped_verts = self.vertices[vert_idx]
    cropped_faces = self.faces[cropped_faces_idx]
    cropped_normals = None

    for face in cropped_faces:
      face[0] = mapping[face[0]]
      face[1] = mapping[face[1]]
      face[2] = mapping[face[2]]

    if self.normals is not None and len(self.normals):
      cropped_normals = self.normals[cropped_faces_idx]

    return Mesh(
      cropped_verts, 
      cropped_faces, 
      cropped_normals, 
      segid=self.segid,
      encoding_type=copy.deepcopy(self.encoding_type),
      encoding_options=copy.deepcopy(self.encoding_options),
    )

  def viewer(self):
    # thanks to ChatGPT for making it easy to figure out
    # how to display VTK meshes.
    try:
      import vtk
    except ImportError:
      print("The mesh viewer requires the OpenGL based vtk. Try: pip install vtk --upgrade")
      return
    polydata = self._create_vtk_mesh(self.vertices, self.faces)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    render_window.SetSize(1024, 1024)

    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.3)  # Background color

    text = "Mesh Viewer"
    if self.segid is not None:
      render_window.SetWindowName(text + " (Label {})".format(self.segid))
    else:
      render_window.SetWindowName(text)

    render_window.Render()
    render_window_interactor.Start()

  def _create_vtk_mesh(self, vertices, faces):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(vertices))
    
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    
    vtk_faces = vtk.vtkCellArray()
    vtk_faces.SetCells(
      faces.shape[0], 
      numpy_to_vtkIdTypeArray(np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten())
    )

    polydata.SetPolys(vtk_faces)
    
    return polydata

