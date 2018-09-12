import re
import os

import numpy as np
import struct
from tqdm import tqdm

from . import lib
from .lib import red
from .storage import Storage

class PrecomputedMeshService(object):
  def __init__(self, vol):
    self.vol = vol

  def _manifest_path(self, segid):
    mesh_dir = self.vol.info['mesh']
    mesh_json_file_name = str(segid) + ':0'
    return os.path.join(mesh_dir, mesh_json_file_name)

  def _get_raw_frags(self, segid):
    """Download the raw mesh fragments for this seg ID."""
    
    mesh_dir = self.vol.info['mesh']
    mesh_json_file_name = str(segid) + ':0'
    download_path = self._manifest_path(segid)

    with Storage(self.vol.layer_cloudpath, progress=self.vol.progress) as stor:
      fragments = stor.get_json(download_path)['fragments']
      
      # Older mesh manifest generation tasks had a bug where they
      # accidently included the manifest file in the list of mesh
      # fragments. Exclude these accidental files, no harm done.
      fragments = [ f for f in fragments if f != mesh_json_file_name ] 

      paths = [ os.path.join(mesh_dir, fragment) for fragment in fragments ]
      frag_datas = stor.get_files(paths)  
    return frag_datas

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
    if type(segids) != list:
      segids = [ segids ]

    dne = self._check_missing_manifests(segids)

    if len(dne) > 0:
      missing = ', '.join([ str(segid) for segid in dne ])
      raise ValueError(red(
        'Segment ID(s) {} are missing corresponding mesh manifests.\nAborted.' \
          .format(missing)
      ))

    # mesh data returned in fragments
    fragments = []
    for segid in segids:
      fragments.extend( self._get_raw_frags(segid) )

    # decode all the fragments
    meshdata = []
    for frag in tqdm(fragments, disable=(not self.vol.progress), desc="Decoding Mesh Buffer"):
      mesh = decode_mesh_buffer(frag['filename'], frag['content'])
      meshdata.append(mesh)

    vertices, faces = [], []
    vertexct = 0
    for mesh in meshdata:
      vertices.extend(mesh['vertices'])
      f = np.array(mesh['faces'], dtype=np.uint32) + vertexct
      faces.extend(f.tolist())
      vertexct += mesh['num_vertices']

    if remove_duplicate_vertices:
      all_vertices = np.array(vertices)[faces]
      unique_vertices, unique_indices = np.unique(all_vertices,
                                                  return_inverse=True, axis=0)
      vertices = list(map(tuple, unique_vertices))
      faces = list(unique_indices)

    output = {
      'num_vertices': len(vertices),
      'vertices': vertices,
      'faces': faces,
    }

    return output

  def _check_missing_manifests(self, segids):
    """Check if there are any missing mesh manifests prior to downloading."""
    manifest_paths = [ self._manifest_path(segid) for segid in segids ]
    with Storage(self.vol.layer_cloudpath, progress=self.vol.progress) as stor:
      exists = stor.files_exist(manifest_paths)
    
    dne = []
    for path, there in exists.items():
      if not there:
        (segid,) = re.search('(\d+):0$', path).groups()
        dne.append(segid)
    return dne

  def save(self, segids, filename=None, file_format='obj'):
    """
    Save one or more segids into a common mesh format as a single file.

    segids: int, string, or list thereof

    Supported Formats: 'obj'
    """
    if type(segids) != list:
      segids = [ segids ]

    meshdata = self.get(segids)

    if file_format != 'obj':
      raise NotImplementedError('Only .obj is currently supported.')

    if not filename:
      filename = str(segids[0])
      if len(segids) > 1:
        filename = "{}_{}".format(segids[0], segids[-1])

    with open('./{}.obj'.format(filename), 'wb') as f:
      objdata = mesh_to_obj(meshdata, progress=self.vol.progress)
      objdata = '\n'.join(objdata) + '\n'
      f.write(objdata.encode('utf8'))
      

def decode_mesh_buffer(filename, fragment):
    num_vertices = struct.unpack("=I", fragment[0:4])[0]
    vertex_data = fragment[4:4+(num_vertices*3)*4]
    face_data = fragment[4+(num_vertices*3)*4:]
    vertices = []

    if len(vertex_data) != 12 * num_vertices:
      raise ValueError("""Unable to process fragment {}. Violation: len vertex data != 12 * num vertices
        Array Length: {}, Vertex Count: {}
      """.format(filename, len(vertex_data), num_vertices))
    elif len(face_data) % 12 != 0:
      raise ValueError("""Unable to process fragment {}. Violation: len face data is not a multiple of 12.
        Array Length: {}""".format(filename, len(face_data)))

    for i in range(0, len(vertex_data), 12):
      x = struct.unpack("=f", vertex_data[i:i+4])[0]
      y = struct.unpack("=f", vertex_data[i+4:i+8])[0]
      z = struct.unpack("=f", vertex_data[i+8:i+12])[0]
      vertices.append((x,y,z))

    faces = []
    for i in range(0, len(face_data), 4):
      vertex_number = struct.unpack("=I", face_data[i:i+4])[0]
      if vertex_number >= num_vertices:
        raise ValueError(
          "Unable to process fragment {}. Vertex number {} greater than num_vertices {}.".format(
            filename, vertex_number, num_vertices
          )
        )
      faces.append(vertex_number)

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
  
  faces = [ face + 1 for face in mesh['faces'] ] # obj counts from 1 not 0 as in python
  for i in tqdm(range(0, len(faces), 3), disable=(not progress), desc='Face Representation'):
    objdata.append('f %s %s %s' % (faces[i], faces[i+1], faces[i+2]))
  
  return objdata