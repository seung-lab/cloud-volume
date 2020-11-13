import os
import numpy as np
import pytest 

from cloudvolume import Vec, Bbox, CloudVolume, Storage, Mesh

@pytest.mark.parametrize(("use_https"),[True, False])
def test_mesh_fragment_download(use_https):
  vol = CloudVolume('gs://seunglab-test/test_v0/segmentation', use_https=use_https)
  paths = vol.mesh._get_manifests(18)
  assert len(paths) == 1
  assert paths[18] == [ '18:0:0-512_0-512_0-100' ]

  paths = vol.mesh._get_manifests(147)
  assert len(paths) == 1
  assert paths[147] == [ '147:0:0-512_0-512_0-100' ]

def test_get_mesh():
  vol = CloudVolume('gs://seunglab-test/test_v0/segmentation')
  vol.cache.flush()
  mesh = vol.mesh.get(18)
  assert len(mesh) == 6123
  assert mesh.vertices.shape[0] == 6123
  assert len(mesh.faces) == 12242
  assert isinstance(mesh.vertices, np.ndarray)
  assert mesh.vertices.dtype == np.float32
  assert mesh.faces.dtype == np.uint32

  meshes = vol.mesh.get([148, 18], fuse=False)
  assert len(meshes) == 2
  mesh = meshes[18]
  assert len(mesh.vertices) == 6123
  assert len(mesh.vertices) == 6123
  assert len(mesh.faces) == 12242
  
  try:
    vol.mesh.get(666666666)
    assert False
  except ValueError:
    pass

  # just don't crash
  mesh = vol.mesh.get(18, chunk_size=(512, 512, 100), fuse=True)

def test_duplicate_vertices():
  verts = np.array([
    [0,0,0], [0,1,0],
    [1,0,0], [1,1,0],
    [2,0,0], [2,1,0],
    [3,0,0], [3,1,0], 
    [3,0,0],
    [4,0,0], [4,1,0],
    [4,0,0],          # duplicate in x direction
    [5,0,0], [5,1,0],
    [5,0,0],
    [6,0,0], [6,1,0], [6,1,2],
    [7,0,0], [7,1,0],
    [4,0,0]
  ], dtype=np.float32)

  faces = np.array([ 
    [0,1,2], [2,3,4], [4,5,6], [7,8,9],
    [9,10,11], [10,11,12],
    [12,13,14], [14,15,16], [15,16,17],
    [15,18,19], [18,19,20]
  ], dtype=np.uint32)

  mesh = Mesh(verts, faces, segid=666)

  def deduplicate(mesh, x, offset_x=0):
    return mesh.deduplicate_chunk_boundaries(
      (x, 100, 100), is_draco=False, 
      offset=(offset_x,-1,-1) # so y=0,z=0 isn't a chunk boundary
    )

  # test that triple 4 isn't affected
  mesh2 = deduplicate(mesh, x=4)
  assert not np.all(mesh.vertices == mesh2.vertices)
  assert mesh2.vertices.shape[0] == mesh.vertices.shape[0] 

  # pop off the last 4
  mesh.vertices = mesh.vertices[:-1]
  mesh.faces = mesh.faces[:-1]

  # test that 4 is now affected
  mesh2 = deduplicate(mesh, x=4)
  assert not np.all(mesh.vertices == mesh2.vertices)
  assert mesh2.vertices.shape[0] == mesh.vertices.shape[0] - 1

  mesh2 = deduplicate(mesh, x=3)
  assert not np.all(mesh.vertices == mesh2.vertices)
  assert mesh2.vertices.shape[0] == mesh.vertices.shape[0] - 1

  mesh2 = deduplicate(mesh, x=4, offset_x=-1)
  assert not np.all(mesh.vertices == mesh2.vertices)
  assert mesh2.vertices.shape[0] == mesh.vertices.shape[0] - 1

  mesh2 = deduplicate(mesh, x=5)
  assert not np.all(mesh.vertices == mesh2.vertices)
  assert mesh2.vertices.shape[0] == mesh.vertices.shape[0] - 1

  mesh2 = deduplicate(mesh, x=1)
  assert not np.all(mesh.vertices == mesh2.vertices)
  assert mesh2.vertices.shape[0] == mesh.vertices.shape[0] - 3

def test_get_mesh_caching():
  vol = CloudVolume('gs://seunglab-test/test_v0/segmentation', cache=True)
  vol.cache.flush()

  mesh = vol.mesh.get(18)
  
  assert set(vol.cache.list_meshes()) == set([ '18:0:0-512_0-512_0-100.gz', '18:0' ])

  assert len(mesh) == 6123
  assert mesh.vertices.shape[0] == 6123
  assert len(mesh.faces) == 12242
  assert isinstance(mesh.vertices, np.ndarray)
  assert mesh.vertices.dtype == np.float32
  assert mesh.faces.dtype == np.uint32

  meshes = vol.mesh.get([148, 18], fuse=False)
  assert len(meshes) == 2
  mesh = meshes[18]
  assert len(mesh.vertices) == 6123
  assert len(mesh.vertices) == 6123
  assert len(mesh.faces) == 12242
  
  try:
    vol.mesh.get(666666666)
    assert False
  except ValueError:
    pass

  vol.cache.flush()

def test_get_mesh_order_stability():
  vol = CloudVolume('gs://seunglab-test/test_v0/segmentation')
  first_mesh = vol.mesh.get([148, 18], fuse=True)
  
  for _ in range(5):
    next_mesh = vol.mesh.get([148, 18], fuse=True)
    assert len(first_mesh.vertices) == len(next_mesh.vertices)
    assert np.all(first_mesh.vertices == next_mesh.vertices)
    assert np.all(first_mesh.faces == next_mesh.faces)
