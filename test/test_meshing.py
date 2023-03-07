import os
import numpy as np
import pytest 

from cloudfiles import CloudFiles
from cloudvolume import Vec, Bbox, CloudVolume, Storage, Mesh

@pytest.fixture
def unsharded_vol():
  test_dir = os.path.dirname(os.path.abspath(__file__))
  test_dir = os.path.join(test_dir, "test_seg_unsharded")
  print(test_dir)
  return CloudVolume("file://" + test_dir)

def test_mesh_fragment_download(unsharded_vol):
  paths = unsharded_vol.mesh._get_manifests(13614423)
  assert len(paths[13614423]) == 1
  assert paths[13614423] == [ '13614423:0:0-256_0-256_0-448' ]

  paths = unsharded_vol.mesh._get_manifests(22270104)
  paths = sorted(paths[22270104])
  assert len(paths) == 2
  assert paths == [ 
    '22270104:0:0-256_0-256_0-448',
    '22270104:0:0-256_0-256_448-512',
  ]

def test_get_mesh(unsharded_vol):
  vol = unsharded_vol
  vol.cache.flush()
  mesh = vol.mesh.get(16649205)
  assert len(mesh) == 4956
  assert mesh.vertices.shape[0] == 4956
  assert len(mesh.faces) == 9876
  assert isinstance(mesh.vertices, np.ndarray)
  assert mesh.vertices.dtype == np.float32
  assert mesh.faces.dtype == np.uint32

  meshes = vol.mesh.get([16649205, 22270104], fuse=False)
  assert len(meshes) == 2
  mesh = meshes[16649205]
  assert len(mesh.vertices) == 5176
  assert len(mesh.faces) == 9876
  
  try:
    vol.mesh.get(666666666)
    assert False
  except ValueError:
    pass

  # just don't crash
  mesh = vol.mesh.get(16649205, chunk_size=(512, 512, 100), fuse=True)

  # test transform
  mesh = vol.mesh.get(16649205)

  vol.mesh.transform = np.eye(4) * 2
  vol.mesh.transform[:,3] = 1

  double_mesh = vol.mesh.get(16649205)

  assert (2 * mesh.vertices + 1 == double_mesh.vertices).all()

  # test non-standard fragment file names
  # 1:0 has the contents {"fragments":["randomname"]}
  # which is a copy of 94081437
  mesh = vol.mesh.get(1)
  mesh_orig = vol.mesh.get(94081437)
  assert mesh == mesh_orig

def test_put(unsharded_vol):
  mesh = Mesh(vertices=[[0,0,0], [1,1,1], [2,2,2]], faces=[0,1,2])
  mesh.segid = 777

  cf = CloudFiles(unsharded_vol.mesh.meta.layerpath)

  unsharded_vol.mesh.put(mesh)
  lst = list(cf)
  assert "777:0" in lst
  assert "777:0:1" in lst
  m = unsharded_vol.mesh.get(777, fuse=False)[777]
  assert len(m.faces) == 1
  assert m.segid == 777
  unsharded_vol.mesh.delete(777)
  lst = list(cf)
  assert "777:0" not in lst
  assert "777:0:1" not in lst

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

def test_get_mesh_caching(unsharded_vol):
  unsharded_vol.cache.enabled = True
  unsharded_vol.cache.flush()

  mesh = unsharded_vol.mesh.get(16649205)
  print(unsharded_vol.cache.list_meshes())
  assert set(unsharded_vol.cache.list_meshes()) == set([ 
    '16649205:0:0-256_0-256_0-448.gz',
    '16649205:0:0-256_0-256_448-512.gz', 
    '16649205:0'
  ])

  assert len(mesh) == 4956
  assert mesh.vertices.shape[0] == 4956
  assert len(mesh.faces) == 9876
  assert isinstance(mesh.vertices, np.ndarray)
  assert mesh.vertices.dtype == np.float32
  assert mesh.faces.dtype == np.uint32

  meshes = unsharded_vol.mesh.get([22270104, 16649205], fuse=False)
  assert len(meshes) == 2
  mesh = meshes[16649205]
  assert len(mesh.vertices) == 5176
  assert len(mesh.faces) == 9876
  
  try:
    unsharded_vol.mesh.get(666666666)
    assert False
  except ValueError:
    pass

  unsharded_vol.cache.flush()

def test_get_mesh_order_stability(unsharded_vol):
  first_mesh = unsharded_vol.mesh.get([22270104, 16649205], fuse=True)
  
  for _ in range(5):
    next_mesh = unsharded_vol.mesh.get([22270104, 16649205], fuse=True)
    assert len(first_mesh.vertices) == len(next_mesh.vertices)
    assert np.all(first_mesh.vertices == next_mesh.vertices)
    assert np.all(first_mesh.faces == next_mesh.faces)

@pytest.mark.parametrize("vqb", (10,16))
def test_stored_model_quantization(vqb):
  from cloudvolume.datasource.precomputed.mesh import multilod

  chunk_shape = [200, 200, 200]
  grid_origin = np.random.randint(0, 101, size=(3,)).astype(np.float32)
  vertices = grid_origin + np.random.uniform(0, 199, size=(1000,3))

  manifest = multilod.MultiLevelPrecomputedMeshManifest(
    segment_id=18, 
    chunk_shape=chunk_shape, 
    grid_origin=grid_origin, 
    num_lods=1, 
    lod_scales=[ 1 ], 
    vertex_offsets=[[0,0,0]],
    num_fragments_per_lod=[1], 
    fragment_positions=[[[0,0,0]]], 
    fragment_offsets=[0],
  )

  lod = 0
  kwargs = {
    "lod": lod,
    "vertex_quantization_bits": vqb, 
    "frag": 0
  }

  quantized_verts = multilod.to_stored_model_space( 
    vertices, manifest, **kwargs
  )
  restored_verts1 = multilod.from_stored_model_space(
    quantized_verts, manifest, **kwargs
  )
  quantized_verts = multilod.to_stored_model_space( 
    restored_verts1, manifest, **kwargs
  )
  restored_verts2 = multilod.from_stored_model_space(
    quantized_verts, manifest, **kwargs
  )

  precision = float(np.max(manifest.chunk_shape) / (2**vqb))
  max_error = float(np.max(np.abs(restored_verts1 - vertices)))
  assert max_error < precision
  assert np.all(np.isclose(restored_verts1, restored_verts2))