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
  assert mesh != mesh2
  assert mesh2.vertices.shape[0] == mesh.vertices.shape[0] - 1

  mesh2 = deduplicate(mesh, x=3)
  assert mesh != mesh2
  assert mesh2.vertices.shape[0] == mesh.vertices.shape[0] - 1

  mesh2 = deduplicate(mesh, x=4, offset_x=-1)
  assert mesh != mesh2
  assert mesh2.vertices.shape[0] == mesh.vertices.shape[0] - 1

  mesh2 = deduplicate(mesh, x=5)
  assert mesh != mesh2
  assert mesh2.vertices.shape[0] == mesh.vertices.shape[0] - 1

  mesh2 = deduplicate(mesh, x=1)
  assert mesh != mesh2
  assert mesh2.vertices.shape[0] == mesh.vertices.shape[0] - 3

def _deduplicate_vertices_np_unique(mesh, is_chunk_aligned):
  """
  The np.unique(..., axis=0) implementation deduplicate_vertices replaced,
  except when no vertex is chunk aligned: then the new code skips the sort
  and keeps the input order, where the old code sorted the vertices.
  """
  if not np.any(is_chunk_aligned):
    # only drop the vertices no face uses
    used, inv = np.unique(mesh.faces, return_inverse=True)
    newfaces = inv.reshape(-1, 3).astype(np.uint32)
    return Mesh(mesh.vertices[used], newfaces, None, segid=mesh.segid)

  verts = mesh.vertices
  unique_vertices, unique_inverse, counts = np.unique(
    verts, return_inverse=True, return_counts=True, axis=0
  )
  is_doubled = np.isin(unique_inverse, np.where(counts == 2)[0])
  do_merge = np.array(is_doubled & is_chunk_aligned)
  new_vertices = np.hstack((verts, np.arange(verts.shape[0])[:, np.newaxis]))
  new_vertices[do_merge, 3] = -1
  faces = mesh.faces.flatten()
  vertices, newfaces = np.unique(new_vertices[faces], return_inverse=True, axis=0)
  newfaces = newfaces.astype(np.uint32).reshape( (len(newfaces) // 3, 3) )
  return Mesh(vertices[:,0:3], newfaces, None, segid=mesh.segid)

@pytest.mark.parametrize("seed", range(50))
def test_deduplicate_vertices_matches_np_unique(seed):
  # Vertices on a small lattice repeat once, twice or more often, some as
  # -0.0; some are used by no face, and some meshes have no faces at all.
  rng = np.random.default_rng(seed)
  n = int(rng.integers(1, 400))
  span = int(rng.integers(1, 6))
  verts = rng.integers(-span, span, size=(n, 3)).astype(np.float32)
  verts[rng.random(n) < 0.2] *= -0.0
  faces = rng.integers(0, n, size=(int(rng.integers(0, 600)), 3)).astype(np.uint32)
  is_chunk_aligned = rng.random(n) < rng.random()
  mesh = Mesh(verts, faces, segid=seed)

  mesh2 = mesh.deduplicate_vertices(is_chunk_aligned)
  assert mesh2 == _deduplicate_vertices_np_unique(mesh, is_chunk_aligned)
  assert mesh2.segid == seed

def test_deduplicate_vertices_none_aligned_keeps_order():
  verts = np.array([[2,0,0], [0,0,0], [9,9,9], [0,0,0], [1,0,0]], dtype=np.float32)
  faces = np.array([[0,1,4], [3,4,0]], dtype=np.uint32)
  mesh = Mesh(verts, faces, segid=1)

  mesh2 = mesh.deduplicate_vertices(np.zeros(5, dtype=bool))
  # vertex 2 is unused and dropped; the rest keep their order
  assert np.array_equal(mesh2.vertices, verts[[0,1,3,4]])
  assert np.array_equal(mesh2.faces, [[0,1,3], [2,3,0]])

def test_deduplicate_vertices_empty():
  mesh = Mesh(np.zeros((0,3)), np.zeros((0,3)), segid=1)
  mesh2 = mesh.deduplicate_vertices(np.zeros(0, dtype=bool))
  assert mesh2.vertices.shape == (0,3)
  assert mesh2.faces.shape == (0,3)

@pytest.mark.parametrize("is_chunk_aligned", [True, False])
def test_deduplicate_vertices_scalar_mask(is_chunk_aligned):
  # a single bool stands for every vertex, as it did with np.unique
  verts = np.array([[0,0,0], [0,0,0], [1,0,0], [2,0,0]], dtype=np.float32)
  faces = np.array([[0,2,3], [1,2,3]], dtype=np.uint32)
  mesh = Mesh(verts, faces, segid=1)

  mesh2 = mesh.deduplicate_vertices(is_chunk_aligned)
  assert mesh2 == _deduplicate_vertices_np_unique(mesh, is_chunk_aligned)
  assert mesh2.vertices.shape[0] == (3 if is_chunk_aligned else 4)

def test_deduplicate_vertices_mask_length_mismatch():
  mesh = Mesh(np.zeros((5,3)), np.array([[0,1,2]]), segid=1)
  with pytest.raises(ValueError):
    mesh.deduplicate_vertices(np.ones(7, dtype=bool))

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

def test_obj_round_trip(unsharded_vol):
  vol = unsharded_vol
  vol.cache.flush()
  mesh = vol.mesh.get(16649205)

  mesh2 = Mesh.from_obj(mesh.to_obj())

  x = mesh.vertices.flatten()
  x.sort()

  y = mesh2.vertices.flatten()
  y.sort()

  assert np.all(np.isclose(x, y, atol=1e-5))

def test_crop_mesh(unsharded_vol):
  vol = unsharded_vol
  vol.cache.flush()

  mesh = vol.mesh.get(16649205)
  
  corner = mesh.vertices.min(axis=0)

  crop_bbox = Bbox(corner, corner + 7000)
  crop_mesh = mesh.crop(crop_bbox)

  assert crop_mesh.vertices.size < mesh.vertices.size 
  assert crop_mesh.faces.size < mesh.faces.size 












