from functools import partial

import tempfile
import cloudvolume
import numpy as np
import shutil
import posixpath
import pytest
import os
from scipy import sparse 
import sys
import json
import re
tempdir = tempfile.mkdtemp()
TEST_PATH = "file://{}".format(tempdir)
TEST_DATASET_NAME = "testvol"
PRECOMPUTED_MESH_TEST_DATASET_NAME = "meshvol_precompute"
DRACO_MESH_TEST_DATASET_NAME = "meshvol_draco"
GRAPHENE_SHARDED_MESH_TEST_DATASET_NAME = "meshvol_graphene_sharded"
PCG_LOCATION = "https://www.dynamicannotationframework.com/"
TEST_SEG_ID = 144115188084020434
TEST_GRAPHENE_SHARDED_ID = 864691135213153080
TEST_TOKEN = "2371270ab23f129cc121dedbeef01294"

def test_graphene_auth_token(graphene_vol):
  cloudpath = "graphene://" + posixpath.join(PCG_LOCATION, 'segmentation', 'api/v1/', TEST_DATASET_NAME)
  
  cloudvolume.CloudVolume(cloudpath, secrets=TEST_TOKEN)
  cloudvolume.CloudVolume(cloudpath, secrets={ "token": TEST_TOKEN })

  try:
    cloudvolume.CloudVolume(cloudpath, secrets=None)
  except cloudvolume.exceptions.AuthenticationError:
    pass

  try:
    cloudvolume.CloudVolume(cloudpath, secrets={ "token": "Z@(ASINAFSOFAFOSNS" })
    assert False
  except cloudvolume.exceptions.AuthenticationError:
    pass

@pytest.fixture()
def cv_graphene_mesh_precomputed(requests_mock):
  test_dir = os.path.dirname(os.path.abspath(__file__))
  graphene_test_cv_dir = os.path.join(test_dir,'test_cv')
  graphene_test_cv_path = "file://{}".format(graphene_test_cv_dir)

  info_d = {
    "app": {
      "supported_api_versions": [
        0,
        1
      ]
    },
    "data_dir": graphene_test_cv_path,
    "data_type": "uint64",
    "graph": {
      "chunk_size": [
        512,
        512,
        128
      ],
      "n_layers": 9,
      "n_bits_for_layer_id": 8,
      "spatial_bit_masks": {
        x: 10 for x in range(255)
      },
    },
    "chunks_start_at_voxel_offset": False,
    "mesh": "mesh_mip_2_err_40_sv16",
    "num_channels": 1,
    "scales": [{
      "chunk_sizes": [[ 512, 512, 16 ]],
      "compressed_segmentation_block_size": [
        8,
        8,
        8
      ],
      "encoding": "compressed_segmentation",
      "key": "8_8_40",
      "resolution": [
        8,
        8,
        40
      ],
      "size": [
        43520,
        26112,
        2176
      ],
      "voxel_offset": [
        17920,
        14848,
        0
      ]
    }],
    "type": "segmentation"
  }
  requests_mock.get(posixpath.join(PCG_LOCATION,
                    'segmentation/table',
                    PRECOMPUTED_MESH_TEST_DATASET_NAME,
                    "info"), json=info_d)
  
  frag_files = os.listdir(os.path.join(graphene_test_cv_dir, info_d['mesh']))
  # the file are saved as .gz but we want to list the non gz version
  # as cloudvolume will take care of finding the compressed files
  frag_files = [f[:-3] for f in frag_files if f[0]=='9']
  frag_d = {'fragments':frag_files}
  mock_url = posixpath.join(PCG_LOCATION,
              "meshing/api/v1/table",
              PRECOMPUTED_MESH_TEST_DATASET_NAME,
              "manifest/{}:0?verify=True".format(TEST_SEG_ID))
  requests_mock.get(mock_url, json=frag_d)

  cloudpath =   posixpath.join(PCG_LOCATION,
                             'segmentation/table',
                             PRECOMPUTED_MESH_TEST_DATASET_NAME)
  yield cloudvolume.CloudVolume("graphene://" + cloudpath, secrets=TEST_TOKEN)

@pytest.fixture()
def cv_graphene_mesh_draco(requests_mock):
  test_dir = os.path.dirname(os.path.abspath(__file__))
  graphene_test_cv_dir = os.path.join(test_dir,'test_cv')
  graphene_test_cv_path = "file://{}".format(graphene_test_cv_dir)

  info_d = {
      "app": {
      "supported_api_versions": [
        0,
        1
      ]
    },
    "data_dir": graphene_test_cv_path,
    "data_type": "uint64",
    "graph": {
      "chunk_size": [
        512,
        512,
        128
      ],
      "n_layers": 9,
      "n_bits_for_layer_id": 8,
      "spatial_bit_masks": {
        x: 10 for x in range(255)
      },
    },
    "chunks_start_at_voxel_offset": False,
    "mesh": "mesh_mip_2_draco_sv16",
    "mesh_metadata": {
      "max_meshed_layer": 6,
      "uniform_draco_grid_size": 21
    },
    "num_channels": 1,
    "scales": [{
      "chunk_sizes": [[ 512, 512, 16 ]],
      "compressed_segmentation_block_size": [
        8,
        8,
        8
      ],
      "encoding": "compressed_segmentation",
      "key": "8_8_40",
      "resolution": [
        8,
        8,
        40
      ],
      "size": [
        43520,
        26112,
        2176
      ],
      "voxel_offset": [
        17920,
        14848,
        0
      ]
    }],
    "type": "segmentation"
  }
  infourl = posixpath.join(PCG_LOCATION,
                           'segmentation/table',
                           DRACO_MESH_TEST_DATASET_NAME,
                           "info")

  requests_mock.get(infourl, json=info_d)
  
  frag_files = os.listdir(os.path.join(graphene_test_cv_dir, info_d['mesh']))
  # we want to filter out the manifest file
  frag_files = [ f for f in frag_files if f[0] == '1' ]
  frag_d = { 'fragments': frag_files }
  mock_url = posixpath.join(
    PCG_LOCATION, 'meshing/api/v1/table', 
    DRACO_MESH_TEST_DATASET_NAME, 
    "manifest/{}:0?verify=True".format(TEST_SEG_ID)
  )
  requests_mock.get(mock_url, json=frag_d)
  cloudpath = posixpath.join(PCG_LOCATION,
                             'segmentation/table',
                             DRACO_MESH_TEST_DATASET_NAME)
  yield cloudvolume.CloudVolume('graphene://' + cloudpath, secrets=TEST_TOKEN)


@pytest.fixture()
def cv_graphene_sharded(requests_mock):
  test_dir = os.path.dirname(os.path.abspath(__file__))
  graphene_test_cv_dir = os.path.join(test_dir,'test_cv')
  graphene_test_cv_path = "gs://seunglab-test/graphene/meshes"

  with open(os.path.join(graphene_test_cv_dir, 'sharded_info.json'), 'r') as fp:
    info_d = json.load(fp)
  info_d['data_dir']=graphene_test_cv_path
  
  infourl = posixpath.join(PCG_LOCATION,
                           'segmentation/table',
                           GRAPHENE_SHARDED_MESH_TEST_DATASET_NAME,
                           "info")
  requests_mock.get(infourl, json=info_d)
  
  valid_manifest={
  "fragments": [
    "~6/29568-0.shard:765877565:4454",
    "~6/50112-0.shard:129695820:17794",
    "~7/3296-0.shard:727627771:13559",
    "~7/3264-2.shard:660015424:21225",
    "~7/6400-3.shard:478017968:31760",
    "~7/7424-2.shard:9298231:40730",
    "~7/4320-0.shard:13324264:53780",
    "~6/29568-0.shard:27890566:21061",
    "516154738544909386:0:40960-49152_57344-65536_0-16384"
  ],
  "seg_ids": [
    440473154180453586,
    446120245900606131,
    511651138917622208,
    511580770173215172,
    518476907102810232,
    520728706916532712,
    513902938730988744,
    440473154180397181,
    516154738544909386
  ]
}
  speculative_manifest = {
  "fragments": [
    "~440473154180453586:6:440473154180087808:29568-0.shard:929",
    "~511651138917622208:7:511651138915794944:3296-0.shard:481",
    "~520728706916532712:7:520728706914713600:7424-2.shard:824",
    "~518476907102810232:7:518476907101028352:6400-3.shard:699",
    "~511580770173215172:7:511580770171617280:3264-2.shard:745",
    "~513902938730988744:7:513902938729480192:4320-0.shard:170",
    "~446120245900606131:6:446120245900345344:50112-0.shard:629",
    "~440473154180397181:6:440473154180087808:29568-0.shard:38",
    "516154738544909386:0:40960-49152_57344-65536_0-16384"
  ],
  "seg_ids": [
    440473154180453586,
    511651138917622208,
    520728706916532712,
    518476907102810232,
    511580770173215172,
    513902938730988744,
    446120245900606131,
    440473154180397181,
    516154738544909386
  ]
}
  verify_manifest_url = posixpath.join(
    PCG_LOCATION, 'meshing/api/v1/table', 
    GRAPHENE_SHARDED_MESH_TEST_DATASET_NAME, 
    "manifest/{}:0?verify=True".format(TEST_GRAPHENE_SHARDED_ID)
  )
  speculative_manifest_url = posixpath.join(
    PCG_LOCATION, 'meshing/api/v1/table', 
    GRAPHENE_SHARDED_MESH_TEST_DATASET_NAME, 
    "manifest/{}:0?verify=False".format(TEST_GRAPHENE_SHARDED_ID)
  )

  requests_mock.get(verify_manifest_url, json=valid_manifest)
  requests_mock.get(speculative_manifest_url, json=speculative_manifest)
  matcher = re.compile('https://storage.googleapis.com/')

  requests_mock.get(matcher,real_http=True)
  cloudpath = posixpath.join(PCG_LOCATION, 'segmentation/table/', GRAPHENE_SHARDED_MESH_TEST_DATASET_NAME)
  yield cloudvolume.CloudVolume('graphene://' + cloudpath, use_https=True, secrets=TEST_TOKEN)


@pytest.fixture(scope='session')
def cv_supervoxels(N=64, blockN=16):

  block_per_row = int(N / blockN)

  chunk_size = [32, 32, 32]
  info = cloudvolume.CloudVolume.create_new_info(
    num_channels=1,
    layer_type='segmentation',
    data_type='uint64',
    encoding="compressed_segmentation",
    resolution=[4, 4, 40],  # Voxel scaling, units are in nanometers
    voxel_offset=[0, 0, 0],  # x,y,z offset in voxels from the origin
    # Pick a convenient size for your underlying chunk representation
    # Powers of two are recommended, doesn't need to cover image exactly
    chunk_size=chunk_size,  # units are voxels
    volume_size=[N, N, N],
  )

  vol = cloudvolume.CloudVolume(TEST_PATH, info=info)
  vol.commit_info()
  xx, yy, zz = np.meshgrid(*[np.arange(0, N) for cs in chunk_size])
  id_ind = (
    np.uint64(xx / blockN),
    np.uint64(yy / blockN),
    np.uint64(zz / blockN)
  )
  id_shape = (block_per_row, block_per_row, block_per_row)

  seg = np.ravel_multi_index(id_ind, id_shape)
  vol[:] = np.uint64(seg)

  yield TEST_PATH

  shutil.rmtree(tempdir)


@pytest.fixture()
def graphene_vol(cv_supervoxels,  requests_mock, monkeypatch, N=64):

  chunk_size = [32, 32, 32]

  info_d = {
    "data_dir": cv_supervoxels,
    "data_type": "uint64",
    "graph": {
      "chunk_size": [64, 64, 64],
      "bounding_box": [2048, 2048, 512], 
      "chunk_size": [256, 256, 512], 
      "cv_mip": 0, 
      "n_bits_for_layer_id": 8, 
      "n_layers": 12, 
      "spatial_bit_masks": {
        '1': 10, '2': 10, '3': 9, 
        '4': 8, '5': 7, '6': 6, 
        '7': 5, '8': 4, '9': 3, 
        '10': 2, '11': 1, '12': 1
      }
    },
    "app": { "supported_api_versions": [0, 1] },
    "mesh": "mesh_mip_2_err_40_sv16",
    "num_channels": 1,
    "scales": [
      {
        "chunk_sizes": [
          [32, 32, 32]
        ],
        "compressed_segmentation_block_size": [8, 8, 8],
        "encoding": "compressed_segmentation",
        "key": "4_4_40",
        "resolution": [4, 4, 40],
        "size": [N, N, N],
        "voxel_offset": [0, 0, 0]
      }
    ],
    "type": "segmentation"
  }

  infourl = posixpath.join(PCG_LOCATION, 'segmentation/table', TEST_DATASET_NAME, "info")
  requests_mock.get(infourl, json=info_d)
  
  def mock_get_leaves(self, root_id, bbox, mip):
    return np.array([0,1,2,3], dtype=np.uint64)

  cloudpath = "graphene://" + posixpath.join(PCG_LOCATION, 'segmentation', 'api/v1/', TEST_DATASET_NAME)

  gcv = cloudvolume.CloudVolume(cloudpath, secrets=TEST_TOKEN)
  gcv.get_leaves = partial(mock_get_leaves, gcv)
  yield gcv

def test_gcv(graphene_vol):
  cutout = graphene_vol.download(np.s_[0:5,0:5,0:5], segids=[999])
  assert (np.all(cutout==999))
  cutout_sv = graphene_vol[0:5,0:5,0:5]
  assert cutout_sv.shape == (5,5,5,1)
  assert graphene_vol[0,0,0].shape == (1,1,1,1)


def test_get_roots(graphene_vol):
  roots = graphene_vol.get_roots([])
  assert np.all(roots == [])

  segids = [0, 0, 0, 0, 0]
  roots = graphene_vol.get_roots(segids)
  assert np.all(roots == segids)

  segids = [0, 864691135462849854, 864691135462849854, 0]
  roots = graphene_vol.get_roots(segids)
  assert np.all(roots == segids)

def faces_to_edges(faces, return_index=False):
  """
  Given a list of faces (n,3), return a list of edges (n*3,2)
  Parameters
  -----------
  faces : (n, 3) int
    Vertex indices representing faces
  Returns
  -----------
  edges : (n*3, 2) int
    Vertex indices representing edges
  """
  faces = np.asanyarray(faces)

  # each face has three edges
  edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2))

  if return_index:
    # edges are in order of faces due to reshape
    face_index = np.tile(np.arange(len(faces)),
               (3, 1)).T.reshape(-1)
    return edges, face_index
  return edges

  
def create_csgraph(vertices, edges, euclidean_weight=True, directed=False):
  '''
  Builds a csr graph from vertices and edges, with optional control
  over weights as boolean or based on Euclidean distance.
  '''
  if euclidean_weight:
    xs = vertices[edges[:,0]]
    ys = vertices[edges[:,1]]
    weights = np.linalg.norm(xs-ys, axis=1)
    use_dtype = np.float32
  else:   
    weights = np.ones((len(edges),)).astype(np.int8)
    use_dtype = np.int8 

  if directed:
    edges = edges.T
  else:
    edges = np.concatenate([edges.T, edges.T[[1, 0]]], axis=1)
    weights = np.concatenate([weights, weights]).astype(dtype=use_dtype)

  csgraph = sparse.csr_matrix((weights, edges),
                shape=[len(vertices), ] * 2,
                dtype=use_dtype)

  return csgraph

@pytest.mark.skipif(sys.version_info < (3, 0), reason="requires python3 or higher")
def test_decode_segid(cv_graphene_mesh_draco):
  decoded = cv_graphene_mesh_draco.meta.decode_label(648518346349515986)
  assert decoded.level == 9
  assert decoded.x == 0
  assert decoded.y == 0
  assert decoded.z == 0
  assert decoded.segid == 8164562

  level = '00000101' # 5
  x = '0000000001' # 1 
  y = '0000000010' # 2 
  z = '0000000011' # 3
  segid = '00000000000000000000001010' # 10

  label = int(level + x + y + z + segid, 2)
  decoded = cv_graphene_mesh_draco.meta.decode_label(label)
  
  assert decoded.level == 5
  assert decoded.x == 1
  assert decoded.y == 2
  assert decoded.z == 3
  assert decoded.segid == 10

  encoded = cv_graphene_mesh_draco.meta.encode_label(*decoded)
  assert decoded == cv_graphene_mesh_draco.meta.decode_label(encoded)


@pytest.mark.skipif(sys.version_info < (3, 0), reason="requires python3 or higher")
def test_graphene_mesh_get_precomputed(cv_graphene_mesh_precomputed):

  mesh = cv_graphene_mesh_precomputed.mesh.get(TEST_SEG_ID)
  edges = faces_to_edges(mesh[TEST_SEG_ID].faces)
  graph = create_csgraph(mesh[TEST_SEG_ID].vertices,
               edges,
               directed=False)
  ccs, labels =  sparse.csgraph.connected_components(graph,
                             directed=False)
  assert(ccs==3)

@pytest.mark.skipif(sys.version_info < (3, 0), reason="requires python3 or higher")
def test_graphene_mesh_get_draco(cv_graphene_mesh_draco):

  mesh = cv_graphene_mesh_draco.mesh.get(TEST_SEG_ID)
  edges = faces_to_edges(mesh[TEST_SEG_ID].faces)
  graph = create_csgraph(mesh[TEST_SEG_ID].vertices,
               edges,
               directed=False)
  ccs, labels =  sparse.csgraph.connected_components(graph,
                             directed=False)
  assert(ccs==3)

@pytest.mark.skipif(sys.version_info < (3, 0), reason="requires python3 or higher")
def test_graphene_mesh_get_graphene_sharded(cv_graphene_sharded):

  mesh = cv_graphene_sharded.mesh.get(TEST_GRAPHENE_SHARDED_ID)
  edges = faces_to_edges(mesh[TEST_GRAPHENE_SHARDED_ID].faces)
  graph = create_csgraph(mesh[TEST_GRAPHENE_SHARDED_ID].vertices,
               edges,
               directed=False)
  ccs, labels =  sparse.csgraph.connected_components(graph,
                             directed=False)
  assert(ccs==21)


