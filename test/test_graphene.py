from functools import partial

import tempfile
import cloudvolume
import numpy as np
import shutil
import pytest
import os
from scipy import sparse 
import sys

tempdir = tempfile.mkdtemp()
TEST_PATH = "file://{}".format(tempdir)
TEST_DATASET_NAME = "testvol"
PRECOMPUTED_MESH_TEST_DATASET_NAME = "meshvol_precompute"
DRACO_MESH_TEST_DATASET_NAME = "meshvol_draco"
PCG_LOCATION = "https://www.dynamicannotationframework.com/segmentation/1.0/"
PCG_MESH_LOCATION = "https://www.dynamicannotationframework.com/meshing/1.0/"
TEST_SEG_ID = 648518346349515986

@pytest.fixture()
def cv_graphene_mesh_precomputed(requests_mock):
    test_dir = os.path.dirname(os.path.abspath(__file__))
    graphene_test_cv_dir = os.path.join(test_dir,'test_cv')
    graphene_test_cv_path = "file://{}".format(graphene_test_cv_dir)

    info_d={
        "data_dir": graphene_test_cv_path,
        "data_type": "uint64",
        "graph": {
        "chunk_size": [
        512,
        512,
        128
        ]
        },
        "chunks_start_at_voxel_offset": False,
        "mesh": "mesh_mip_2_err_40_sv16",
        "num_channels": 1,
        "scales": [
        {
        "chunk_sizes": [
            [
            512,
            512,
            16
            ]
        ],
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
    requests_mock.get(PCG_LOCATION+PRECOMPUTED_MESH_TEST_DATASET_NAME+"/info", json=info_d)
  
    frag_files = os.listdir(os.path.join(graphene_test_cv_dir, info_d['mesh']))
    # the file are saved as .gz but we want to list the non gz version
    # as cloudvolume will take care of finding the compressed files
    frag_files = [f[:-3] for f in frag_files if f[0]=='9']
    frag_d = {'fragments':frag_files}
    mock_url = PCG_MESH_LOCATION + PRECOMPUTED_MESH_TEST_DATASET_NAME+"/manifest/{}:0?verify=True".format(TEST_SEG_ID)
    requests_mock.get(mock_url,
                      json=frag_d)
    print(mock_url)
    gcv = cloudvolume.CloudVolume(
        "graphene://{}{}".format(PCG_LOCATION, PRECOMPUTED_MESH_TEST_DATASET_NAME)
    )

    yield gcv

@pytest.fixture()
def cv_graphene_mesh_draco(requests_mock):
    test_dir = os.path.dirname(os.path.abspath(__file__))
    graphene_test_cv_dir = os.path.join(test_dir,'test_cv')
    graphene_test_cv_path = "file://{}".format(graphene_test_cv_dir)

    info_d={
        "data_dir": graphene_test_cv_path,
        "data_type": "uint64",
        "graph": {
        "chunk_size": [
        512,
        512,
        128
        ]
        },
        "chunks_start_at_voxel_offset": False,
        "mesh": "mesh_mip_2_draco_sv16",
        "mesh_metadata": {
            "max_meshed_layer": 6,
            "uniform_draco_grid_size": 21
        },
        "num_channels": 1,
        "scales": [
        {
        "chunk_sizes": [
            [
            512,
            512,
            16
            ]
        ],
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
    requests_mock.get(PCG_LOCATION+DRACO_MESH_TEST_DATASET_NAME+"/info", json=info_d)
  
    frag_files = os.listdir(os.path.join(graphene_test_cv_dir, info_d['mesh']))
    # we want to filter out the manifest file
    frag_files = [f for f in frag_files if f[0]=='1']
    frag_d = {'fragments':frag_files}
    mock_url = PCG_MESH_LOCATION + DRACO_MESH_TEST_DATASET_NAME+"/manifest/{}:0?verify=True".format(TEST_SEG_ID)
    requests_mock.get(mock_url,
                      json=frag_d)
    print(mock_url)
    gcv = cloudvolume.CloudVolume(
        "graphene://{}{}".format(PCG_LOCATION, DRACO_MESH_TEST_DATASET_NAME)
    )

    yield gcv

@pytest.fixture(scope='session')
def cv_supervoxels(N=64, blockN=16):

    block_per_row = int(N / blockN)

    chunk_size = [32, 32, 32]
    info = cloudvolume.CloudVolume.create_new_info(
        num_channels=1,
        layer_type='segmentation',
        data_type='uint64',
        encoding='raw',
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
            "chunk_size": [64, 64, 64]
        },
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

    requests_mock.get(PCG_LOCATION+TEST_DATASET_NAME+"/info", json=info_d)
    
    def mock_get_leaves(self, root_id, bbox, mip):
        return np.array([0,1,2,3], dtype=np.uint64)

    gcv = cloudvolume.CloudVolume(
        "graphene://{}{}".format(PCG_LOCATION, TEST_DATASET_NAME)
    )
    gcv.get_leaves = partial(mock_get_leaves, gcv)
    yield gcv

def test_gcv(graphene_vol):
    cutout = graphene_vol.download(np.s_[0:5,0:5,0:5], segids=[999])
    assert (np.all(cutout==999))
    cutout_sv = graphene_vol[0:5,0:5,0:5]
    assert cutout_sv.shape == (5,5,5,1)
    assert graphene_vol[0,0,0].shape == (1,1,1,1)


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
def test_graphene_mesh_get(cv_graphene_mesh_precomputed):

    mesh = cv_graphene_mesh_precomputed.mesh.get(TEST_SEG_ID)
    edges = faces_to_edges(mesh[TEST_SEG_ID].faces)
    graph = create_csgraph(mesh[TEST_SEG_ID].vertices,
                           edges,
                           directed=False)
    ccs, labels =  sparse.csgraph.connected_components(graph,
                                                       directed=False)
    assert(ccs==3)

@pytest.mark.skipif(sys.version_info < (3, 0), reason="requires python3 or higher")
def test_graphene_mesh_get(cv_graphene_mesh_draco):

    mesh = cv_graphene_mesh_draco.mesh.get(TEST_SEG_ID)
    edges = faces_to_edges(mesh[TEST_SEG_ID].faces)
    graph = create_csgraph(mesh[TEST_SEG_ID].vertices,
                           edges,
                           directed=False)
    ccs, labels =  sparse.csgraph.connected_components(graph,
                                                       directed=False)
    assert(ccs==3)

