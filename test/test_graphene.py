from functools import partial

import tempfile
import cloudvolume
import numpy as np
import shutil
import pytest
import os
from meshparty import trimesh_io
from scipy import sparse 

tempdir = tempfile.mkdtemp()
TEST_PATH = "file:/{}".format(tempdir)
TEST_DATASET_NAME = "testvol"
MESH_TEST_DATASET_NAME = "meshvol"
PCG_LOCATION = "http://localhost/segmentation/1.0/"
PCG_MESH_LOCATION = "http://localhost./meshing/1.0/"
TEST_SEG_ID = 648518346349515986

@pytest.fixture()
def cv_graphene_mesh_precomputed(requests_mock):
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_cv_dir = os.path.join(test_dir,'test_cv')
    test_cv_path = "file://{}".format(test_cv_dir)

    info_d={
        "data_dir": test_cv_path,
        "data_type": "uint64",
        "graph": {
        "chunk_size": [
        512,
        512,
        128
        ]
        },
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
    requests_mock.get(PCG_LOCATION+MESH_TEST_DATASET_NAME+"/info", json=info_d)
    requests_mock.get(PCG_LOCATION+MESH_TEST_DATASET_NAME+"/info/", json=info_d)
    frag_files = os.listdir(os.path.join(test_cv_dir, info_d['mesh']))
    frag_files = [f[:-3] for f in frag_files if f[0]=='9']
    frag_d = {'fragments':frag_files}
    mock_url = PCG_MESH_LOCATION + MESH_TEST_DATASET_NAME+f"/manifest/{TEST_SEG_ID}:0?verify=True"
    requests_mock.get(mock_url,
                      json=frag_d)
    print(mock_url)
    gcv = cloudvolume.CloudVolume(
        "graphene://{}{}".format(PCG_LOCATION, MESH_TEST_DATASET_NAME)
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
    requests_mock.get(PCG_LOCATION+TEST_DATASET_NAME+"/info/")
    
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

def test_graphene_mesh_get(cv_graphene_mesh_precomputed):

    mesh = cv_graphene_mesh_precomputed.mesh.get(TEST_SEG_ID)
    tmesh = trimesh_io.Mesh(mesh[TEST_SEG_ID].vertices, mesh[TEST_SEG_ID].faces)
    ccs, labels =  sparse.csgraph.connected_components(tmesh.csgraph,
                                                       directed=False)
    assert(ccs==3)

