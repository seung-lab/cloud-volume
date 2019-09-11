from functools import partial

import tempfile
import cloudvolume
import numpy as np
import shutil
import pytest

tempdir = tempfile.mkdtemp()
TEST_PATH = "file:/{}".format(tempdir)
TEST_DATASET_NAME = "testvol"
PCG_LOCATION = "http://localhost/segmentation/1.0/"

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

    