import pytest 
import sys

import numpy as np

import cloudvolume

# Basic test of sharded meshes
# The following test files are used from https://storage.googleapis.com:
#   /neuroglancer-janelia-flyem-hemibrain/v1.0/segmentation/info
#   /fafb-ffn1-20190805/segmentation/mesh/info
#   /neuroglancer-janelia-flyem-hemibrain/v1.0/segmentation/mesh/171.shard
#   /neuroglancer-janelia-flyem-hemibrain/v1.0/segmentation/mesh/193.shard
#   /neuroglancer-janelia-flyem-hemibrain/v1.0/segmentation/mesh/185.shard 

@pytest.fixture
def hemibrain_vol():
  return cloudvolume.CloudVolume('gs://neuroglancer-janelia-flyem-hemibrain/v1.0/segmentation', mip=0, cache=False, use_https=True)

@pytest.mark.skipif(sys.version_info < (3, 0), reason="requires python3")
def test_get_sharded_mesh(hemibrain_vol):
  exists = hemibrain_vol.mesh.exists([511271574, 360284300])
  assert(all(exists))

  exists = hemibrain_vol.mesh.exists([666666666, 666666667, 666666668])
  assert(all([a == None for a in exists]))

  meshes = hemibrain_vol.mesh.get([511271574, 360284300], lod=2)
  assert len(meshes[511271574].faces) == 258647

  meshes = hemibrain_vol.mesh.get([511271574, 360284300], lod=3)
  assert len(meshes[511271574].faces) == 50501

  mesh = meshes[511271574]
  
  hemibrain_vol.mesh.transform[:,3] = 1

  double_mesh = hemibrain_vol.mesh.get(511271574, lod=3)[511271574]
  assert (mesh.vertices + 1 == double_mesh.vertices).all()

@pytest.mark.skipif(sys.version_info < (3, 0), reason="requires python3")
def test_get_sharded_mesh_invalid_lod(hemibrain_vol):
  try:
    meshes = hemibrain_vol.mesh.get([511271574, 360284300], lod=8)
  except ValueError:
    pass
  except:
    raise
