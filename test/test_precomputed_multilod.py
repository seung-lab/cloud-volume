import pytest 
import sys

import cloudvolume

# Basic test of sharded meshes
# The following test files are used from https://storage.googleapis.com:
#   /neuroglancer-janelia-flyem-hemibrain/v1.0/segmentation/info
#   /fafb-ffn1-20190805/segmentation/mesh/info
#   /neuroglancer-janelia-flyem-hemibrain/v1.0/segmentation/mesh/171.shard
#   /neuroglancer-janelia-flyem-hemibrain/v1.0/segmentation/mesh/193.shard
#   /neuroglancer-janelia-flyem-hemibrain/v1.0/segmentation/mesh/185.shard 


@pytest.mark.skipif(sys.version_info < (3, 0), reason="requires python3")
def test_get_sharded_mesh():
  vol = cloudvolume.CloudVolume('gs://neuroglancer-janelia-flyem-hemibrain/v1.0/segmentation', mip=0, cache=False, use_https=True)

  exists = vol.mesh.exists([511271574, 360284300])
  assert(all(exists))

  exists = vol.mesh.exists([666666666, 666666667, 666666668])
  assert(all([a == None for a in exists]))

  meshes = vol.mesh.get([511271574, 360284300], lod=2)
  assert len(meshes[511271574].faces) == 258647

  meshes = vol.mesh.get([511271574, 360284300], lod=3)
  assert len(meshes[511271574].faces) == 50501
