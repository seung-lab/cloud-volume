import pytest

import os

import numpy as np

import cloudvolume

@pytest.fixture
def asrc():
  test_dir = os.path.dirname(os.path.abspath(__file__))
  test_dir = os.path.join(test_dir, "test_precomputed_annotation")
  return cloudvolume.from_cloudpath("file://" + test_dir)


def test_annotation(asrc):
	assert asrc.meta.info["@type"] == "neuroglancer_annotations_v1"

	elem = asrc.get_by_id(3867588737)
	ans = np.array([[1.937760e+06, 1.318752e+06, 9.692100e+04]], dtype=np.float32)
	assert np.all(np.isclose(elem.geometry, ans))

	elem = asrc.get_by_relationship("skeleton_id", 243895108)
	assert elem.geometry.shape[0] == 4

	pd = elem.pandas()
	assert set(pd["class_label"]) == {'axon'}

	all_pts = asrc.get_all(mip=0)
	assert str(all_pts.type) == "POINT"
	assert all_pts.geometry.shape == (10043,3)

	pd = all_pts.pandas()
	assert len(set(pd["class_label"])) == 7