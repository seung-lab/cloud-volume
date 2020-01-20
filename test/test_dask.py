import os
import pytest
import sys

import numpy as np

from cloudvolume import dask as dasklib

@pytest.mark.skipif(sys.version_info[0] < 3, reason="Python 2 not supported.")
def test_roundtrip_3d():
  da = pytest.importorskip('dask.array')
  du = pytest.importorskip('dask.utils')
  a = da.zeros((3, 3, 3), chunks=1)
  with du.tmpdir() as d:
    d = 'file://' + d
    dasklib.to_cloudvolume(a, d)
    a2 = dasklib.from_cloudvolume(d)
    a2 = da.squeeze(a2, axis=3)
    da.utils.assert_eq(a, a2, check_meta=False)  # TODO: add meta check
    assert a.chunks == a2.chunks

@pytest.mark.skipif(sys.version_info[0] < 3, reason="Python 2 not supported.")
def test_roundtrip_4d():
  da = pytest.importorskip('dask.array')
  du = pytest.importorskip('dask.utils')
  a = da.zeros((3, 3, 3, 3))
  with du.tmpdir() as d:
    d = 'file://' + d
    dasklib.to_cloudvolume(a, d)
    a2 = dasklib.from_cloudvolume(d)
    da.utils.assert_eq(a, a2, check_meta=False)  # TODO: add meta check
    assert a.chunks == a2.chunks

@pytest.mark.skipif(sys.version_info[0] < 3, reason="Python 2 not supported.")
def test_roundtrip_4d_channel_rechunked():
  da = pytest.importorskip('dask.array')
  du = pytest.importorskip('dask.utils')
  a = da.zeros((3, 3, 3, 3), chunks=(2, 2, 2, 3))
  with du.tmpdir() as d:
    d = 'file://' + d
    dasklib.to_cloudvolume(a, d)
    a2 = dasklib.from_cloudvolume(d)
    np.testing.assert_array_equal(a.compute(), a2.compute())
    # Channel has single chunk
    assert a2.chunks == ((2, 1), (2, 1), (2, 1), (3, ))

@pytest.mark.skipif(sys.version_info[0] < 3, reason="Python 2 not supported.")
def test_roundtrip_4d_1channel():
  da = pytest.importorskip('dask.array')
  du = pytest.importorskip('dask.utils')
  a = da.zeros((3, 3, 3, 1))
  with du.tmpdir() as d:
    d = 'file://' + d
    dasklib.to_cloudvolume(a, d)
    a2 = dasklib.from_cloudvolume(d)
    da.utils.assert_eq(a, a2, check_meta=False)  # TODO: add meta check
    assert a.chunks == a2.chunks

@pytest.mark.skipif(sys.version_info[0] < 3, reason="Python 2 not supported.")
def test_roundtrip_rechunk_3d():
  da = pytest.importorskip('dask.array')
  du = pytest.importorskip('dask.utils')
  a = da.zeros((9, 9, 9), chunks=3)
  with du.tmpdir() as d:
    d = 'file://' + d
    dasklib.to_cloudvolume(a, d)
    a2 = dasklib.from_cloudvolume(d, chunks=5)
    assert a2.chunks == ((5, 4), (5, 4), (5, 4), (1, ))

@pytest.mark.skipif(sys.version_info[0] < 3, reason="Python 2 not supported.")
def test_roundtrip_rechunk_4d():
  da = pytest.importorskip('dask.array')
  du = pytest.importorskip('dask.utils')
  a = da.zeros((9, 9, 9, 3), chunks=3)
  with du.tmpdir() as d:
    d = 'file://' + d
    dasklib.to_cloudvolume(a, d)
    a2 = dasklib.from_cloudvolume(d, chunks=5)
    assert a2.chunks == ((5, 4), (5, 4), (5, 4), (3, ))

@pytest.mark.skipif(sys.version_info[0] < 3, reason="Python 2 not supported.")
def test_delayed_compute():
  dask = pytest.importorskip('dask')
  da = pytest.importorskip('dask.array')
  dd = pytest.importorskip('dask.delayed')
  du = pytest.importorskip('dask.utils')
  a = da.zeros((3, 3, 3, 1), chunks=1)
  with du.tmpdir() as d:
    d = 'file://' + d
    out = dasklib.to_cloudvolume(a, d, compute=False)
    assert isinstance(out, dd.Delayed)
    dask.compute(out)
    a2 = dasklib.from_cloudvolume(d)
    da.utils.assert_eq(a, a2, check_meta=False)  # TODO: add meta check
    assert a.chunks == a2.chunks

@pytest.mark.skipif(sys.version_info[0] < 3, reason="Python 2 not supported.")
def test_roundtrip_delayed():
  da = pytest.importorskip('dask.array')
  du = pytest.importorskip('dask.utils')
  a = da.zeros((3, 3, 3, 3))
  with du.tmpdir() as d:
    cloudpath = 'file://' + d
    task = dasklib.to_cloudvolume(a, cloudpath, compute=False)
    assert not os.listdir(d)
    task.compute()
    a2 = dasklib.from_cloudvolume(cloudpath)
    da.utils.assert_eq(a, a2, check_meta=False)  # TODO: add meta check
    assert a.chunks == a2.chunks
