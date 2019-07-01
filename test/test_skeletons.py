import pytest

import copy
import gzip
import json
import math
import numpy as np
import os
import shutil

from cloudvolume import CloudVolume, chunks, Storage, PrecomputedSkeleton
from cloudvolume.storage import SimpleStorage
from cloudvolume.lib import mkdir, Bbox, Vec

from cloudvolume.exceptions import SkeletonDecodeError

info = CloudVolume.create_new_info(
  num_channels=1, # Increase this number when we add more tests for RGB
  layer_type='segmentation', 
  data_type='uint16', 
  encoding='raw',
  resolution=[1,1,1], 
  voxel_offset=(0,0,0), 
  skeletons=True,
  volume_size=(100, 100, 100),
  chunk_size=(64, 64, 64),
)

def test_skeletons():
  
  # Skeleton of my initials
  # z=0: W ; z=1 S
  vertices = np.array([
    [ 0, 1, 0 ],
    [ 1, 0, 0 ],
    [ 1, 1, 0 ],

    [ 2, 0, 0 ],
    [ 2, 1, 0 ],
    [ 0, 0, 1 ],

    [ 1, 0, 1 ],
    [ 1, 1, 1 ],
    [ 0, 1, 1 ],

    [ 0, 2, 1 ],
    [ 1, 2, 1 ],
  ], np.float32)

  edges = np.array([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [8, 9],
    [9, 10],
    [10, 11],
  ], dtype=np.uint32)

  radii = np.array([
    1.0,
    2.5,
    3.0,
    4.1,
    1.2,
    5.6,
    2000.123123,
    15.33332221,
    8128.124,
    -1,
    1824.03
  ], dtype=np.float32)

  vertex_types = np.array([
   1,
   2,
   3,
   5,
   8,
   2,
   0,
   5,
   9,
   11,
   22,
  ], dtype=np.uint8)

  vol = CloudVolume('file:///tmp/cloudvolume/test-skeletons', info=info)
  vol.skeleton.upload_raw(
    segid=1, vertices=vertices, edges=edges, 
    radii=radii, vertex_types=vertex_types
  )
  skel = vol.skeleton.get(1)

  assert skel.id == 1
  assert np.all(skel.vertices == vertices)
  assert np.all(skel.edges == edges)
  assert np.all(skel.radii == radii)
  assert np.all(skel.vertex_types == vertex_types)
  assert vol.skeleton.path == 'skeletons'
  assert not skel.empty()

  with SimpleStorage('file:///tmp/cloudvolume/test-skeletons/') as stor:
    rawskel = stor.get_file('skeletons/1')
    assert len(rawskel) == 8 + 11 * (12 + 8 + 4 + 1) 
    stor.delete_file('skeletons/1')
  
  try:
    vol.skeleton.get(5)
    assert False
  except SkeletonDecodeError:
    pass

def test_no_edges():
  vertices = np.array([
    [ 0, 1, 0 ],
    [ 1, 0, 0 ],
  ], np.float32)

  edges = None
  vol = CloudVolume('file:///tmp/cloudvolume/test-skeletons', info=info)
  vol.skeleton.upload_raw(2, vertices, edges)
  skel = vol.skeleton.get(2)

  assert skel.id == 2
  assert np.all(skel.vertices == vertices)
  assert np.all(skel.edges.shape == (0, 2))
  assert vol.skeleton.path == 'skeletons'
  assert skel.empty()

  with SimpleStorage('file:///tmp/cloudvolume/test-skeletons/') as stor:
    rawskel = stor.get_file('skeletons/2')
    assert len(rawskel) == 8 + 2 * (12 + 0 + 4 + 1) 
    stor.delete_file('skeletons/2')

def test_no_vertices():
  vertices = np.array([], np.float32).reshape(0,3)

  edges = None
  vol = CloudVolume('file:///tmp/cloudvolume/test-skeletons', info=info)
  vol.skeleton.upload_raw(3, vertices, edges)
  skel = vol.skeleton.get(3)

  assert skel.id == 3
  assert np.all(skel.vertices == vertices)
  assert np.all(skel.edges.shape == (0, 2))
  assert skel.empty()
  assert vol.skeleton.path == 'skeletons'

  with SimpleStorage('file:///tmp/cloudvolume/test-skeletons/') as stor:
    rawskel = stor.get_file('skeletons/3')
    assert len(rawskel) == 8 + 0 * (12 + 8 + 4 + 1)
    stor.delete_file('skeletons/3')

def test_consolidate():
  skel = PrecomputedSkeleton(
    vertices=np.array([
      (0, 0, 0),
      (1, 0, 0),
      (2, 0, 0),
      (0, 0, 0),
      (2, 1, 0),
      (2, 2, 0),
      (2, 2, 1),
      (2, 2, 2),
    ], dtype=np.float32),

    edges=np.array([
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
      [5, 6],
      [6, 7],
    ], dtype=np.uint32),

    radii=np.array([
      0, 1, 2, 3, 4, 5, 6, 7
    ], dtype=np.float32),

    vertex_types=np.array([
      0, 1, 2, 3, 4, 5, 6, 7
    ], dtype=np.uint8),
  )

  correct_skel = PrecomputedSkeleton(
    vertices=np.array([
      (0, 0, 0),
      (1, 0, 0),
      (2, 0, 0),
      (2, 1, 0),
      (2, 2, 0),
      (2, 2, 1),
      (2, 2, 2),
    ], dtype=np.float32),

    edges=np.array([
      [0, 1],
      [0, 2],
      [0, 3],
      [1, 2],
      [3, 4],
      [4, 5],
      [5, 6],
    ], dtype=np.uint32),

    radii=np.array([
      0, 1, 2, 4, 5, 6, 7
    ], dtype=np.float32),

    vertex_types=np.array([
      0, 1, 2, 4, 5, 6, 7
    ], dtype=np.uint8),
  )

  consolidated = skel.consolidate()

  assert np.all(consolidated.vertices == correct_skel.vertices)
  assert np.all(consolidated.edges == correct_skel.edges)
  assert np.all(consolidated.radii == correct_skel.radii)
  assert np.all(consolidated.vertex_types == correct_skel.vertex_types)

def test_equivalent():
  assert PrecomputedSkeleton.equivalent(PrecomputedSkeleton(), PrecomputedSkeleton())

  identity = PrecomputedSkeleton([ (0,0,0), (1,0,0) ], [(0,1)] )
  assert PrecomputedSkeleton.equivalent(identity, identity)

  diffvertex = PrecomputedSkeleton([ (0,0,0), (0,1,0) ], [(0,1)])
  assert not PrecomputedSkeleton.equivalent(identity, diffvertex)

  single1 = PrecomputedSkeleton([ (0,0,0), (1,0,0) ], edges=[ (1,0) ])
  single2 = PrecomputedSkeleton([ (0,0,0), (1,0,0) ], edges=[ (0,1) ])
  assert PrecomputedSkeleton.equivalent(single1, single2)

  double1 = PrecomputedSkeleton([ (0,0,0), (1,0,0) ], edges=[ (1,0) ])
  double2 = PrecomputedSkeleton([ (0,0,0), (1,0,0) ], edges=[ (0,1) ])
  assert PrecomputedSkeleton.equivalent(double1, double2)

  double1 = PrecomputedSkeleton([ (0,0,0), (1,0,0), (1,1,0) ], edges=[ (1,0), (1,2) ])
  double2 = PrecomputedSkeleton([ (0,0,0), (1,0,0), (1,1,0) ], edges=[ (2,1), (0,1) ])
  assert PrecomputedSkeleton.equivalent(double1, double2)

  double1 = PrecomputedSkeleton([ (0,0,0), (1,0,0), (1,1,0), (1,1,3) ], edges=[ (1,0), (1,2), (1,3) ])
  double2 = PrecomputedSkeleton([ (0,0,0), (1,0,0), (1,1,0), (1,1,3) ], edges=[ (3,1), (2,1), (0,1) ])
  assert PrecomputedSkeleton.equivalent(double1, double2)

def test_cable_length():
  skel = PrecomputedSkeleton([ 
      (0,0,0), (1,0,0), (2,0,0), (3,0,0), (4,0,0), (5,0,0)
    ], 
    edges=[ (1,0), (1,2), (2,3), (3,4), (5,4) ],
    radii=[ 1, 2, 3, 4, 5, 6 ],
    vertex_types=[1, 2, 3, 4, 5, 6]
  )

  assert skel.cable_length() == (skel.vertices.shape[0] - 1)

  skel = PrecomputedSkeleton([ 
      (2,0,0), (1,0,0), (0,0,0), (0,5,0), (0,6,0), (0,7,0)
    ], 
    edges=[ (1,0), (1,2), (2,3), (3,4), (5,4) ],
    radii=[ 1, 2, 3, 4, 5, 6 ],
    vertex_types=[1, 2, 3, 4, 5, 6]
  )
  assert skel.cable_length() == 9

  skel = PrecomputedSkeleton([ 
      (1,1,1), (0,0,0), (1,0,0)
    ], 
    edges=[ (1,0), (1,2) ],
    radii=[ 1, 2, 3],
    vertex_types=[1, 2, 3]
  )
  assert abs(skel.cable_length() - (math.sqrt(3) + 1)) < 1e-6

def test_downsample():
  skel = PrecomputedSkeleton([ 
      (0,0,0), (1,0,0), (1,1,0), (1,1,3), (2,1,3), (2,2,3)
    ], 
    edges=[ (1,0), (1,2), (2,3), (3,4), (5,4) ],
    radii=[ 1, 2, 3, 4, 5, 6 ],
    vertex_types=[1, 2, 3, 4, 5, 6],
    segid=1337,
  )

  def should_error(x):
    try:
      skel.downsample(x)
      assert False
    except ValueError:
      pass

  should_error(-1)
  should_error(0)
  should_error(.5)
  should_error(2.00000000000001)

  dskel = skel.downsample(1)
  assert PrecomputedSkeleton.equivalent(dskel, skel)
  assert dskel.id == skel.id
  assert dskel.id == 1337

  dskel = skel.downsample(2)
  dskel_gt = PrecomputedSkeleton(
    [ (0,0,0), (1,1,0), (2,1,3), (2,2,3) ], 
    edges=[ (1,0), (1,2), (2,3) ],
    radii=[1,3,5,6], vertex_types=[1,3,5,6] 
  )
  assert PrecomputedSkeleton.equivalent(dskel, dskel_gt)

  dskel = skel.downsample(3)
  dskel_gt = PrecomputedSkeleton(
    [ (0,0,0), (1,1,3), (2,2,3) ], edges=[ (1,0), (1,2) ],
    radii=[1,4,6], vertex_types=[1,4,6],
  )
  assert PrecomputedSkeleton.equivalent(dskel, dskel_gt)

  skel = PrecomputedSkeleton([ 
      (0,0,0), (1,0,0), (1,1,0), (1,1,3), (2,1,3), (2,2,3)
    ], 
    edges=[ (1,0), (1,2), (3,4), (5,4) ],
    radii=[ 1, 2, 3, 4, 5, 6 ],
    vertex_types=[1, 2, 3, 4, 5, 6]
  )
  dskel = skel.downsample(2)
  dskel_gt = PrecomputedSkeleton(
    [ (0,0,0), (1,1,0), (1,1,3), (2,2,3) ], 
    edges=[ (1,0), (2,3) ],
    radii=[1,3,4,6], vertex_types=[1,3,4,6] 
  )
  assert PrecomputedSkeleton.equivalent(dskel, dskel_gt)


def test_downsample_joints():
  skel = PrecomputedSkeleton([ 
      
                        (2, 3,0), # 0
                        (2, 2,0), # 1
                        (2, 1,0), # 2
      (0,0,0), (1,0,0), (2, 0,0), (3,0,0), (4,0,0), # 3, 4, 5, 6, 7
                        (2,-1,0), # 8
                        (2,-2,0), # 9
                        (2,-3,0), # 10

    ], 
    edges=[ 
                  (0, 1),
                  (1, 2),
                  (2, 5),
        (3,4), (4,5), (5, 6), (6,7),
                  (5, 8),
                  (8, 9),
                  (9,10)
    ],
    radii=[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ],
    vertex_types=[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ],
    segid=1337,
  )

  ds_skel = skel.downsample(2)
  ds_skel_gt = PrecomputedSkeleton([ 

                        (2, 3,0), # 0
                        
                        (2, 2,0), # 1
      (0,0,0),          (2, 0,0),     (4,0,0), # 2, 3, 4

                        (2,-2,0), # 5                        
                        (2,-3,0), # 6

    ], 
    edges=[ 
                  (0,1),
                  (1,3),
              (2,3),  (3,4), 
                  (3,5),
                  (5,6)
    ],
    radii=[ 0, 1, 3, 5, 7, 9, 10 ],
    vertex_types=[ 0, 1, 3, 5, 7, 9, 10 ],
    segid=1337,
  )

  assert PrecomputedSkeleton.equivalent(ds_skel, ds_skel_gt)


def test_read_swc():

  # From http://research.mssm.edu/cnic/swc.html
  test_file = """# ORIGINAL_SOURCE NeuronStudio 0.8.80
# CREATURE
# REGION
# FIELD/LAYER
# TYPE
# CONTRIBUTOR
# REFERENCE
# RAW
# EXTRAS
# SOMA_AREA
# SHINKAGE_CORRECTION 1.0 1.0 1.0
# VERSION_NUMBER 1.0
# VERSION_DATE 2007-07-24
# SCALE 1.0 1.0 1.0
1 1 14.566132 34.873772 7.857000 0.717830 -1
2 0 16.022520 33.760513 7.047000 0.463378 1
3 5 17.542000 32.604973 6.885001 0.638007 2
4 0 19.163984 32.022469 5.913000 0.602284 3
5 0 20.448090 30.822802 4.860000 0.436025 4
6 6 21.897903 28.881084 3.402000 0.471886 5
7 0 18.461960 30.289471 8.586000 0.447463 3
8 6 19.420759 28.730757 9.558000 0.496217 7"""

  skel = PrecomputedSkeleton.from_swc(test_file)
  assert skel.vertices.shape[0] == 8
  assert skel.edges.shape[0] == 7

  skel_gt = PrecomputedSkeleton(
    vertices=[
      [14.566132, 34.873772, 7.857000],
      [16.022520, 33.760513, 7.047000],
      [17.542000, 32.604973, 6.885001],
      [19.163984, 32.022469, 5.913000],
      [20.448090, 30.822802, 4.860000],
      [21.897903, 28.881084, 3.402000],
      [18.461960, 30.289471, 8.586000],
      [19.420759, 28.730757, 9.558000]
    ],
    edges=[ (0,1), (1,2), (2,3), (3,4), (4,5), (2,6), (7,6) ],
    radii=[ 
      0.717830, 0.463378, 0.638007, 0.602284, 
      0.436025, 0.471886, 0.447463, 0.496217
    ],
    vertex_types=[
      1, 0, 5, 0, 0, 6, 0, 6
    ],
  )

  assert PrecomputedSkeleton.equivalent(skel, skel_gt)

def test_components():
  skel = PrecomputedSkeleton(
    [ 
      (0,0,0), (1,0,0), (2,0,0),
      (0,1,0), (0,2,0), (0,3,0),
    ], 
    edges=[ 
      (0,1), (1,2), 
      (3,4), (4,5), (3,5)
    ],
    segid=666,
  )

  components = skel.components()
  assert len(components) == 2
  assert components[0].vertices.shape[0] == 3
  assert components[1].vertices.shape[0] == 3
  assert components[0].edges.shape[0] == 2
  assert components[1].edges.shape[0] == 3

  skel1_gt = PrecomputedSkeleton([(0,0,0), (1,0,0), (2,0,0)], [(0,1), (1,2)])
  skel2_gt = PrecomputedSkeleton([(0,1,0), (0,2,0), (0,3,0)], [(0,1), (0,2), (1,2)])

  assert PrecomputedSkeleton.equivalent(components[0], skel1_gt)
  assert PrecomputedSkeleton.equivalent(components[1], skel2_gt)

def test_caching():
  vol = CloudVolume('file:///tmp/cloudvolume/test-skeletons', 
    info=info, cache=True)

  vol.cache.flush()

  skel = PrecomputedSkeleton(
    [ 
      (0,0,0), (1,0,0), (2,0,0),
      (0,1,0), (0,2,0), (0,3,0),
    ], 
    edges=[ 
      (0,1), (1,2), 
      (3,4), (4,5), (3,5)
    ],
    segid=666,
  )

  vol.skeleton.upload(skel)

  assert vol.cache.list_skeletons() == [ '666.gz' ]

  skel.id = 1
  with open(os.path.join(vol.cache.path, 'skeletons/1'), 'wb') as f:
    f.write(skel.encode())

  cached_skel = vol.skeleton.get(1)

  assert cached_skel == skel

  vol.cache.flush()











