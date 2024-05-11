import pytest

import copy
import gzip
import json
import math
import numpy as np
import os
import shutil

from cloudvolume import CloudVolume, chunks, Storage, Skeleton
from cloudvolume.storage import SimpleStorage
from cloudvolume.lib import mkdir, Bbox, Vec, jsonify

from cloudvolume.datasource.precomputed.sharding import ShardingSpecification
from cloudvolume.exceptions import SkeletonDecodeError, SkeletonAttributeMixingError

from cloudfiles import CloudFile

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

skel_info = {
  "@type": "neuroglancer_skeletons", 
  "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], 
  "vertex_attributes": [
    {"id": "radius", "data_type": "float32", "num_components": 1}
  ], 
  "mip": 3,
}

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
  assert vol.skeleton.meta.skeleton_path == 'skeletons'
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
  skel = Skeleton(
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

  correct_skel = Skeleton(
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

def test_remove_disconnected_vertices():
  skel = Skeleton(
    [ 
      (0,0,0), (1,0,0), (2,0,0),
      (0,1,0), (0,2,0), (0,3,0),
      (-1, -1, -1)
    ], 
    edges=[ 
      (0,1), (1,2), 
      (3,4), (4,5), (3,5)
    ],
    segid=666,
  )

  res = skel.remove_disconnected_vertices()
  assert res.vertices.shape[0] == 6
  assert res.edges.shape[0] == 5 
  assert res.radii.shape[0] == 6
  assert res.vertex_types.shape[0] == 6
  assert res.id == 666


def test_equivalent():
  assert Skeleton.equivalent(Skeleton(), Skeleton())

  identity = Skeleton([ (0,0,0), (1,0,0) ], [(0,1)] )
  assert Skeleton.equivalent(identity, identity)

  diffvertex = Skeleton([ (0,0,0), (0,1,0) ], [(0,1)])
  assert not Skeleton.equivalent(identity, diffvertex)

  single1 = Skeleton([ (0,0,0), (1,0,0) ], edges=[ (1,0) ])
  single2 = Skeleton([ (0,0,0), (1,0,0) ], edges=[ (0,1) ])
  assert Skeleton.equivalent(single1, single2)

  double1 = Skeleton([ (0,0,0), (1,0,0) ], edges=[ (1,0) ])
  double2 = Skeleton([ (0,0,0), (1,0,0) ], edges=[ (0,1) ])
  assert Skeleton.equivalent(double1, double2)

  double1 = Skeleton([ (0,0,0), (1,0,0), (1,1,0) ], edges=[ (1,0), (1,2) ])
  double2 = Skeleton([ (0,0,0), (1,0,0), (1,1,0) ], edges=[ (2,1), (0,1) ])
  assert Skeleton.equivalent(double1, double2)

  double1 = Skeleton([ (0,0,0), (1,0,0), (1,1,0), (1,1,3) ], edges=[ (1,0), (1,2), (1,3) ])
  double2 = Skeleton([ (0,0,0), (1,0,0), (1,1,0), (1,1,3) ], edges=[ (3,1), (2,1), (0,1) ])
  assert Skeleton.equivalent(double1, double2)

def test_cable_length():
  skel = Skeleton([ 
      (0,0,0), (1,0,0), (2,0,0), (3,0,0), (4,0,0), (5,0,0)
    ], 
    edges=[ (1,0), (1,2), (2,3), (3,4), (5,4) ],
    radii=[ 1, 2, 3, 4, 5, 6 ],
    vertex_types=[1, 2, 3, 4, 5, 6]
  )

  assert skel.cable_length() == (skel.vertices.shape[0] - 1)

  skel = Skeleton([ 
      (2,0,0), (1,0,0), (0,0,0), (0,5,0), (0,6,0), (0,7,0)
    ], 
    edges=[ (1,0), (1,2), (2,3), (3,4), (5,4) ],
    radii=[ 1, 2, 3, 4, 5, 6 ],
    vertex_types=[1, 2, 3, 4, 5, 6]
  )
  assert skel.cable_length() == 9

  skel = Skeleton([ 
      (1,1,1), (0,0,0), (1,0,0)
    ], 
    edges=[ (1,0), (1,2) ],
    radii=[ 1, 2, 3],
    vertex_types=[1, 2, 3]
  )
  assert abs(skel.cable_length() - (math.sqrt(3) + 1)) < 1e-6

def test_transform():
  skelv = Skeleton([ 
      (0,0,0), (1,0,0), (1,1,0), (1,1,3), (2,1,3), (2,2,3)
    ], 
    edges=[ (1,0), (1,2), (2,3), (3,4), (5,4) ],
    radii=[ 1, 2, 3, 4, 5, 6 ],
    vertex_types=[1, 2, 3, 4, 5, 6],
    segid=1337,
    transform=np.array([
      [2, 0, 0, 0],
      [0, 2, 0, 0],
      [0, 0, 2, 0],
    ])
  )

  skelp = skelv.physical_space()
  assert np.all(skelp.vertices == skelv.vertices * 2)
  assert np.all(skelv.vertices == skelp.voxel_space().vertices)

  skelv.transform = [
    [1, 0, 0, 1],
    [0, 1, 0, 2],
    [0, 0, 1, 3],
  ]

  skelp = skelv.physical_space()
  tmpskel = skelv.clone() 
  tmpskel.vertices[:,0] += 1
  tmpskel.vertices[:,1] += 2
  tmpskel.vertices[:,2] += 3
  assert np.all(skelp.vertices == tmpskel.vertices)
  assert np.all(skelp.voxel_space().vertices == skelv.vertices)


def test_downsample():
  skel = Skeleton([ 
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
  assert Skeleton.equivalent(dskel, skel)
  assert dskel.id == skel.id
  assert dskel.id == 1337

  dskel = skel.downsample(2)
  dskel_gt = Skeleton(
    [ (0,0,0), (1,1,0), (2,1,3), (2,2,3) ], 
    edges=[ (1,0), (1,2), (2,3) ],
    radii=[1,3,5,6], vertex_types=[1,3,5,6] 
  )
  assert Skeleton.equivalent(dskel, dskel_gt)

  dskel = skel.downsample(3)
  dskel_gt = Skeleton(
    [ (0,0,0), (1,1,3), (2,2,3) ], edges=[ (1,0), (1,2) ],
    radii=[1,4,6], vertex_types=[1,4,6],
  )
  assert Skeleton.equivalent(dskel, dskel_gt)

  skel = Skeleton([ 
      (0,0,0), (1,0,0), (1,1,0), (1,1,3), (2,1,3), (2,2,3)
    ], 
    edges=[ (1,0), (1,2), (3,4), (5,4) ],
    radii=[ 1, 2, 3, 4, 5, 6 ],
    vertex_types=[1, 2, 3, 4, 5, 6]
  )
  dskel = skel.downsample(2)
  dskel_gt = Skeleton(
    [ (0,0,0), (1,1,0), (1,1,3), (2,2,3) ], 
    edges=[ (1,0), (2,3) ],
    radii=[1,3,4,6], vertex_types=[1,3,4,6] 
  )
  assert Skeleton.equivalent(dskel, dskel_gt)


def test_downsample_joints():
  skel = Skeleton([ 
      
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
  ds_skel_gt = Skeleton([ 

                        (2, 3,0), # 0
                        
                        (2, 1,0), # 1
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
    radii=[ 0, 2, 3, 5, 7, 9, 10 ],
    vertex_types=[ 0, 2, 3, 5, 7, 9, 10 ],
    segid=1337,
  )

  assert Skeleton.equivalent(ds_skel, ds_skel_gt)


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

  skel = Skeleton.from_swc(test_file)
  assert skel.vertices.shape[0] == 8
  assert skel.edges.shape[0] == 7

  skel_gt = Skeleton(
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

  assert Skeleton.equivalent(skel, skel_gt)

  skel = Skeleton.from_swc(skel.to_swc())
  assert Skeleton.equivalent(skel, skel_gt)

  # sorts edges
  skel = skel.consolidate()
  skel_gt = skel_gt.consolidate()
  assert np.all(skel.edges == skel_gt.edges)
  assert np.all(np.abs(skel.radii - skel_gt.radii) < 0.00001)

  Nv = skel.vertices.shape[0]
  Ne = skel.edges.shape[0]

  for _ in range(10):
    skel = Skeleton.from_swc(skel.to_swc())
    assert skel.vertices.shape[0] == Nv 
    assert skel.edges.shape[0] == Ne

def test_read_duplicate_vertex_swc():
  test_file = """
1 0 -18.458370 23.227150 -84.035016 1.000000 -1
2 0 -18.159709 22.925778 -82.984344 1.000000 1
3 0 -17.861047 22.624407 -82.984344 1.000000 2
4 0 -17.562385 22.624407 -82.984344 1.000000 3
5 0 -16.965061 22.021663 -82.984344 1.000000 4
6 0 -16.965061 21.720292 -82.984344 1.000000 5
7 0 -16.069075 21.720292 -82.984344 1.000000 6
8 0 -16.069075 21.117548 -80.883000 1.000000 7
9 0 -15.770414 20.816176 -80.883000 1.000000 8
10 0 -15.770414 20.514805 -80.883000 1.000000 9
11 0 -15.770414 20.816176 -80.883000 1.000000 10
12 0 -16.069075 21.117548 -80.883000 1.000000 11
13 0 -16.069075 21.418920 -80.883000 1.000000 12
14 0 -16.069075 20.816176 -78.781655 1.000000 13
15 0 -15.471752 20.213433 -76.680311 1.000000 14
16 0 -15.471752 19.309318 -76.680311 1.000000 15
17 0 -15.471752 19.007946 -75.629639 1.000000 16
18 0 -15.173090 18.706574 -74.578966 1.000000 17
19 0 -14.874428 18.706574 -74.578966 1.000000 18
20 0 -14.575766 18.405202 -74.578966 1.000000 19
"""

  skel = Skeleton.from_swc(test_file)
  assert skel.vertices.shape[0] == 20
  skel2 = Skeleton.from_swc(skel.to_swc())
  assert skel2.vertices.shape[0] == 20
  assert Skeleton.equivalent(skel, skel2)

def test_components():
  skel = Skeleton(
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

  skel1_gt = Skeleton([(0,0,0), (1,0,0), (2,0,0)], [(0,1), (1,2)])
  skel2_gt = Skeleton([(0,1,0), (0,2,0), (0,3,0)], [(0,1), (0,2), (1,2)])

  assert Skeleton.equivalent(components[0], skel1_gt)
  assert Skeleton.equivalent(components[1], skel2_gt)

def test_caching():
  vol = CloudVolume('file:///tmp/cloudvolume/test-skeletons', 
    info=info, cache=True)

  vol.cache.flush()

  skel = Skeleton(
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
    f.write(skel.to_precomputed())

  cached_skel = vol.skeleton.get(1)

  assert cached_skel == skel

  vol.cache.flush()


def test_simple_merge():
  skel1 = Skeleton(
    [ (0,0,0), (1,0,0), (2,0,0),  ], 
    edges=[ (0,1), (1,2), ],
    segid=1,
  )

  skel2 = Skeleton(
    [ (0,0,1), (1,0,2), (2,0,3),  ], 
    edges=[ (0,1), (1,2), ],
    segid=1,
  )

  result = Skeleton.simple_merge([ skel1, skel2 ])

  expected = Skeleton(
    [ (0,0,0), (1,0,0), (2,0,0), (0,0,1), (1,0,2), (2,0,3), ], 
    edges=[ (0,1), (1,2), (3,4), (4,5) ],
    segid=1,
  )

  assert result == expected

  wow_attr = {
    "id": "wow",
    "data_type": "uint8",
    "components": 1,
  }

  skel1.extra_attributes = [copy.deepcopy(wow_attr)]
  skel1.wow = np.array([1,2,3], dtype=np.uint8)

  skel2.extra_attributes = [copy.deepcopy(wow_attr)]
  skel2.wow = np.array([4,5,6], dtype=np.uint8)

  result = Skeleton.simple_merge([ skel1, skel2 ])
  expected.extra_attributes = [copy.deepcopy(wow_attr)]
  expected.wow = np.array([1,2,3,4,5,6], dtype=np.uint8)

  assert result == expected

  skel2.extra_attributes[0]['data_type'] = np.uint8

  try:
    Skeleton.simple_merge([ skel1, skel2 ])
    assert False
  except SkeletonAttributeMixingError:
    pass

  skel2.extra_attributes[0]['data_type'] = 'uint8'
  skel2.extra_attributes.append({
    "id": "amaze",
    "data_type": "float32",
    "components": 2,
  })
  skel2.amaze = np.array([])

  try:
    Skeleton.simple_merge([ skel1, skel2 ])
    assert False
  except SkeletonAttributeMixingError:
    pass

def test_sharded():
  skel = Skeleton(
    [ 
      (0,0,0), (1,0,0), (2,0,0),
      (0,1,0), (0,2,0), (0,3,0),
    ], 
    edges=[ 
      (0,1), (1,2), 
      (3,4), (4,5), (3,5)
    ],
    segid=1,
    extra_attributes=[
      {
          "id": "radius",
          "data_type": "float32",
          "num_components": 1,
      }
    ]
  ).physical_space()

  skels = {}
  for i in range(10):
    sk = skel.clone()
    sk.id = i
    skels[i] = sk.to_precomputed()

  mkdir('/tmp/removeme/skeletons/sharded/skeletons')
  with open('/tmp/removeme/skeletons/sharded/info', 'wt') as f:
    f.write(jsonify(info))

  for idxenc in ('raw', 'gzip'):
    for dataenc in ('raw', 'gzip'):

      spec = ShardingSpecification(
        'neuroglancer_uint64_sharded_v1', 
        preshift_bits=1,
        hash='murmurhash3_x86_128', 
        minishard_bits=2, 
        shard_bits=1, 
        minishard_index_encoding=idxenc, 
        data_encoding=dataenc,
      )
      skel_info['sharding'] = spec.to_dict()

      with open('/tmp/removeme/skeletons/sharded/skeletons/info', 'wt') as f:
        f.write(jsonify(skel_info))

      files = spec.synthesize_shards(skels)
      for fname in files.keys():
        with open('/tmp/removeme/skeletons/sharded/skeletons/' + fname, 'wb') as f:
          f.write(files[fname])      

      cv = CloudVolume('file:///tmp/removeme/skeletons/sharded/')
      assert cv.skeleton.meta.mip == 3

      for i in range(10):
        sk = cv.skeleton.get(i).physical_space()
        sk.id = 1
        assert sk == skel

      labels = []
      for fname in files.keys():
        lbls = cv.skeleton.reader.list_labels(fname, path='skeletons')
        labels += list(lbls)
      
      labels.sort()
      assert labels == list(range(10))

      for filename, shard in files.items():
        decoded_skels = cv.skeleton.reader.disassemble_shard(shard)
        for label, binary in decoded_skels.items():
          Skeleton.from_precomputed(binary)

      exists = cv.skeleton.reader.exists(list(range(11)), path='skeletons')
      assert exists == {
        0: 'skeletons/0.shard', 
        1: 'skeletons/0.shard', 
        2: 'skeletons/0.shard', 
        3: 'skeletons/0.shard', 
        4: 'skeletons/0.shard', 
        5: 'skeletons/0.shard', 
        6: 'skeletons/0.shard', 
        7: 'skeletons/0.shard', 
        8: 'skeletons/1.shard', 
        9: 'skeletons/1.shard',
        10: None,
      }

  shutil.rmtree('/tmp/removeme/skeletons')

def test_integer_spatial_index():
  test_dir = os.path.dirname(os.path.abspath(__file__))
  vol = CloudVolume(f'file://{test_dir}/test_cv')
  
  spatial_index = vol.skeleton.spatial_index

  idx_filename = "0-8192_0-8192_0-20480.spatial"

  labels = spatial_index.query(vol.bounds * vol.resolution)
  labels = list(labels)
  labels.sort()

  cf = CloudFile(f"file://" + os.path.join(test_dir, "test_cv", "skeletons_mip_2", idx_filename))
  gt_labels = [ int(x) for x in cf.get_json().keys() ]
  gt_labels.sort()

  assert labels == gt_labels

  locs = spatial_index.file_locations_per_label()
  assert 71297420 in locs
  assert locs[71297420] == ['0-8192_0-8192_0-20480.spatial']

  locs = spatial_index.file_locations_per_label(labels=[71297420])
  assert 71297420 in locs
  assert locs[71297420] == ['0-8192_0-8192_0-20480.spatial']

  bbox = spatial_index.get_bbox(59524925)
  assert bbox == Bbox([80, 6864, 19960], [400, 8176, 20440])




