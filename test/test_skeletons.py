import pytest

import copy
import json
import os
import numpy as np
import shutil
import gzip
import json

from cloudvolume import CloudVolume, chunks, Storage, PrecomputedSkeleton
from cloudvolume.storage import SimpleStorage
from cloudvolume.lib import mkdir, Bbox, Vec

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

  vol = CloudVolume('file:///tmp/cloudvolume/test-skeletons', info=info)
  vol.skeleton.upload(segid=1, vertices=vertices, edges=edges)
  skel = vol.skeleton.get(1)

  assert skel.id == 1
  assert np.all(skel.vertices == vertices)
  assert np.all(skel.edges == edges)
  assert vol.skeleton.path == 'skeletons'

  with SimpleStorage('file:///tmp/cloudvolume/test-skeletons/') as stor:
    rawskel = stor.get_file('skeletons/1')
    assert len(rawskel) == 228 # 8 + 11 * 12 + 11 * 8
    stor.delete_file('skeletons/1')
  

def test_no_edges():
  vertices = np.array([
    [ 0, 1, 0 ],
    [ 1, 0, 0 ],
  ], np.float32)

  edges = None
  vol = CloudVolume('file:///tmp/cloudvolume/test-skeletons', info=info)
  vol.skeleton.upload(2, vertices, edges)
  skel = vol.skeleton.get(2)

  assert skel.id == 2
  assert np.all(skel.vertices == vertices)
  assert np.all(skel.edges.shape == (0, 2))
  assert vol.skeleton.path == 'skeletons'

  with SimpleStorage('file:///tmp/cloudvolume/test-skeletons/') as stor:
    rawskel = stor.get_file('skeletons/2')
    assert len(rawskel) == 8 + 2 * 12 + 0 * 8
    stor.delete_file('skeletons/2')

def test_no_vertices():
  vertices = np.array([], np.float32).reshape(0,3)

  edges = None
  vol = CloudVolume('file:///tmp/cloudvolume/test-skeletons', info=info)
  vol.skeleton.upload(3, vertices, edges)
  skel = vol.skeleton.get(3)

  assert skel.id == 3
  assert np.all(skel.vertices == vertices)
  assert np.all(skel.edges.shape == (0, 2))
  assert vol.skeleton.path == 'skeletons'

  with SimpleStorage('file:///tmp/cloudvolume/test-skeletons/') as stor:
    rawskel = stor.get_file('skeletons/3')
    assert len(rawskel) == 8 + 0 * 12 + 0 * 8
    stor.delete_file('skeletons/3')