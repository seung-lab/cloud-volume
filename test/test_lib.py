import pytest

import os
import re

import numpy as np

import cloudvolume.lib as lib
from cloudvolume.lib import Bbox, Vec

from cloudvolume.exceptions import UnsupportedProtocolError

def test_divisors():

  divisors = (
    (1, (1,)), 
    (2, (1,2)),
    (3, (1,3)),
    (4, (1,2,4)),
    (6, (1,2,3,6)),
    (35, (1,5,7,35)),
    (128, (1,2,4,8,16,32,64,128)),
    (258, (1,2,3,6,43,86,129,258)),
  )

  for num, ans in divisors:
    result = [ _ for _ in lib.divisors(num) ]
    result.sort()
    assert tuple(result) == ans

def test_find_closest_divisor():
  size = lib.find_closest_divisor( (128,128,128), (64,64,64) )
  assert tuple(size) == (64,64,64)

  size = lib.find_closest_divisor( (240,240,240), (64,64,64) )
  assert tuple(size) == (60,60,60)

  size = lib.find_closest_divisor( (224,224,224), (64,64,64) )
  assert tuple(size) == (56,56,56)

  size = lib.find_closest_divisor( (73,73,73), (64,64,64) )
  assert tuple(size) == (73,73,73)

def test_bucket_path_extraction():
  extract = lib.extract_bucket_path(r'file://C:\wow\this\is\a\cool\path', windows=True, disable_toabs=True)
  print(extract)
  assert extract.protocol == 'file'
  assert extract.bucket == 'C:\\wow\\'

  # on linux the toabs prepends the current path because it
  # doesn't understand C:\... so we can't test that here.
  # assert extract.path == 'this\\is\\a\\cool\\path' 

  try:
    extract = lib.extract_bucket_path(r'file://C:\wow\this\is\a\cool\path', windows=False, disable_toabs=True)
    assert False 
  except UnsupportedProtocolError:
    pass

def test_path_extraction():
  def shoulderror(url):
    try:
        lib.extract_path(url)
        assert False, url
    except:
        pass

  def okgoogle(url):
    path = lib.extract_path(url)
    assert path.protocol == 'gs', url
    assert path.bucket == 'bucket', url
    assert path.intermediate_path == '', url
    assert path.dataset == 'dataset', url
    assert path.layer == 'layer', url

  okgoogle('gs://bucket/dataset/layer') 
  shoulderror('s4://dataset/layer')
  shoulderror('dataset/layer')
  shoulderror('s3://dataset')

  firstdir = lambda x: '/' + x.split('/')[1]

  homepath = lib.toabs('~')
  homerintermediate = homepath.replace(firstdir(homepath), '')[1:]

  curpath = lib.toabs('.')
  curintermediate = curpath.replace(firstdir(curpath), '')[1:]

  assert (lib.extract_path('s3://seunglab-test/intermediate/path/dataset/layer') 
      == lib.ExtractedPath('s3', 'intermediate/path/', 'seunglab-test', 'dataset', 'layer'))

  assert (lib.extract_path('file:///tmp/dataset/layer') 
      == lib.ExtractedPath('file', '', "/tmp", 'dataset', 'layer'))

  assert (lib.extract_path('file://seunglab-test/intermediate/path/dataset/layer') 
      == lib.ExtractedPath('file', os.path.join(curintermediate, 'seunglab-test', 'intermediate/path/'), firstdir(curpath), 'dataset', 'layer'))

  assert (lib.extract_path('gs://seunglab-test/intermediate/path/dataset/layer') 
      == lib.ExtractedPath('gs', 'intermediate/path/', 'seunglab-test', 'dataset', 'layer'))

  assert (lib.extract_path('file://~/seunglab-test/intermediate/path/dataset/layer') 
      == lib.ExtractedPath('file', os.path.join(homerintermediate, 'seunglab-test', 'intermediate/path/'), firstdir(homepath), 'dataset', 'layer'))

  assert (lib.extract_path('file:///User/me/.cloudvolume/cache/gs/bucket/dataset/layer') 
      == lib.ExtractedPath('file', 'me/.cloudvolume/cache/gs/bucket/', '/User', 'dataset', 'layer'))

  shoulderror('s3://dataset/layer/')

  shoulderror('ou3bouqjsa fkj aojsf oaojf ojsaf')

  okgoogle('gs://bucket/dataset/layer/')
  shoulderror('gs://bucket/dataset/layer/info')

  path = lib.extract_path('s3://bucketxxxxxx/datasetzzzzz91h8__3/layer1br9bobasjf/')
  assert path.protocol == 's3'
  assert path.bucket == 'bucketxxxxxx'
  assert path.dataset == 'datasetzzzzz91h8__3'
  assert path.layer == 'layer1br9bobasjf'

  path = lib.extract_path('file:///bucket/dataset/layer/')
  assert path.protocol == 'file'
  assert path.bucket == '/bucket'
  assert path.dataset == 'dataset'
  assert path.layer == 'layer'

  shoulderror('lucifer://bucket/dataset/layer/')
  shoulderror('gs://///')
  shoulderror('gs://seunglab-test//segmentation')

  path = lib.extract_path('file:///tmp/removeme/layer/')
  assert path.protocol == 'file'
  assert path.bucket == '/tmp'
  assert path.dataset == 'removeme'
  assert path.layer == 'layer'

def test_bbox_subvoxel():
  bbox = Bbox( (0,0,0), (1,1,1), dtype=np.float32)
  
  assert not bbox.subvoxel()
  assert not bbox.empty()

  bbox.maxpt[:] *= -1
  bbox.maxpt.z = 20

  # pathological case
  assert not (bbox.volume() < 1)
  assert bbox.subvoxel()
  assert bbox.empty()

  bbox = Bbox( (1,1,1), (1,1,1) )
  assert bbox.empty()

  bbox = Bbox( (0,0,0), (0.9, 1.0, 1.0) )
  assert bbox.subvoxel()
  assert not bbox.empty()

def test_vec_division():
  vec = Vec(2,4,8)
  assert np.all( (vec/2) == Vec(1,2,4) )


def test_bbox_division():
  box = Bbox( (0,2,4), (4,8,16) )
  assert (box//2) == Bbox( (0,1,2), (2,4,8) )

  box = Bbox( (0,3,4), (4,8,16), dtype=np.float32 )
  print((box/2.))
  print(Bbox( (0., 1.5, 2.), (2., 4., 8.) ))
  assert (box/2.) == Bbox( (0., 1.5, 2.), (2., 4., 8.) )

def test_bbox_intersect():
  box = Bbox( (0,0,0), (10, 10, 10) )
  
  assert Bbox.intersects(box, box)
  assert Bbox.intersects(box, Bbox((1,1,1), (11,11,11)) )
  assert Bbox.intersects(box, Bbox((-1,-1,-1), (9,9,9)) ) 
  assert Bbox.intersects(box, Bbox((5, -5, 0), (15, 5, 10)))
  assert not Bbox.intersects(box, Bbox( (30,30,30), (40,40,40) ))
  assert not Bbox.intersects(box, Bbox( (-30,-30,-30), (-40,-40,-40) ))
  assert not Bbox.intersects(box, Bbox( (10, 0, 0), (20, 10, 10) ))

def test_bbox_intersection():
  bbx1 = Bbox( (0,0,0), (10,10,10) )
  bbx2 = Bbox( (5,5,5), (15,15,15) )

  assert Bbox.intersection(bbx1, bbx2) == Bbox((5,5,5),(10,10,10))
  assert Bbox.intersection(bbx2, bbx1) == Bbox((5,5,5),(10,10,10))
  bbx2.minpt = Vec(11,11,11)
  assert Bbox.intersection(bbx1, bbx2) == Bbox((0,0,0),(0,0,0))

def test_bbox_hashing():
  bbx = Bbox.from_list([ 1,2,3,4,5,6 ])
  d = {}
  d[bbx] = 1

  assert len(d) == 1
  for _,v in d.items():
    assert v == 1

  bbx = Bbox( (1., 1.3, 2.), (3., 4., 4.) )
  d = {}
  d[bbx] = 1

  assert len(d) == 1
  for _,v in d.items():
    assert v == 1

def test_bbox_serialize():
  bbx = Bbox( (24.125, 2512.2, 2112.3), (33.,532., 124.12412), dtype=np.float32)

  reconstituted = Bbox.deserialize(bbx.serialize())
  assert bbx == reconstituted

def test_bbox_volume():
  bbx = Bbox( (0,0,0), (2000, 2000, 2000) )
  # important thing is 8B is > int32 size
  assert bbx.volume() == 8000000000

  bbx = bbx.astype(np.float32)
  assert bbx.volume() == 8000000000

def test_jsonify():
  obj = {
    'x': [ np.array([1,2,3,4,5], dtype=np.uint64) ],
    'y': [ {}, {} ],
    'z': 5,
    'w': '1 2 34 5'
  }

  assert lib.jsonify(obj, sort_keys=True) == r"""{"w": "1 2 34 5", "x": [[1, 2, 3, 4, 5]], "y": [{}, {}], "z": 5}"""

