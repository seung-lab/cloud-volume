import pytest

import numpy as np

import cloudvolume.lib as lib
from cloudvolume.lib import Bbox, Vec

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
    'z': np.int32(5),
    'w': '1 2 34 5'
  }

  assert lib.jsonify(obj, sort_keys=True) == r"""{"w": "1 2 34 5", "x": [[1, 2, 3, 4, 5]], "y": [{}, {}], "z": 5}"""

