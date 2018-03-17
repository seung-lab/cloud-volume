from __future__ import print_function
from six.moves import range, reduce

from collections import namedtuple
import json
import os
import io
import re 
import sys
import math
import shutil
import operator
import time
from itertools import product

import numpy as np
from tqdm import tqdm

COLORS = {
  'RESET': "\033[m",
  'YELLOW': "\033[1;93m",
  'RED': '\033[1;91m',
  'GREEN': '\033[1;92m',
}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def jsonify(obj, **kwargs):
  return json.dumps(obj, cls=NumpyEncoder, **kwargs)

def yellow(text):
  return colorize('yellow', text)

def red(text):
  return colorize('red', text)

def colorize(color, text):
  color = color.upper()
  return COLORS[color] + text + COLORS['RESET']

ExtractedPath = namedtuple('ExtractedPath', 
  ('protocol', 'intermediate_path', 'bucket', 'dataset','layer')
)

def extract_path(cloudpath):
  """cloudpath: e.g. gs://neuroglancer/DATASET/LAYER/info or s3://..."""
  protocol_re = r'^(gs|file|s3|boss)://'
  bucket_re = r'^(/?[~\d\w_\.\-]+)/'
  tail_re = r'([\d\w_\.\-]+)/([\d\w_\.\-]+)/?$'

  error = ValueError("""
    Cloud path must conform to PROTOCOL://BUCKET/zero/or/more/dirs/DATASET/LAYER
    Example: gs://test_bucket/mouse_dataset/em

    Supported protocols: gs, s3, file, boss

    Received: {}
    """.format(cloudpath))

  match = re.match(protocol_re, cloudpath)

  if not match:
    raise error

  (protocol,) = match.groups()
  cloudpath = re.sub(protocol_re, '', cloudpath)
  if protocol == 'file':
    cloudpath = toabs(cloudpath)

  match = re.match(bucket_re, cloudpath)
  if not match:
    raise error

  (bucket,) = match.groups()
  cloudpath = re.sub(bucket_re, '', cloudpath)

  match = re.search(tail_re, cloudpath)
  if not match:
    raise error
  dataset, layer = match.groups()

  intermediate_path = re.sub(tail_re, '', cloudpath)
  return ExtractedPath(protocol, intermediate_path, bucket, dataset, layer)

def toabs(path):
  home = os.path.join(os.environ['HOME'], '')
  path = re.sub('^~/?', home, path)
  return os.path.abspath(path)

def mkdir(path):
  path = toabs(path)

  try:
    if path != '' and not os.path.exists(path):
      os.makedirs(path)
  except OSError as e:
    if e.errno == 17: # File Exists
      time.sleep(0.1)
      return mkdir(path)
    else:
      raise

  return path

def touch(path):
  mkdir(os.path.dirname(path))
  open(path, 'a').close()

def list_shape(shape, elem=None):
    """create Nd list filled wtih elem. e.g. shape([2,2], 0) => [ [0,0], [0,0] ]"""

    if (len(shape) == 0):
        return []

    def helper(elem, shape, i):
        if len(shape) - 1 == i:
            return [elem] * shape[i]
        return [ helper(elem, shape, i+1) for _ in range(shape[i]) ]

    return helper(elem, shape, 0)

def find_closest_divisor(to_divide, closest_to):
  """
  This is used to find the right chunk size for
  importing a neuroglancer dataset that has a
  chunk import size that's not evenly divisible by
  64,64,64. 

  e.g. 
    neuroglancer_chunk_size = find_closest_divisor(build_chunk_size, closest_to=[64,64,64])

  Required:
    to_divide: (tuple) x,y,z chunk size to rechunk
    closest_to: (tuple) x,y,z ideal chunk size

  Return: [x,y,z] chunk size that works for ingestion
  """
  def find_closest(td, ct):
    min_distance = td
    best = td
    
    for divisor in divisors(td):
      if abs(divisor - ct) < min_distance:
        min_distance = abs(divisor - ct)
        best = divisor
    return best
  
  return [ find_closest(td, ct) for td, ct in zip(to_divide, closest_to) ]

def divisors(n):
  """Generate the divisors of n"""
  for i in range(1, int(math.sqrt(n) + 1)):
    if n % i == 0:
      yield i
      if i*i != n:
        yield n / i

def xyzrange(start_vec, end_vec=None, stride_vec=(1,1,1)):
  if end_vec is None:
    end_vec = start_vec
    start_vec = (0,0,0)

  start_vec = np.array(start_vec, dtype=int)
  end_vec = np.array(end_vec, dtype=int)

  rangeargs = ( (start, end, stride) for start, end, stride in zip(start_vec, end_vec, stride_vec) )
  xyzranges = [ range(*arg) for arg in rangeargs ]
  
  # iterate then x first, then y, then z
  # this way you process in the xy plane slice by slice
  # but you don't create process lots of prefix-adjacent keys
  # since all the keys start with X
  zyxranges = xyzranges[::-1]

  def vectorize():
    pt = Vec(0,0,0)
    for z,y,x in product(*zyxranges):
      pt.x, pt.y, pt.z = x, y, z
      yield pt

  return vectorize()

def map2(fn, a, b):
  assert len(a) == len(b), "Vector lengths do not match: {} (len {}), {} (len {})".format(a[:3], len(a), b[:3], len(b))

  result = np.empty(len(a))

  for i in range(len(result)):
    result[i] = fn(a[i], b[i])

  if isinstance(a, Vec) or isinstance(b, Vec):
    return Vec(*result)

  return result

def max2(a, b):
  return map2(max, a, b).astype(a.dtype)

def min2(a, b):
  return map2(min, a, b).astype(a.dtype)

def clamp(val, low, high):
  return min(max(val, low), high)

def check_bounds(val, low, high):
  if val > high or val < low:
    raise ValueError('Value {} cannot be outside of inclusive range {} to {}'.format(val,low,high))
  return val

class Vec(np.ndarray):
    def __new__(cls, *args, **kwargs):
      dtype = kwargs['dtype'] if 'dtype' in kwargs else int
      return super(Vec, cls).__new__(cls, shape=(len(args),), buffer=np.array(args).astype(dtype), dtype=dtype)

    @classmethod
    def clamp(cls, val, minvec, maxvec):
      return Vec(*min2(max2(val, minvec), maxvec))

    def clone(self):
      return Vec(*self[:], dtype=self.dtype)

    def null(self):
        return self.length() <= 10 * np.finfo(np.float32).eps

    def dot(self, vec):
      return sum(self * vec)

    def length2(self):
        return self.dot(self)

    def length(self):
        return math.sqrt(self.dot(self))

    def rectVolume(self):
        return reduce(operator.mul, self)

    def __hash__(self):
      return int(''.join(map(str, self)))

    def __repr__(self):
      values = u",".join(self.astype(str))
      return u"Vec({}, dtype={})".format(values, self.dtype)

def __assign(self, val, index):
  self[index] = val

Vec.x = property(lambda self: self[0], lambda self,val: __assign(self,val,0))
Vec.y = property(lambda self: self[1], lambda self,val: __assign(self,val,1))
Vec.z = property(lambda self: self[2], lambda self,val: __assign(self,val,2))
Vec.w = property(lambda self: self[3], lambda self,val: __assign(self,val,3))

Vec.r = Vec.x
Vec.g = Vec.y
Vec.b = Vec.z
Vec.a = Vec.w


class Bbox(object):
  """Represents a three dimensional cuboid in space."""
  def __init__(self, a, b):
    self.minpt = Vec(
      min(a[0], b[0]),
      min(a[1], b[1]),
      min(a[2], b[2])
    )

    self.maxpt = Vec(
      max(a[0], b[0]),
      max(a[1], b[1]),
      max(a[2], b[2])
    )

  @classmethod
  def intersects(cls, bbx1, bbx2):
    return (
          bbx1.minpt.x < bbx2.maxpt.x 
      and bbx1.maxpt.x > bbx2.minpt.x 
      and bbx1.minpt.y < bbx2.maxpt.y
      and bbx1.maxpt.y > bbx2.minpt.y
      and bbx1.minpt.z < bbx2.maxpt.z
      and bbx1.maxpt.z > bbx2.minpt.z 
    )

  @classmethod
  def from_vec(cls, vec):
    return Bbox( (0,0,0), vec )

  @classmethod
  def from_filename(cls, filename):
    match = re.search(r'(-?\d+)-(-?\d+)_(-?\d+)-(-?\d+)_(-?\d+)-(-?\d+)(?:\.gz)?$', os.path.basename(filename))

    (xmin, xmax,
     ymin, ymax,
     zmin, zmax) = map(int, match.groups())

    return Bbox( (xmin, ymin, zmin), (xmax, ymax, zmax) )

  @classmethod
  def from_slices(cls, slices3):
    return Bbox(
      (slices3[0].start, slices3[1].start, slices3[2].start), 
      (slices3[0].stop, slices3[1].stop, slices3[2].stop) 
    )

  @classmethod
  def from_list(cls, lst):
    return Bbox( lst[:3], lst[3:6] )

  def to_filename(self):
    return '{}-{}_{}-{}_{}-{}'.format(
      self.minpt.x, self.maxpt.x,
      self.minpt.y, self.maxpt.y,
      self.minpt.z, self.maxpt.z,
    )

  def to_slices(self):
    return (
      slice(int(self.minpt.x), int(self.maxpt.x)),
      slice(int(self.minpt.y), int(self.maxpt.y)),
      slice(int(self.minpt.z), int(self.maxpt.z))
    )

  def to_list(self):
    return list(self.minpt) + list(self.maxpt)

  @classmethod
  def expand(cls, *args):
    result = args[0].clone()
    for bbx in args:
      result.minpt = min2(result.minpt, bbx.minpt)
      result.maxpt = max2(result.maxpt, bbx.maxpt)
    return result

  @classmethod
  def clamp(cls, bbx0, bbx1):
    result = bbx0.clone()
    result.minpt = Vec.clamp(bbx0.minpt, bbx1.minpt, bbx1.maxpt)
    result.maxpt = Vec.clamp(bbx0.maxpt, bbx1.minpt, bbx1.maxpt)
    return result

  def size3(self):
    return Vec(*(self.maxpt - self.minpt))

  def volume(self):
    return self.size3().rectVolume()

  def center(self):
    return (self.minpt + self.maxpt) / 2.0

  def expand_to_chunk_size(self, chunk_size, offset=Vec(0,0,0, dtype=int)):
    """
    Align a potentially non-axis aligned bbox to the grid by growing it
    to the nearest grid lines.

    Required:
      chunk_size: arraylike (x,y,z), the size of chunks in the 
                    dataset e.g. (64,64,64)
    Optional:
      offset: arraylike (x,y,z), the starting coordinate of the dataset
    """
    chunk_size = np.array(chunk_size, dtype=np.float32)
    result = self.clone()
    result = result - offset
    result.minpt = np.floor(result.minpt / chunk_size) * chunk_size
    result.maxpt = np.ceil(result.maxpt / chunk_size) * chunk_size 
    return result + offset

  def shrink_to_chunk_size(self, chunk_size, offset=Vec(0,0,0, dtype=int)):
    """
    Align a potentially non-axis aligned bbox to the grid by shrinking it
    to the nearest grid lines.

    Required:
      chunk_size: arraylike (x,y,z), the size of chunks in the 
                    dataset e.g. (64,64,64)
    Optional:
      offset: arraylike (x,y,z), the starting coordinate of the dataset
    """
    chunk_size = np.array(chunk_size, dtype=np.float32)
    result = self.clone()
    result = result - offset
    result.minpt = np.ceil(result.minpt / chunk_size) * chunk_size
    result.maxpt = np.floor(result.maxpt / chunk_size) * chunk_size 
    return result + offset

  def round_to_chunk_size(self, chunk_size, offset=Vec(0,0,0, dtype=int)):
    """
    Align a potentially non-axis aligned bbox to the grid by rounding it
    to the nearest grid lines.

    Required:
      chunk_size: arraylike (x,y,z), the size of chunks in the 
                    dataset e.g. (64,64,64)
    Optional:
      offset: arraylike (x,y,z), the starting coordinate of the dataset
    """
    chunk_size = np.array(chunk_size, dtype=np.float32)
    result = self.clone()
    result = result - offset
    result.minpt = np.round(result.minpt / chunk_size) * chunk_size
    result.maxpt = np.round(result.maxpt / chunk_size) * chunk_size
    return result + offset

  def contains(self, point):
    return (
          point[0] >= self.minpt[0] 
      and point[1] >= self.minpt[1]
      and point[2] >= self.minpt[2] 
      and point[0] <= self.maxpt[0] 
      and point[1] <= self.maxpt[1]
      and point[2] <= self.maxpt[2]
    )

  def contains_bbox(self, bbox):
    return self.contains(bbox.minpt) and self.contains(bbox.maxpt)

  def clone(self):
    return Bbox(self.minpt, self.maxpt)

  def astype(self, dtype):
    result = self.clone()
    result.minpt = self.minpt.astype(dtype)
    result.maxpt = self.maxpt.astype(dtype)
    return result

  def transpose(self):
    return Bbox(self.minpt[::-1], self.maxpt[::-1])

  # note that operand can be a vector 
  # or a scalar thanks to numpy
  def __sub__(self, operand): 
    tmp = self.clone()
    
    if isinstance(operand, Bbox):
      tmp.minpt -= operand.minpt
      tmp.maxpt -= operand.maxpt
    else:
      tmp.minpt -= operand
      tmp.maxpt -= operand

    return tmp

  def __add__(self, operand):
    tmp = self.clone()
    
    if isinstance(operand, Bbox):
      tmp.minpt += operand.minpt
      tmp.maxpt += operand.maxpt
    else:
      tmp.minpt += operand
      tmp.maxpt += operand

    return tmp

  def __mul__(self, operand):
    tmp = self.clone()
    tmp.minpt *= operand
    tmp.maxpt *= operand
    return tmp

  def __div__(self, operand):
    return self.__floordiv__(operand)

  def __floordiv__(self, operand):
    tmp = self.clone()
    tmp.minpt //= operand
    tmp.maxpt //= operand
    return tmp

  def __truediv__(self, operand):
    tmp = self.clone()

    if type(operand) is int:
      operand = float(operand)

    tmp.minpt = Vec(*( tmp.minpt.astype(float) / operand ))
    tmp.maxpt = Vec(*( tmp.maxpt.astype(float) / operand ))
    return tmp

  def __ne__(self, other):
    return not (self == other)

  def __eq__(self, other):
    return np.array_equal(self.minpt, other.minpt) and np.array_equal(self.maxpt, other.maxpt)

  def __hash__(self):
    return int(''.join(self.to_list()))

  def __repr__(self):
    return "Bbox({},{})".format(self.minpt, self.maxpt)
