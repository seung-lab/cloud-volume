import decimal
from functools import reduce
import json
import math
import operator
import os
import random
import re 
import sys
import time
import types
import string 
from itertools import product

import numpy as np
from PIL import Image
from tqdm import tqdm

from .exceptions import OutOfBoundsError

if sys.version_info < (3,):
  integer_types = (int, long, np.integer)
  string_types = (str, basestring, unicode)
else:
  integer_types = (int, np.integer)
  string_types = (str,)

floating_types = (float, np.floating)

COLORS = {
  'RESET': "\033[m",
  'YELLOW': "\033[1;93m",
  'RED': '\033[1;91m',
  'GREEN': '\033[1;92m',
}

# Formula produces machine epsilon regardless of platform architecture
MACHINE_EPSILON = (7. / 3) - (4. / 3) - 1

class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    if isinstance(obj, np.integer):
      return int(obj)
    if isinstance(obj, np.floating):
      return float(obj)
    return json.JSONEncoder.default(self, obj)

def nvl(*args):
  for arg in args:
    if arg is not None:
      return arg
  return None

def first(lst):
  if isinstance(lst, types.GeneratorType):
    try:
      return next(lst)
    except StopIteration:
      return None
  try:
    return lst[0]
  except TypeError:
    try:
      return next(iter(lst))
    except StopIteration:
      return None
  except IndexError:
    return None

def sip(iterable, block_size):
  """Sips a fixed size from the iterable."""
  ct = 0
  block = []
  for x in iterable:
    ct += 1
    block.append(x)
    if ct == block_size:
      yield block
      ct = 0
      block = []

  if len(block) > 0:
    yield block

def toiter(obj, is_iter=False):
  if isinstance(obj, str) or isinstance(obj, dict):
    if is_iter:
      return [ obj ], False
    return [ obj ]

  try:
    iter(obj)
    if is_iter:
      return obj, True
    return obj 
  except TypeError:
    if is_iter:
      return [ obj ], False
    return [ obj ]

def duplicates(lst):
  dupes = []
  seen = set()
  for elem in lst:
    if elem in seen:
      dupes.append(elem)
    seen.add(elem)
  return set(dupes)

def jsonify(obj, **kwargs):
  return json.dumps(obj, cls=NumpyEncoder, **kwargs)

def green(text):
  return colorize('green', text)

def yellow(text):
  return colorize('yellow', text)

def red(text):
  return colorize('red', text)

def colorize(color, text):
  color = color.upper()
  return COLORS[color] + text + COLORS['RESET']

def generate_random_string(size=6):
  return ''.join(random.SystemRandom().choice(
    string.ascii_lowercase + string.digits) for _ in range(size)
  )

def toabs(path):
  path = os.path.expanduser(path)
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

def getprecision(num):
  try:
    return len(str(num).split('.')[1])
  except IndexError:
    return 0

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

def scatter(sequence, n):
  """Scatters elements of ``sequence`` into ``n`` blocks. Returns generator."""
  for i in range(n):
    yield sequence[i::n]

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
      pt.x, pt.y, pt.z = min(x, end_vec[0]), min(y, end_vec[1]), min(z, end_vec[2])
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
    raise OutOfBoundsError('Value {} cannot be outside of inclusive range {} to {}'.format(val,low,high))
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
      values = ",".join([ str(x) for x in self ])
      return f"Vec({values}, dtype={self.dtype})"

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


def floating(lst):
  return any(( isinstance(x, float) for x in lst ))

FILENAME_RE = re.compile(r'(-?\d+)-(-?\d+)_(-?\d+)-(-?\d+)_(-?\d+)-(-?\d+)(?:\.gz|\.br)?$')

class Bbox(object):
  __slots__ = [ 'minpt', 'maxpt', '_dtype' ]

  """Represents a three dimensional cuboid in space."""
  def __init__(self, a, b, dtype=None):
    if dtype is None:
      if floating(a) or floating(b):
        dtype = np.float32
      else:
        dtype = np.int32

    self.minpt = Vec(*[ min(ai,bi) for ai,bi in zip(a,b) ], dtype=dtype)
    self.maxpt = Vec(*[ max(ai,bi) for ai,bi in zip(a,b) ], dtype=dtype)

    self._dtype = np.dtype(dtype)

  @classmethod
  def deserialize(cls, bbx_data):
    bbx_data = json.loads(bbx_data)
    return Bbox.from_dict(bbx_data)

  def serialize(self):
    return json.dumps(self.to_dict())

  @property
  def ndim(self):
    return len(self.minpt)

  @property 
  def dtype(self):
    return self._dtype

  @classmethod
  def intersection(cls, bbx1, bbx2):
    result = Bbox( [ 0 ] * bbx1.ndim, [ 0 ] * bbx2.ndim )

    if not Bbox.intersects(bbx1, bbx2):
      return result
    
    for i in range(result.ndim):
      result.minpt[i] = max(bbx1.minpt[i], bbx2.minpt[i])
      result.maxpt[i] = min(bbx1.maxpt[i], bbx2.maxpt[i])

    return result

  @classmethod
  def intersects(cls, bbx1, bbx2):
    return np.all(bbx1.minpt < bbx2.maxpt) and np.all(bbx1.maxpt > bbx2.minpt)

  @classmethod
  def near_edge(cls, bbx1, bbx2, distance=0):
    return (
         np.any( np.abs(bbx1.minpt - bbx2.minpt) <= distance )
      or np.any( np.abs(bbx1.maxpt - bbx2.maxpt) <= distance )
    )

  @classmethod
  def create(cls, obj, context=None, bounded=False, autocrop=False):
    typ = type(obj)
    if typ is Bbox:
      obj = obj
    elif typ in (list, tuple):
      obj = Bbox.from_slices(obj, context, bounded, autocrop)
    elif typ is Vec:
      obj = Bbox.from_vec(obj)
    elif typ in string_types:
      obj = Bbox.from_filename(obj)
    elif typ is dict:
      obj = Bbox.from_dict(obj)
    else:
      raise NotImplementedError("{} is not a Bbox convertible type.".format(typ))

    if context and autocrop:
      obj = Bbox.intersection(obj, context)

    if context and bounded:
      if not context.contains_bbox(obj):
        raise OutOfBoundsError(
          "{} did not fully contain the specified bounding box {}.".format(
            context, obj
        ))

    return obj

  @classmethod
  def from_delta(cls, minpt, plus):
    return Bbox( minpt, Vec(*minpt) + plus )

  @classmethod
  def from_dict(cls, data):
    dtype = data['dtype'] if 'dtype' in data else np.float32
    return Bbox( data['minpt'], data['maxpt'], dtype=dtype)

  @classmethod
  def from_vec(cls, vec, dtype=int):
    return Bbox( (0,0,0), vec, dtype=dtype)

  @classmethod
  def from_filename(cls, filename, dtype=int):
    match = FILENAME_RE.search(os.path.basename(filename))

    if match is None:
      raise ValueError("Unable to decode bounding box from: " + str(filename))

    (xmin, xmax,
     ymin, ymax,
     zmin, zmax) = map(int, match.groups())

    return Bbox( (xmin, ymin, zmin), (xmax, ymax, zmax), dtype=dtype)

  @classmethod
  def from_slices(cls, slices, context=None, bounded=False, autocrop=False):
    if context:
      slices = context.reify_slices(
        slices, bounded=bounded, autocrop=autocrop
      )

    for slc in slices:
      if slc.step not in (None, 1):
        raise ValueError("Non-unitary steps are unsupported. Got: " + str(slc.step))

    return Bbox(
      [ slc.start for slc in slices ],
      [ (slc.start if slc.stop < slc.start else slc.stop) for slc in slices ]
    )

  @classmethod
  def from_list(cls, lst):
    """
    from_list(cls, lst)
    
    the first half of the values are the minpt, 
    the last half are the maxpt
    """
    half = len(lst) // 2 
    return Bbox( lst[:half], lst[half:] )

  @classmethod
  def from_points(cls, arr):
    """Create a Bbox from a point cloud arranged as:
      [
        [x,y,z],
        [x,y,z],
        ...
      ]
    """
    arr = np.array(arr, dtype=np.float32)

    mins = []
    maxes = []

    for i in range(arr.shape[1]):
      mins.append( np.min(arr[:,i]) )
      maxes.append( np.max(arr[:,i]) )

    return Bbox( mins, maxes, dtype=np.int64)

  def to_filename(self, precision=None):
    """
    Renders the Bbox as a string. For example:
    
    >>> Bbox([0,2,4],[1,3,5]).to_filename()
    > '0-1_2-3_4-5'

    If the data is floating point, adding a precision
    allows will round the numbers to that decimal place.
    """
    def render(x):
      if precision:
        return f"{round(x, precision):.{precision}f}"
      return str(x)

    return '_'.join(
      ( render(self.minpt[i]) + '-' + render(self.maxpt[i]) for i in range(self.ndim) )
    )

  def to_slices(self):
    return tuple([
      slice(int(self.minpt[i]), int(self.maxpt[i])) for i in range(self.ndim)
    ])

  def to_list(self):
    return list(self.minpt) + list(self.maxpt)

  def to_dict(self):
    return {
      'minpt': self.minpt.tolist(),
      'maxpt': self.maxpt.tolist(),
      'dtype': np.dtype(self.dtype).name,
    }

  def reify_slices(self, slices, bounded=True, autocrop=False):
    """
    Convert free attributes of a slice object 
    (e.g. None (arr[:]) or Ellipsis (arr[..., 0]))
    into bound variables in the context of this
    bounding box.

    That is, for a ':' slice, slice.start will be set
    to the value of the respective minpt index of 
    this bounding box while slice.stop will be set 
    to the value of the respective maxpt index.

    Example:
      bbx = Bbox( (-1,-2,-3), (1,2,3) )
      bbx.reify_slices( (np._s[:],) )
      
      >>> [ slice(-1,1,1), slice(-2,2,1), slice(-3,3,1) ]

    Returns: [ slice, ... ]
    """
    if isinstance(slices, integer_types) or isinstance(slices, floating_types):
      slices = [ slice(int(slices), int(slices)+1, 1) ]
    elif type(slices) == slice:
      slices = [ slices ]
    elif type(slices) == Bbox:
      slices = slices.to_slices()
    elif slices == Ellipsis:
      slices = []

    slices = list(slices)

    for index, slc in enumerate(slices):
      if slc == Ellipsis:
        fill = self.ndim - len(slices) + 1
        slices = slices[:index] +  (fill * [ slice(None, None, None) ]) + slices[index+1:]
        break

    while len(slices) < self.ndim:
      slices.append( slice(None, None, None) )

    # First three slices are x,y,z, last is channel. 
    # Handle only x,y,z here, channel seperately
    for index, slc in enumerate(slices):
      if isinstance(slc, integer_types) or isinstance(slc, floating_types):
        slices[index] = slice(int(slc), int(slc)+1, 1)
      elif slc == Ellipsis:
        raise ValueError("More than one Ellipsis operator used at once.")
      else:
        start = self.minpt[index] if slc.start is None else slc.start
        end = self.maxpt[index] if slc.stop is None else slc.stop 
        step = 1 if slc.step is None else slc.step

        if step < 0:
          raise ValueError('Negative step sizes are not supported. Got: {}'.format(step))

        if autocrop:
          start = clamp(start, self.minpt[index], self.maxpt[index])
          end = clamp(end, self.minpt[index], self.maxpt[index])
        # note: when unbounded, negative indicies do not refer to
        # the end of the volume as they can describe, e.g. a 1px
        # border on the edge of the beginning of the dataset as in
        # marching cubes.
        elif bounded:
          # if start < 0: # this is support for negative indicies
            # start = self.maxpt[index] + start         
          check_bounds(start, self.minpt[index], self.maxpt[index])
          # if end < 0: # this is support for negative indicies
          #   end = self.maxpt[index] + end
          check_bounds(end, self.minpt[index], self.maxpt[index])

        slices[index] = slice(start, end, step)

    return slices

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

  def size(self):
    return Vec(*(self.maxpt - self.minpt), dtype=self.dtype)

  def size3(self):
    return Vec(*(self.maxpt[:3] - self.minpt[:3]), dtype=self.dtype)

  def subvoxel(self):
    """
    Previously, we used bbox.volume() < 1 for testing
    if a bounding box was larger than one voxel. However, 
    if two out of three size dimensions are negative, the 
    product will be positive. Therefore, we first test that 
    the maxpt is to the right of the minpt before computing 
    whether conjunctioned with volume() < 1.

    Returns: boolean
    """
    return (not self.valid()) or self.volume() < 1

  def empty(self):
    """
    Previously, we used bbox.volume() <= 0 for testing
    if a bounding box was empty. However, if two out of 
    three size dimensions are negative, the product will 
    be positive. Therefore, we first test that the maxpt 
    is to the right of the minpt before computing whether 
    the bbox is empty and account for 20x machine epsilon 
    of floating point error.

    Returns: boolean
    """
    return (not self.valid()) or (self.volume() < (20 * MACHINE_EPSILON))

  def valid(self):
    return np.all(self.minpt <= self.maxpt)

  def volume(self):
    if np.issubdtype(self.dtype, np.integer):
      return self.size3().astype(np.int64).rectVolume()
    else:
      return self.size3().astype(np.float64).rectVolume()

  def center(self):
    return (self.minpt + self.maxpt) / 2.0

  def grow(self, amt):
    assert amt >= 0
    self.minpt -= amt
    self.maxpt += amt
    return self

  def shrink(self, amt):
    assert amt >= 0
    self.minpt += amt
    self.maxpt -= amt

    if not self.valid():
      raise ValueError("Cannot shrink bbox below zero volume.")

    return self

  def expand_to_chunk_size(self, chunk_size, offset=Vec(0,0,0, dtype=int)):
    """
    Align a potentially non-axis aligned bbox to the grid by growing it
    to the nearest grid lines.

    Required:
      chunk_size: arraylike (x,y,z), the size of chunks in the 
                    dataset e.g. (64,64,64)
      offset: arraylike (x,y,z) the origin of the coordinate system
        so that this offset can be accounted for in the grid line 
        calculation.
    Optional:
      offset: arraylike (x,y,z), the starting coordinate of the dataset
    """
    chunk_size = np.array(chunk_size, dtype=np.float32)
    result = self.clone()
    result = result - offset
    result.minpt = np.floor(result.minpt / chunk_size) * chunk_size
    result.maxpt = np.ceil(result.maxpt / chunk_size) * chunk_size 
    return (result + offset).astype(self.dtype)

  def shrink_to_chunk_size(self, chunk_size, offset=Vec(0,0,0, dtype=int)):
    """
    Align a potentially non-axis aligned bbox to the grid by shrinking it
    to the nearest grid lines.

    Required:
      chunk_size: arraylike (x,y,z), the size of chunks in the 
                    dataset e.g. (64,64,64)
      offset: arraylike (x,y,z) the origin of the coordinate system
        so that this offset can be accounted for in the grid line 
        calculation.
    Optional:
      offset: arraylike (x,y,z), the starting coordinate of the dataset
    """
    chunk_size = np.array(chunk_size, dtype=np.float32)
    result = self.clone()
    result = result - offset
    result.minpt = np.ceil(result.minpt / chunk_size) * chunk_size
    result.maxpt = np.floor(result.maxpt / chunk_size) * chunk_size 

    # If we are inside a single chunk, the ends
    # can invert, which tells us we should collapse
    # to a single point.
    if np.any(result.minpt > result.maxpt):
      result.maxpt = result.minpt.clone()

    return (result + offset).astype(self.dtype)

  def round_to_chunk_size(self, chunk_size, offset=Vec(0,0,0, dtype=int)):
    """
    Align a potentially non-axis aligned bbox to the grid by rounding it
    to the nearest grid lines.

    Required:
      chunk_size: arraylike (x,y,z), the size of chunks in the 
                    dataset e.g. (64,64,64)
      offset: arraylike (x,y,z) the origin of the coordinate system
        so that this offset can be accounted for in the grid line 
        calculation.
    Optional:
      offset: arraylike (x,y,z), the starting coordinate of the dataset
    """
    chunk_size = np.array(chunk_size, dtype=np.float32)
    result = self.clone()
    result = result - offset
    result.minpt = np.round(result.minpt / chunk_size) * chunk_size
    result.maxpt = np.round(result.maxpt / chunk_size) * chunk_size
    return (result + offset).astype(self.dtype)

  def num_chunks(self, chunk_size):
    """Computes the number of chunks inside this bbox for a given chunk size."""
    Nfn = lambda i: math.ceil((self.maxpt[i] - self.minpt[i]) / chunk_size[i])
    return reduce(operator.mul, map(Nfn, range(len(self.minpt))))

  def contains(self, point):
    """
    Tests if a point on or within a bounding box.

    Returns: boolean
    """
    return np.all(point >= self.minpt) and np.all(point <= self.maxpt)

  def contains_bbox(self, bbox):
    return self.contains(bbox.minpt) and self.contains(bbox.maxpt)

  def clone(self):
    return Bbox(self.minpt, self.maxpt, dtype=self.dtype)

  def astype(self, typ):
    tmp = self.clone()
    tmp.minpt = tmp.minpt.astype(typ)
    tmp.maxpt = tmp.maxpt.astype(typ)
    tmp._dtype = tmp.minpt.dtype 
    return tmp

  def transpose(self):
    return Bbox(self.minpt[::-1], self.maxpt[::-1])

  # note that operand can be a vector 
  # or a scalar thanks to numpy
  def __isub__(self, operand): 
    if isinstance(operand, Bbox):
      self.minpt = np.subtract(self.minpt, operand.minpt, casting="safe")
      self.maxpt = np.subtract(self.maxpt, operand.maxpt, casting="safe")
    else:
      self.minpt = np.subtract(self.minpt, operand, casting="safe")
      self.maxpt = np.subtract(self.maxpt, operand, casting="safe")

    return self.astype(self.minpt.dtype)

  def __sub__(self, operand):
    tmp = self.clone()
    return tmp.__isub__(operand)

  def __iadd__(self, operand):
    if isinstance(operand, Bbox):
      self.minpt = np.add(self.minpt, operand.minpt, casting="safe")
      self.maxpt = np.add(self.maxpt, operand.maxpt, casting="safe")
    else:
      self.minpt = np.add(self.minpt, operand, casting="safe")
      self.maxpt = np.add(self.maxpt, operand, casting="safe")

    return self

  def __add__(self, operand):
    tmp = self.clone()
    return tmp.__iadd__(operand)

  def __imul__(self, operand):
    self.minpt = np.multiply(self.minpt, operand, casting="safe")
    self.maxpt = np.multiply(self.maxpt, operand, casting="safe")
    self._dtype = self.minpt.dtype 
    return self

  def __mul__(self, operand):
    tmp = self.clone()
    tmp.minpt = np.multiply(tmp.minpt, operand, casting="safe")
    tmp.maxpt = np.multiply(tmp.maxpt, operand, casting="safe")
    return tmp.astype(tmp.minpt.dtype)

  def __idiv__(self, operand):
    if (
      isinstance(operand, float) \
      or self.dtype in (float, np.float32, np.float64) \
      or (hasattr(operand, 'dtype') and operand.dtype in (float, np.float32, np.float64))
    ):
      return self.__itruediv__(operand)
    else:
      return self.__ifloordiv__(operand)

  def __div__(self, operand):
    if (
      isinstance(operand, float) \
      or self.dtype in (float, np.float32, np.float64) \
      or (hasattr(operand, 'dtype') and operand.dtype in (float, np.float32, np.float64))
    ):

      return self.__truediv__(operand)
    else:
      return self.__floordiv__(operand)

  def __ifloordiv__(self, operand):
    self.minpt //= operand
    self.maxpt //= operand
    return self

  def __floordiv__(self, operand):
    tmp = self.astype(float)
    tmp.minpt //= operand
    tmp.maxpt //= operand
    return tmp.astype(int)

  def __itruediv__(self, operand):
    res = self.__truediv__(operand)
    self.minpt[:] = res.minpt[:]
    self.maxpt[:] = res.maxpt[:]
    return self

  def __truediv__(self, operand):
    tmp = self.clone()

    if isinstance(operand, int):
      operand = float(operand)

    tmp.minpt = Vec(*( tmp.minpt.astype(float) / operand ), dtype=float)
    tmp.maxpt = Vec(*( tmp.maxpt.astype(float) / operand ), dtype=float)
    return tmp.astype(tmp.minpt.dtype)

  def __ne__(self, other):
    return not (self == other)

  def __eq__(self, other):
    return np.array_equal(self.minpt, other.minpt) and np.array_equal(self.maxpt, other.maxpt)

  def __hash__(self):
    return int(''.join(map(str, map(int, self.to_list()))))

  def __repr__(self):
    return f"Bbox({list(self.minpt)},{list(self.maxpt)}, dtype={self.dtype})"

def save_images(
  image, directory=None, axis='z', 
  channel=None, global_norm=True, 
  image_format='PNG', progress=True):
  """
  Serialize a 3D or 4D array into a series of PNGs for visualization.

  image: A 3D or 4D numpy array. Supported dtypes: integer, float, boolean
  axis: 'x', 'y', 'z'
  channel: None, 0, 1, 2, etc, which channel to serialize. Does all by default.
  directory: override the default output directory
  global_norm: Normalize floating point volumes globally or per slice?
  image_format: 'PNG', 'JPEG', etc
  progress: display progress bars and messages

  Returns: the directory path written to
  """
  if directory is None:
    directory = os.path.join('./saved_images', 'default', 'default', '0', Bbox( (0,0,0), image.shape[:3] ).to_filename())
  
  mkdir(directory)

  if progress:
    print("Saving to {}".format(directory))

  indexmap = {
    'x': 0,
    'y': 1,
    'z': 2,
  }

  index = indexmap[axis]

  channel = slice(None) if channel is None else channel

  while image.ndim < 4:
    image = image[..., np.newaxis ]

  def normalize_float(img):
    img = np.copy(img)
    img[ img ==  np.inf ] = 0
    img[ img == -np.inf ] = 0
    lower, upper = img.min(), img.max()
    img = (img - lower) / (upper - lower) * 255.0
    return img.astype(np.uint8)

  if global_norm and np.issubdtype(image.dtype, np.floating):
    image = normalize_float(image)      

  for level in tqdm(range(image.shape[index]), desc="Saving Images", disable=(not progress)):
    if index == 0:
      img = image[level, :, :, channel ]
    elif index == 1:
      img = image[:, level, :, channel ]
    elif index == 2:
      img = image[:, :, level, channel ]
    else:
      raise IndexError("Index {} is not valid. Expected 0, 1, or 2.".format(index))

    while img.ndim < 3:
      img = img[..., np.newaxis ]

    num_channels = img.shape[2]

    for channel_index in range(num_channels):
      img2d = img[:, :, channel_index]

      if not global_norm and img2d.dtype in (np.float32, np.float64):
        img2d = normalize_float(img2d)

      # discovered that downloaded cube is in a weird rotated state.
      # it requires a 90deg counterclockwise rotation on xy plane (leaving z alone)
      # followed by a flip on Y
      if axis == 'z':
        img2d = np.flipud(np.rot90(img2d, 1)) 

      if img2d.dtype == np.uint8:
        img2d = Image.fromarray(img2d, 'L')
      elif img2d.dtype == np.bool:
        img2d = img2d.astype(np.uint8) * 255
        img2d = Image.fromarray(img2d, 'L')
      else:
        img2d = img2d.astype('uint32')
        img2d[:,:] |= 0xff000000 # for little endian abgr
        img2d = Image.fromarray(img2d, 'RGBA')

      file_index = str(level).zfill(3)
      filename = '{}.{}'.format(file_index, image_format.lower())
      if num_channels > 1:
        filename = '{}-{}'.format(channel_index, filename)

      path = os.path.join(directory, filename)
      img2d.save(path, image_format)

  return toabs(directory)