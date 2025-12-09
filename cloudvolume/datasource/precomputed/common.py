import math

import numpy as np

def content_type(encoding):
  if encoding == 'jpeg':
    return 'image/jpeg'
  elif encoding in ('compresso', 'compressed_segmentation', 'fpzip', 'kempressed', 'zfpc', 'crackle'):
    return 'image/x.' + encoding 
  return 'application/octet-stream'

def should_compress(encoding, compress, cache, iscache=False):
  if iscache and cache.compress != None:
    return cache.compress
  
  if compress is None:
    return 'gzip' if encoding in ('raw', 'compressed_segmentation', 'compresso', 'crackle') else None
  elif compress == True:
    return 'gzip'
  elif compress == False:
    return None
  else:
    return compress

def cdn_cache_control(val):
  """Translate cdn_cache into a Cache-Control HTTP header."""
  if val is None:
    return 'max-age=3600, s-max-age=3600'
  elif type(val) is str:
    return val
  elif type(val) is bool:
    if val:
      return 'max-age=3600, s-max-age=3600'
    else:
      return 'no-cache'
  elif type(val) is int:
    if val < 0:
      raise ValueError(
        'cdn_cache must be a positive integer, boolean, or string. Got: ' + str(val)
      )

    if val == 0:
      return 'no-cache'
    else:
      return 'max-age={}, s-max-age={}'.format(val, val)
  else:
    raise NotImplementedError(type(val) + ' is not a supported cache_control setting.')

def compressed_morton_code(gridpt, grid_size):
  if hasattr(gridpt, "__len__") and len(gridpt) == 0: # generators don't have len
    return np.zeros((0,), dtype=np.uint32)

  gridpt = np.asarray(gridpt, dtype=np.uint32)
  single_input = False
  if gridpt.ndim == 1:
    gridpt = np.atleast_2d(gridpt)
    single_input = True

  code = np.zeros((gridpt.shape[0],), dtype=np.uint64)
  num_bits = [ math.ceil(math.log2(size)) for size in grid_size ]
  j = np.uint64(0)
  one = np.uint64(1)

  if sum(num_bits) > 64:
    raise ValueError(f"Unable to represent grids that require more than 64 bits. Grid size {grid_size} requires {num_bits} bits.")

  max_coords = np.max(gridpt, axis=0)
  if np.any(max_coords >= grid_size):
    raise ValueError(f"Unable to represent grid points larger than the grid. Grid size: {grid_size} Grid points: {gridpt}")

  for i in range(max(num_bits)):
    for dim in range(3):
      if 2 ** i < grid_size[dim]:
        bit = (((np.uint64(gridpt[:, dim]) >> np.uint64(i)) & one) << j)
        code |= bit
        j += one

  if single_input:
    return code[0]
  return code

def morton_code_to_bbox(code, volume_bbox, chunk_size):
  chunk_size = Vec(*chunk_size)

  grid_size = np.ceil(volume_bbox.size3() / chunk_size).astype(np.int64)
  
  gridpt = morton_code_to_gridpt(code, grid_size)
  
  bbox = Bbox(gridpt, gridpt + 1) 
  bbox *= chunk_size
  bbox += volume_bbox.minpt
  return bbox

def morton_code_to_gridpt(code, grid_size):
  gridpt = np.zeros([3,], dtype=int)

  num_bits = [ math.ceil(math.log2(size)) for size in grid_size ]
  j = np.uint64(0)
  one = np.uint64(1)

  if sum(num_bits) > 64:
    raise ValueError(f"Unable to represent grids that require more than 64 bits. Grid size {grid_size} requires {num_bits} bits.")

  max_coords = np.max(gridpt, axis=0)
  if np.any(max_coords >= grid_size):
    raise ValueError(f"Unable to represent grid points larger than the grid. Grid size: {grid_size} Grid points: {gridpt}")

  code = np.uint64(code)

  for i in range(max(num_bits)):
    for dim in range(3):
      i = np.uint64(i)
      if 2 ** i < grid_size[dim]:
        bit = np.uint64((code >> j) & one)
        gridpt[dim] += (bit << i)
        j += one

  return gridpt