def content_type(encoding):
  if encoding == 'jpeg':
    return 'image/jpeg'
  elif encoding in ('compressed_segmentation', 'fpzip', 'kempressed'):
    return 'image/x.' + encoding 
  return 'application/octet-stream'

def should_compress(encoding, compress, cache, iscache=False):
  if iscache and cache.compress != None:
    return cache.compress

  if compress is None:
    return 'gzip' if encoding in ('raw', 'compressed_segmentation', 'compresso') else None
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
