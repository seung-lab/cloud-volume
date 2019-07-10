from collections import namedtuple
import re
import sys

from .exceptions import UnsupportedProtocolError
from .lib import yellow, toabs

ExtractedPath = namedtuple('ExtractedPath', 
  ('format', 'protocol', 'bucket', 'path', 'intermediate_path', 'dataset', 'layer')
)

ALLOWED_PROTOCOLS = [ 'gs', 'file', 's3', 'matrix', 'http', 'https' ]
ALLOWED_FORMATS = [ 'graphene', 'precomputed', 'boss' ] 

CLOUDPATH_ERROR = yellow("""
Cloud Path must conform to FORMAT://PROTOCOL://BUCKET/PATH
Examples: 
  precomputed://gs://test_bucket/em
  gs://test_bucket/em
  graphene://https://example.com/image/em

Supported Formats: None (precomputed), {}
Supported Protocols: {}

Cloud Path Recieved: {}
""").format(
  ", ".join(ALLOWED_FORMATS), ", ".join(ALLOWED_PROTOCOLS), '{}' # curry first two
)

def pop_protocol(cloudpath):
  protocol_re = re.compile(r'(\w+)://')

  match = re.match(protocol_re, cloudpath)

  if not match:
    return (None, cloudpath)

  (protocol,) = match.groups()
  cloudpath = re.sub(protocol_re, '', cloudpath, count=1)

  return (protocol, cloudpath)

def extract_format_protocol(cloudpath):
  error = UnsupportedProtocolError(CLOUDPATH_ERROR.format(cloudpath))
  
  (proto, cloudpath) = pop_protocol(cloudpath)
  
  if proto is None:
    raise error # e.g. ://test_bucket, test_bucket, wow//test_bucket

  fmt, protocol = None, None

  if proto in ALLOWED_PROTOCOLS:
    fmt = 'precomputed'
    protocol = proto 
  elif proto in ALLOWED_FORMATS:
    fmt = proto

  (proto, cloudpath) = pop_protocol(cloudpath)

  if proto in ALLOWED_FORMATS:
    raise error # e.g. gs://graphene://
  elif proto in ALLOWED_PROTOCOLS:
    if protocol is None:
      protocol = proto
    else:
      raise error # e.g. gs://gs:// 

  (proto, cloudpath) = pop_protocol(cloudpath)
  if proto is not None:
    raise error # e.g. gs://gs://gs://

  return (fmt, protocol, cloudpath)

def strict_extract(cloudpath, windows=None, disable_toabs=False):
  """
  Same as cloudvolume.paths.extract, but raise an additional 
  cloudvolume.exceptions.UnsupportedProtocolError
  if either dataset or layer is not set.

  Returns: ExtractedPath
  """
  path = extract(cloudpath, windows, disable_toabs)

  if path.dataset == '' or path.layer == '':
    raise UnsupportedProtocolError(CLOUDPATH_ERROR.format(cloudpath))

  return path

def extract(cloudpath, windows=None, disable_toabs=False):
  """
  Given a valid cloudpath of the form 
  format://protocol://bucket/.../dataset/layer

  Where format in: None, precomputed, boss, graphene
  Where protocol in: None, file, gs, s3, http(s), matrix

  Return an ExtractedPath which breaks out the components
  format, protocol, bucket, path, intermediate_path, dataset, layer

  Raise a cloudvolume.exceptions.UnsupportedProtocolError if the
  path does not conform to a valid path.

  Windows OS may handle file protocol paths slightly differently
  than other OSes.

  Returns: ExtractedPath
  """

  windows_file_re = re.compile(r'((?:\w:\\)[\d\w_\.\-]+(?:\\)?)') # for C:\what\a\great\path
  bucket_re = re.compile(r'^(/?[~\d\w_\.\-]+)/') # posix /what/a/great/path
  
  tail_re = re.compile(r'([\d\w_\.\-]+)/([\d\w_\.\-]+)/?$')
  windows_file_tail_re = re.compile(r'([\d\w_\.\-]+)\\([\d\w_\.\-]+)\\?$')

  error = UnsupportedProtocolError(CLOUDPATH_ERROR.format(cloudpath))

  if windows is None:
    windows = sys.platform == 'win32'

  if disable_toabs:
    abspath = lambda x: x # can't prepend linux paths when force testing windows
  else:
    abspath = toabs    

  fmt, protocol, cloudpath = extract_format_protocol(cloudpath)
  
  if protocol == 'file':
    cloudpath = abspath(cloudpath)
    if windows:
      bucket_re = windows_file_re
      tail_re = windows_file_tail_re

  match = re.match(bucket_re, cloudpath)
  if not match:
    raise error

  (bucket,) = match.groups()
  cloudpath = re.sub(bucket_re, '', cloudpath, count=1)

  match = re.search(tail_re, cloudpath)
  if not match:
    return ExtractedPath(fmt, protocol, bucket, cloudpath, '', '', '')

  dataset, layer = match.groups()
  intermediate_path = re.sub(tail_re, '', cloudpath)

  return ExtractedPath(
    fmt, protocol, bucket, 
    cloudpath, intermediate_path, dataset, layer
  )
