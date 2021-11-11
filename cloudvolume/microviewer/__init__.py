import os

from http.server import BaseHTTPRequestHandler, HTTPServer

import json
from six.moves import range

import numpy as np
from tqdm import tqdm

from cloudvolume.lib import Vec, Bbox, mkdir, save_images, yellow
from cloudvolume.paths import ExtractedPath

DEFAULT_PORT = 8080

def getresolution(img, resolution):
  if resolution is not None:
    return Vec(*resolution)
  else:
    try:
      return img.resolution
    except AttributeError:
      return Vec(0,0,0) 

def getoffset(img, offset):
  if offset is not None:
    return Vec(*offset)
  else:
    try:
      return img.bounds.minpt
    except AttributeError:
      return Vec(0,0,0)

def to_volumecutout(img, image_type, resolution=None, offset=None, hostname='localhost'):
  from cloudvolume.volumecutout import VolumeCutout
  if type(img) == VolumeCutout:
    try:
      img.dataset_name # check if it's an intact VolumeCutout
      return img
    except AttributeError:
      pass
  
  resolution = getresolution(img, resolution)
  offset = getoffset(img, offset)

  return VolumeCutout(
    buf=img,
    path=ExtractedPath('mem', hostname, '/', '', '', '', ''),
    cloudpath='IN MEMORY',
    resolution=resolution,
    mip=-1,
    layer_type=image_type,
    bounds=Bbox( offset, offset + Vec(*(img.shape[:3])) ),
    handle=None,
  )

def to3d(img):
  while len(img.shape) > 3:
    img = img[..., 0]
  while len(img.shape) < 3:
    img = img[..., np.newaxis]
  return img  

def hyperview(
    img, segmentation, resolution=None, offset=None,
    hostname='localhost', port=DEFAULT_PORT
  ):

  img = to3d(img)
  segmentation = to3d(segmentation)

  assert np.all(img.shape[:3] == segmentation.shape[:3])

  img = to_volumecutout(img, 'image', resolution, offset, hostname)
  segmentation = to_volumecutout(
    segmentation, 'segmentation', resolution, offset, hostname
  )

  return run([ img, segmentation ], hostname=hostname, port=port)

def view(
    img, segmentation=False, resolution=None, offset=None,
    hostname="localhost", port=DEFAULT_PORT
  ):
  from cloudvolume.volumecutout import VolumeCutout

  img = to3d(img)
  resolution = getresolution(img, resolution)
  offset = getoffset(img, offset)

  cutout = VolumeCutout(
    buf=img,
    path=ExtractedPath('mem', hostname, '/', '', '', '', ''),
    cloudpath='IN MEMORY',
    resolution=resolution,
    mip=-1,
    layer_type=('segmentation' if segmentation else 'image'),
    bounds=Bbox( offset, offset + Vec(*(img.shape[:3])) ),
    handle=None,
  )
  return run([ cutout ], hostname=hostname, port=port)

def run(cutouts, hostname="localhost", port=DEFAULT_PORT):
  """Start a local web app on the given port that lets you explore this cutout."""
  def handler(*args):
    return ViewerServerHandler(cutouts, *args)

  myServer = HTTPServer((hostname, port), handler)
  print("Viewer server listening to http://{}:{}".format(hostname, port))
  try:
    myServer.serve_forever()
  except KeyboardInterrupt:
    # extra \n to prevent display of "^CContinuing"
    print("\nContinuing program execution...")
  finally:
    myServer.server_close()

class ViewerServerHandler(BaseHTTPRequestHandler):
  def __init__(self, cutouts, *args):
    self.cutouts = cutouts
    BaseHTTPRequestHandler.__init__(self, *args)

  def do_GET(self):
    self.send_response(200)
  
    allowed_files = ('/', '/datacube.js', '/jquery-3.3.1.js', '/favicon.ico')

    if self.path in allowed_files:
      self.serve_file()
    elif self.path == '/parameters':
      self.serve_parameters()
    elif self.path == '/channel':
      self.serve_data(self.cutouts[0])
    elif self.path == '/segmentation':
      self.serve_data(self.cutouts[1])

  def serve_data(self, data):
    self.send_header('Content-type', 'application/octet-stream')
    self.send_header('Content-length', str(data.nbytes))
    self.end_headers()
    self.wfile.write(data.tobytes('F'))

  def serve_parameters(self):
    self.send_header('Content-type', 'application/json')
    self.end_headers()

    if len(self.cutouts) == 1:
      cutout = self.cutouts[0]
      msg = json.dumps({
        'viewtype': 'single',
        'dataset': cutout.dataset_name,
        'layer': cutout.layer,
        'layer_type': cutout.layer_type,
        'protocol': cutout.path.protocol,
        'cloudpath': [ cutout.cloudpath ],
        'mip': cutout.mip,
        'bounds': [ int(_) for _ in cutout.bounds.to_list() ],
        'resolution': cutout.resolution.tolist(),
        'data_types': [ str(cutout.dtype) ],
        'data_bytes': np.dtype(cutout.dtype).itemsize,
      })
    else:
      img, seg = self.cutouts
      msg = json.dumps({
        'viewtype': 'hyper',
        'dataset': img.dataset_name,
        'layers': [ img.layer, seg.layer ],
        'protocol': img.path.protocol,
        'cloudpath': [ img.cloudpath, seg.cloudpath ],
        'mip': img.mip,
        'bounds': [ int(_) for _ in img.bounds.to_list() ],
        'resolution': img.resolution.tolist(),
        'data_types': [ str(img.dtype), str(seg.dtype) ],
        'data_bytes': [ 
          np.dtype(img.dtype).itemsize,
          np.dtype(seg.dtype).itemsize
        ],
      })
    self.wfile.write(msg.encode('utf-8'))

  def serve_file(self):
    self.send_header('Content-type', 'text/html')
    self.end_headers()

    path = self.path.replace('/', '')

    if path == '':
      path = 'index.html'

    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, './' + path)
    with open(filepath, 'rb') as f:
      self.wfile.write(f.read())  
