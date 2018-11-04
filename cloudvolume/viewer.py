import os

try: 
  from http.server import BaseHTTPRequestHandler, HTTPServer
except ImportError:
  from SocketServer import TCPServer as HTTPServer
  from BaseHTTPServer import BaseHTTPRequestHandler

import json
from six.moves import range

import numpy as np
from tqdm import tqdm

from .lib import Vec, Bbox, mkdir, save_images, ExtractedPath

def visualize(img, segmentation=False, port=8080):
  from . import VolumeCutout
  cutout = VolumeCutout(
    buf=img,
    path=ExtractedPath('mem', 'localhost', '/', '', ''),
    cloudpath='IN MEMORY',
    resolution=Vec(0, 0, 0),
    mip=-1,
    layer_type=('segmentation' if segmentation else 'image'),
    bounds=Bbox( (0,0,0), list(img.shape)[:3]),
    handle=None,
  )
  return run(cutout, port=port)

def run(cutout, port=8080):
  """Start a local web app on the given port that lets you explore this cutout."""
  def handler(*args):
    return ViewerServerHandler(cutout, *args)

  myServer = HTTPServer(('localhost', port), handler)
  print("Viewer running at http://localhost:" + str(port))
  myServer.serve_forever()
  myServer.server_close()

class ViewerServerHandler(BaseHTTPRequestHandler):
    def __init__(self, cutout, *args):
      self.cutout = cutout
      BaseHTTPRequestHandler.__init__(self, *args)

    def do_GET(self):
      self.send_response(200)
    
      if self.path == '/favicon.ico':
        return
      elif self.path in ('/', '/datacube.js', '/jquery-3.3.1.js'):
        self.serve_file()
      elif self.path == '/parameters':
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        msg = json.dumps({
          'dataset': self.cutout.dataset_name,
          'layer': self.cutout.layer,
          'layer_type': self.cutout.layer_type,
          'path': self.cutout.cloudpath,
          'mip': self.cutout.mip,
          'bounds': [ int(_) for _ in self.cutout.bounds.to_list() ],
          'resolution': self.cutout.resolution.tolist(),
          'data_type': str(self.cutout.dtype),
          'data_bytes': np.dtype(self.cutout.dtype).itemsize,
        })
        self.wfile.write(msg.encode('utf-8'))
      elif self.path == '/data':
        self.send_header('Content-type', 'application/octet-stream')
        self.send_header('Content-length', str(self.cutout.nbytes))
        self.end_headers()
        self.wfile.write(self.cutout.tobytes('F'))

    def serve_file(self):
      self.send_header('Content-type', 'text/html')
      self.end_headers()

      path = self.path.replace('/', '')

      if path == '':
        path = 'index.html'

      dirname = os.path.dirname(__file__)
      filepath = os.path.join(dirname, '../ext/volumeviewer/' + path)
      with open(filepath, 'rb') as f:
        self.wfile.write(f.read())  
