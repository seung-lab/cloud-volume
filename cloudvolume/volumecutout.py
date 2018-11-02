import os

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from six.moves import range

import numpy as np
from tqdm import tqdm

from .lib import mkdir, save_images

class VolumeCutout(np.ndarray):

  def __new__(cls, buf, path, cloudpath, resolution, mip, layer_type, bounds, handle, *args, **kwargs):
    return super(VolumeCutout, cls).__new__(cls, shape=buf.shape, buffer=np.asfortranarray(buf), dtype=buf.dtype, order='F')

  def __init__(self, buf, path, cloudpath, resolution, mip, layer_type, bounds, handle, *args, **kwargs):
    super(VolumeCutout, self).__init__()
    
    self.dataset_name = path.dataset
    self.layer = path.layer
    self.path = path
    self.resolution = resolution
    self.cloudpath = cloudpath
    self.mip = mip
    self.layer_type = layer_type
    self.bounds = bounds
    self.handle = handle

  def close(self):
    # This bizzare construction is because of this error:
    # Traceback (most recent call last):
    #   File "cloud-volume/cloudvolume/volumecutout.py", line 30, in __del__
    #     self.close()
    #   File "cloud-volume/cloudvolume/volumecutout.py", line 26, in close
    #     if self.handle and not self.handle.closed:
    # ValueError: mmap closed or invalid

    # However testing if it is closed does not throw an error. So we test
    # for closure and capture the exception if self.handle is None.

    try:
      if not self.handle.closed:
        self.handle.close()
    except AttributeError:
      pass

  def __del__(self):
    sup = super(VolumeCutout, self)
    if hasattr(sup, '__del__'):
      sup.__del__()
    self.close()

  @classmethod
  def from_volume(cls, volume, buf, bounds, handle=None):
    return VolumeCutout(
      buf=buf,
      path=volume.path,
      cloudpath=volume.cloudpath,
      resolution=volume.resolution,
      mip=volume.mip,
      layer_type=volume.layer_type,
      bounds=bounds,
      handle=handle,
    )

  @property
  def num_channels(self):
    return self.shape[3]

  def save_images(self, directory=None, axis='z', channel=None, global_norm=True, image_format='PNG'):
    """See cloudvolume.lib.save_images for more information."""
    if directory is None:
      directory = os.path.join('./saved_images', self.dataset_name, self.layer, str(self.mip), self.bounds.to_filename())

    return save_images(self, directory, axis, channel, global_norm, image_format)

  def viewer(self, port=8080):
    """Start a local web app on the given port that lets you explore this cutout."""
    def handler(*args):
      return ViewerServerHandler(self, *args)

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
        print(self.path)
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
