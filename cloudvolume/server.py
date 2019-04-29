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

from cloudvolume.storage import Storage
from cloudvolume.lib import Vec, Bbox, mkdir, save_images, ExtractedPath, yellow

DEFAULT_PORT = 8080

def view(cloudpath, hostname="localhost", port=DEFAULT_PORT):
  """Start a local web app on the given port that lets you explore this cutout."""
  def handler(*args):
    return ViewerServerHandler(cloudpath, *args)

  myServer = HTTPServer((hostname, port), handler)
  print("Neuroglancer server listening to http://{}:{}".format(hostname, port))
  try:
    myServer.serve_forever()
  except KeyboardInterrupt:
    # extra \n to prevent display of "^CContinuing"
    print("\nContinuing program execution...")
  finally:
    myServer.server_close()

class ViewerServerHandler(BaseHTTPRequestHandler):
  def __init__(self, cloudpath, *args):
    self.storage = Storage(cloudpath)
    BaseHTTPRequestHandler.__init__(self, *args)

  def __del__(self):
    self.storage.kill_threads()

  def do_GET(self):  
    if self.path.find('..') != -1:
      raise ValueError("Relative paths are not allowed.")

    path = self.path[1:]
    data = self.storage.get_file(path)

    if data is None:
      self.send_response(404)
      return 

    self.send_response(200)
    self.serve_data(data)

  def serve_data(self, data):
    self.send_header('Content-type', 'application/octet-stream')
    self.send_header('Access-Control-Allow-Origin', '*')
    self.send_header('Content-length', str(len(data)))
    self.end_headers()
    self.wfile.write(data)
