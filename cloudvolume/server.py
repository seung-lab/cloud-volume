import os

from http.server import BaseHTTPRequestHandler, HTTPServer

import json
import re
from six.moves import range

import numpy as np
from tqdm import tqdm

from cloudfiles import CloudFiles

from cloudvolume.lib import Vec, Bbox, mkdir, save_images, yellow
from cloudvolume.paths import ExtractedPath

RANGE_REGEXP = re.compile(r'bytes=(\d+)-(\d+)')
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
    self.cloudpath = cloudpath
    BaseHTTPRequestHandler.__init__(self, *args)

  def do_OPTIONS(self):
    self.send_response(200, "ok")
    self.send_header('Access-Control-Allow-Origin', '*')
    self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
    self.send_header("Access-Control-Allow-Headers", "Content-Type, Range")
    self.end_headers()

  def do_GET(self):  
    if self.path.find('..') != -1:
      self.send_error(403, "Relative paths are not allowed.")
      raise ValueError("Relative paths are not allowed.")

    range_header = self.headers.get("Range", None)

    byte_range = (None, None)
    if range_header:
      byte_range = re.match(RANGE_REGEXP, range_header).groups()
      byte_range = [ int(x) for x in byte_range ]

    path = self.path[1:]
    cf = CloudFiles(self.cloudpath)
    data = cf[path, byte_range[0]:byte_range[1]]

    if data is None:
      self.send_error(404, '/' + path + ": Not Found")
      return 

    self.send_response(200)
    self.serve_data(data, byte_range)

  def serve_data(self, data, byte_range):
    self.send_header('Content-type', 'application/octet-stream')
    self.send_header('Access-Control-Allow-Origin', '*')
    self.send_header('Content-length', str(len(data)))

    if any(byte_range):
      self.send_header('Content-range', f"bytes {byte_range[0]}-{byte_range[1]}/*")

    self.end_headers()
    self.wfile.write(data)
