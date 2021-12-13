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

DEFAULT_PORT = 8080

RANGE_RE = re.compile(r"^bytes=(\d+)?-(\d+)?$")

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

def parse_range_header(header):
  start, end = re.match(RANGE_RE, header).groups()
  return int(start), int(end)

class ViewerServerHandler(BaseHTTPRequestHandler):
  def __init__(self, cloudpath, *args):
    self.cloudpath = cloudpath
    BaseHTTPRequestHandler.__init__(self, *args)

  def do_OPTIONS(self):
    self.send_response(200, "ok")   
    self.send_header('Access-Control-Allow-Origin', '*')              
    self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    self.send_header("Access-Control-Allow-Headers", "range")
    self.end_headers()
    
  def do_GET(self):  
    if self.path.find('..') != -1:
      self.send_error(403, "Relative paths are not allowed.")
      raise ValueError("Relative paths are not allowed.")

    path = self.path[1:]

    query = { "path": path }
    if self.headers["Range"]:
      start, end = parse_range_header(self.headers["Range"])
      query["start"] = start
      query["end"] = end + 1
      if start >= end:
        self.send_error(416, f"{self.headers['Range']} had start >= end.")
        return

    data = CloudFiles(self.cloudpath).get(query)

    if data is None:
      self.send_error(404, '/' + path + ": Not Found")
      return 

    self.send_response(200)
    self.serve_data(data)

  def serve_data(self, data):
    self.send_header('Content-type', 'application/octet-stream')
    self.send_header('Access-Control-Allow-Origin', '*')
    self.send_header('Content-length', str(len(data)))
    self.end_headers()
    self.wfile.write(data)
