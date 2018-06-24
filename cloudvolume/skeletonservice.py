import re
import os

try:
  from StringIO import cStringIO as BytesIO
except ImportError:
  from io import BytesIO

import numpy as np
import struct

from . import lib
from .lib import red
from .storage import Storage, SimpleStorage

class SkeletonDecodeError(Exception):
  pass

class PrecomputedSkeleton(object):
  def __init__(self, vertices, edges, segid=None):
    self.id = segid

    self.vertices = vertices.astype(np.float32)

    if edges is None:
      self.edges = np.array([[]], dtype=np.uint32)
    else:
      self.edges = edges.astype(np.uint32)

  def encode(self):
    edges = self.edges
    vertices = self.vertices
    
    result = BytesIO()

    # Write number of positions and edges as first two uint32s
    result.write(struct.pack('<II', vertices.size // 3, edges.size // 2))
    result.write(vertices.tobytes('C'))
    result.write(edges.tobytes('C'))
    return result.getvalue()

  @classmethod
  def decode(kls, skelbuf, segid=None):
    if len(skelbuf) < 8:
      raise SkeletonDecodeError("{} bytes is fewer than needed to specify the number of verices and edges.".format(len(skelbuf)))

    num_vertices, num_edges = struct.unpack('<II', skelbuf[:8])
    format_length = 8 + 12 * num_vertices + 8 * num_edges

    if len(skelbuf) < format_length:
      raise SkeletonDecodeError("The input skeleton was {} bytes but the format requires {} bytes.".format(
        len(skelbuf), format_length
      ))

    vstart = 2 * 4 # two uint32s in
    vend = vstart + num_vertices * 3 * 4 # float32s
    vertbuf = skelbuf[ vstart : vend ]

    estart = vend
    eend = estart + num_edges * 4 * 2 # 2x uint32s

    edgebuf = skelbuf[ estart : eend ]

    vertices = np.frombuffer(vertbuf, dtype='<f4').reshape( (num_vertices, 3) )
    edges = np.frombuffer(edgebuf, dtype='<u4').reshape( (num_edges, 2) )

    return PrecomputedSkeleton(vertices, edges, segid=segid)

class PrecomputedSkeletonService(object):
  def __init__(self, vol):
    self.vol = vol

  @property
  def path(self):
    path = 'skeletons'
    if 'skeletons' in self.vol.info:
      path = self.vol.info['skeletons']
    return path

  def get(self, segid):
    with SimpleStorage(self.vol.layer_cloudpath) as stor:
      path = os.path.join(self.path, str(segid))
      skelbuf = stor.get_file(path)
    return PrecomputedSkeleton.decode(skelbuf, segid=segid)

  def upload(self, segid, vertices, edges):
    with SimpleStorage(self.vol.layer_cloudpath) as stor:
      path = os.path.join(self.path, str(segid))
      skel = PrecomputedSkeleton(vertices, edges, segid=segid).encode()
      
      stor.put_file(
        file_path='{}/{}'.format(self.path, segid),
        content=skel,
        compress='gzip',
      )
    
  def get_point_cloud(self, segid):
    with SimpleStorage(self.vol.layer_cloudpath) as stor:
      path = os.path.join(self.path, '{}.ptc').format(segid)
      buf = stor.get_file(path)
    points = np.frombuffer(buf, dtype=np.int32) \
      .reshape( (len(buf) // 3, 3) ) # xyz coordinate triplets
    return points


