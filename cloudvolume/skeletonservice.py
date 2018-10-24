import datetime
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
from .txrx import cdn_cache_control
from .storage import Storage, SimpleStorage


class SkeletonDecodeError(Exception):
  pass

class SkeletonEncodeError(Exception):
  pass

class PrecomputedSkeleton(object):
  def __init__(self, 
    vertices=None, edges=None, 
    radii=None, vertex_types=None, 
    segid=None
  ):

    self.id = segid

    if vertices is None:
      self.vertices = np.array([[]], dtype=np.float32)
    elif type(vertices) is list:
      self.vertices = np.array(vertices, dtype=np.float32)
    else:
      self.vertices = vertices.astype(np.float32)

    if edges is None:
      self.edges = np.array([[]], dtype=np.uint32)
    elif type(edges) is list:
      self.edges = np.array(edges, dtype=np.uint32)
    else:
      self.edges = edges.astype(np.uint32)

    if radii is None:
      self.radii = -1 * np.ones(shape=self.vertices.shape[0], dtype=np.float32)
    elif type(radii) is list:
      self.radii = np.array(radii, dtype=np.float32)
    else:
      self.radii = radii

    if vertex_types is None:
      # 0 = undefined in SWC (http://research.mssm.edu/cnic/swc.html)
      self.vertex_types = np.zeros(shape=self.vertices.shape[0], dtype=np.uint8)
    elif type(vertex_types) is list:
      self.vertex_types = np.array(vertex_types, dtype=np.uint8)
    else:
      self.vertex_types = vertex_types.astype(np.uint8)

  def empty(self):
    return self.vertices.size == 0 or self.edges.size == 0

  def encode(self):
    edges = self.edges.astype(np.uint32)
    vertices = self.vertices.astype(np.float32)
    
    result = BytesIO()

    # Write number of positions and edges as first two uint32s
    result.write(struct.pack('<II', vertices.size // 3, edges.size // 2))
    result.write(vertices.tobytes('C'))
    result.write(edges.tobytes('C'))

    def writeattr(attr, dtype, text):
      if attr is None:
        return

      attr = attr.astype(dtype)

      if attr.shape[0] != vertices.shape[0]:
        raise SkeletonEncodeError("Number of {} {} ({}) must match the number of vertices ({}).".format(
          dtype, text, attr.shape[0], vertices.shape[0]
        ))
      
      result.write(attr.tobytes('C'))

    writeattr(self.radii, np.float32, 'Radii')
    writeattr(self.vertex_types, np.uint8, 'SWC Vertex Types')

    return result.getvalue()

  @classmethod
  def decode(kls, skelbuf, segid=None):
    """
    Convert a buffer into a PrecomputedSkeleton object.

    Format:
    num vertices (Nv) (uint32)
    num edges (Ne) (uint32)
    XYZ x Nv (float32)
    edge x Ne (2x uint32)
    radii x Nv (optional, float32)
    vertex_type x Nv (optional, req radii, uint8) (SWC definition)

    More documentation: 
    https://github.com/seung-lab/cloud-volume/wiki/Advanced-Topic:-Skeletons-and-Point-Clouds
    """
    if len(skelbuf) < 8:
      raise SkeletonDecodeError("{} bytes is fewer than needed to specify the number of verices and edges.".format(len(skelbuf)))

    num_vertices, num_edges = struct.unpack('<II', skelbuf[:8])
    min_format_length = 8 + 12 * num_vertices + 8 * num_edges

    if len(skelbuf) < min_format_length:
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

    if len(skelbuf) == min_format_length:
      return PrecomputedSkeleton(vertices, edges, segid=segid)

    radii_format_length = min_format_length + num_vertices * 4

    if len(skelbuf) < radii_format_length:
      raise SkeletonDecodeError("Input buffer did not have enough float32 radii to correspond to each vertex. # vertices: {}, # radii: {}".format(
        num_vertices, (radii_format_length - min_format_length) / 4
      ))

    rstart = eend
    rend = rstart + num_vertices * 4 # 4 bytes np.float32
    radiibuf = skelbuf[ rstart : rend ]
    radii = np.frombuffer(radiibuf, dtype=np.float32)

    if len(skelbuf) == radii_format_length:
      return PrecomputedSkeleton(vertices, edges, radii, segid=segid)

    type_format_length = radii_format_length + num_vertices * 1 

    if len(skelbuf) < type_format_length:
      raise SkeletonDecodeError("Input buffer did not have enough uint8 SWC vertex types to correspond to each vertex. # vertices: {}, # types: {}".format(
        num_vertices, (type_format_length - radii_format_length)
      ))

    tstart = rend
    tend = tstart + num_vertices
    typebuf = skelbuf[ tstart:tend ]
    vertex_types = np.frombuffer(typebuf, dtype=np.uint8)

    return PrecomputedSkeleton(vertices, edges, radii, vertex_types, segid=segid)

  @classmethod
  def equivalent(kls, first, second):
    """
    Tests that two skeletons are the same in form not merely that
    their array contents are exactly the same. This test can be
    made more sophisticated. 
    """
    if first.empty() and second.empty():
      return True
    elif first.vertices.shape[0] != second.vertices.shape[0]:
      return False
    elif first.edges.shape[0] != second.edges.shape[0]:
      return False

    EPSILON = 1e-7

    vertex_match = np.all(np.abs(first.vertices - second.vertices) < EPSILON)
    if not vertex_match:
      return False

    edges1 = np.sort(np.unique(first.edges, axis=0), axis=1)
    edges1 = edges1[np.lexsort(edges1[:,::-1].T)]
    edges2 = np.sort(np.unique(second.edges, axis=0), axis=1)
    edges2 = edges2[np.lexsort(edges2[:,::-1].T)]
    edges_match = np.all(edges1 == edges2)
    del edges1
    del edges2

    if not edges_match:
      return False

    radii_match = np.all(np.abs(first.radii - second.radii) < EPSILON)
    if not radii_match:
      return False   

    return np.all(first.vertex_types == second.vertex_types)

  def consolidate(self):
    """
    Remove duplicate vertices and edges from this skeleton without
    side effects.

    Returns: a consolidated skeleton 
    """
    nodes = self.vertices
    edges = self.edges 
    radii = self.radii
    vertex_types = self.vertex_types

    if self.empty():
      return PrecomputedSkeleton()
    
    eff_nodes, uniq_idx, idx_representative = np.unique(
      nodes, axis=0, return_index=True, return_inverse=True
    )

    edge_vector_map = np.vectorize(lambda x: idx_representative[x])
    eff_edges = edge_vector_map(edges)
    eff_edges = np.sort(eff_edges, axis=1) # sort each edge [2,1] => [1,2]
    eff_edges = eff_edges[np.lexsort(eff_edges[:,::-1].T)] # Sort rows 
    eff_edges = np.unique(eff_edges, axis=0)

    radii_vector_map = np.vectorize(lambda idx: radii[idx])
    eff_radii = radii_vector_map(uniq_idx)

    vertex_type_map = np.vectorize(lambda idx: vertex_types[idx])
    eff_vtype = vertex_type_map(uniq_idx)  
      
    return PrecomputedSkeleton(eff_nodes, eff_edges, eff_radii, eff_vtype, segid=self.id)

  def clone(self):
    vertices = np.copy(self.vertices)
    edges = np.copy(self.edges)
    radii = np.copy(self.radii)
    vertex_types = np.copy(self.vertex_types)

    return PrecomputedSkeleton(vertices, edges, radii, vertex_types, segid=self.id)

  def __eq__(self, other):
    if self.id != other.id:
      return False
    elif self.vertices.shape[0] != other.vertices.shape[0]:
      return False
    elif self.edges.shape[0] != other.edges.shape[0]:
      return False

    return (np.all(self.vertices == other.vertices, axis=0) \
      and np.any(self.edges == other.edges, axis=0) \
      and np.any(self.radii == other.radii) \
      and np.any(self.vertex_types == other.vertex_types))

  def __str__(self):
    return "PrecomputedSkeleton(segid={}, vertices=(shape={}, {}), edges=(shape={}, {}), radii=(shape={}, {}), vertex_types=(shape={}, {}))".format(
      self.id,
      self.vertices.shape[0], self.vertices.dtype,
      self.edges.shape[0], self.edges.dtype,
      self.radii.shape[0], self.radii.dtype,
      self.vertex_types.shape[0], self.vertex_types.dtype
    )

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

    if skelbuf is None:
      raise SkeletonDecodeError("File does not exist: {}".format(path))

    return PrecomputedSkeleton.decode(skelbuf, segid=segid)

  def upload(self, segid, vertices, edges, radii=None, vertex_types=None):
    with SimpleStorage(self.vol.layer_cloudpath) as stor:
      path = os.path.join(self.path, str(segid))
      skel = PrecomputedSkeleton(
        vertices, edges, radii, 
        vertex_types, segid=segid
      ).encode()

      stor.put_file(
        file_path='{}/{}'.format(self.path, segid),
        content=skel,
        compress='gzip',
        cache_control=cdn_cache_control(self.vol.cdn_cache),
      )
    
  def upload_multiple(self, skeletons):
    with Storage(self.vol.layer_cloudpath, progress=self.vol.progress) as stor:
      for skel in skeletons:
        path = os.path.join(self.path, str(skel.id))
        stor.put_file(
          file_path='{}/{}'.format(self.path, str(skel.id)),
          content=skel.encode(),
          compress='gzip',
          cache_control=cdn_cache_control(self.vol.cdn_cache),
        )
    

  def get_point_cloud(self, segid):
    with SimpleStorage(self.vol.layer_cloudpath) as stor:
      path = os.path.join(self.path, "{}.json".format(segid))
      frags = stor.get_json(path)

    ptcloud = np.array([], dtype=np.int32).reshape(0, 3)

    if frags is None:
      return ptcloud
    
    for frag in frags: 
      bbox = Bbox.from_filename(frag)
      img = vol[ bbox.to_slices() ][:,:,:,0]
      ptc = np.argwhere( img == segid )
      ptcloud = np.concatenate((ptcloud, ptc), axis=0)

    ptcloud.sort(axis=0) # sorts x column, but not y unfortunately
    return np.unique(ptcloud, axis=0)

  def swc(self, segid):
    """Prototype SWC file generator. 

    c.f. http://research.mssm.edu/cnic/swc.html"""
    from . import __version__
    swc = """# ORIGINAL_SOURCE CloudVolume {}
# CREATURE 
# REGION
# FIELD/LAYER
# TYPE
# CONTRIBUTOR {}
# REFERENCE
# RAW 
# EXTRAS 
# SOMA_AREA
# SHINKAGE_CORRECTION 
# VERSION_NUMBER 
# VERSION_DATE {}
# SCALE 1.0 1.0 1.0

""".format(
      __version__, 
      ", ".join([ str(_) for _ in self.vol.provenance.owners ]),
      datetime.datetime.utcnow().isoformat()
    )

    skel = self.vol.skeleton.get(segid)

    def parent(i):
      coords = np.where( skel.edges == i )
      edge = skel.edges[ coords[0][0] ]
      if edge[0] == i:
        return edge[1] + 1
      return edge[0] + 1

    for i in range(skel.vertices.shape[0]):
      line = "{n} {T} {x} {y} {z} {R} {P}".format(
          n=i+1,
          T=skel.vertex_types[i],
          x=skel.vertices[i][0],
          y=skel.vertices[i][1],
          z=skel.vertices[i][2],
          R=skel.radii[i],
          P=-1 if i == 0 else parent(i),
        )

      swc += line + '\n'

    return swc

