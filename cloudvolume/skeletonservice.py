from collections import defaultdict
import copy
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
from .lib import red, Bbox
from .txrx import cdn_cache_control
from .storage import Storage, SimpleStorage

class SkeletonUnassignedEdgeError(Exception):
  pass

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

  @classmethod
  def from_path(kls, vertices):
    """
    Given an Nx3 array of vertices that constitute a single path, 
    generate a skeleton with appropriate edges.
    """
    if vertices.shape[0] == 0:
      return PrecomputedSkeleton()

    skel = PrecomputedSkeleton(vertices)
    edges = np.zeros(shape=(skel.vertices.shape[0] - 1, 2), dtype=np.uint32)
    edges[:,0] = np.arange(skel.vertices.shape[0] - 1)
    edges[:,1] = np.arange(1, skel.vertices.shape[0])
    skel.edges = edges
    return skel

  @classmethod
  def simple_merge(kls, skeletons):
    """
    Simple concatenation of skeletons into one object 
    without adding edges between them.
    """
    if len(skeletons) == 0:
      return PrecomputedSkeleton()

    if type(skeletons[0]) is np.ndarray:
      skeletons = [ skeletons ]

    ct = 0
    edges = []
    for skel in skeletons:
      edge = skel.edges + ct
      edges.append(edge)
      ct += skel.vertices.shape[0]

    return PrecomputedSkeleton(
      vertices=np.concatenate([ skel.vertices for skel in skeletons ], axis=0),
      edges=np.concatenate(edges, axis=0),
      radii=np.concatenate([ skel.radii for skel in skeletons ], axis=0),
      vertex_types=np.concatenate([ skel.vertex_types for skel in skeletons ], axis=0),
      segid=skeletons[0].id,
    )

  def merge(self, skel):
    """Combine with an additional skeleton and consolidate."""
    return PrecomputedSkeleton.simple_merge((self, skel)).consolidate()

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

  def crop(self, bbox):
    """
    Crop away all vertices and edges that lie outside of the given bbox.
    The edge counts as inside.

    Returns: new PrecomputedSkeleton
    """
    skeleton = self.clone()
    bbox = Bbox.create(bbox)

    if skeleton.empty():
      return skeleton

    nodes_valid_mask = np.array(
      [ bbox.contains(vtx) for vtx in skeleton.vertices ], dtype=np.bool
    )
    nodes_valid_idx = np.where(nodes_valid_mask)[0]

    # Set invalid vertices to be duplicates
    # so they'll be removed during consolidation
    if nodes_valid_idx.shape[0] == 0:
      return PrecomputedSkeleton()

    first_node = nodes_valid_idx[0]
    skeleton.vertices[~nodes_valid_mask] = skeleton.vertices[first_node]
  
    edges_valid_mask = np.isin(skeleton.edges, nodes_valid_idx)
    edges_valid_idx = edges_valid_mask[:,0] * edges_valid_mask[:,1] 
    skeleton.edges = skeleton.edges[edges_valid_idx,:]
    return skeleton.consolidate()

  def consolidate(self):
    """
    Remove duplicate vertices and edges from this skeleton without
    side effects.

    Returns: new consolidated PrecomputedSkeleton 
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
    eff_edges = eff_edges[ eff_edges[:,0] != eff_edges[:,1] ] # remove trivial loops

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

  def cable_length(self):
    """
    Returns cable length of connected skeleton vertices in the same
    metric that this volume uses (typically nanometers).
    """
    dist = 0
    for e1, e2 in self.edges:
      try:
        v1, v2 = self.vertices[e1], self.vertices[e2]
      except IndexError:
        raise SkeletonUnassignedEdgeError(
          "Edge ({},{}) points to an index outside the number of vertices ({}).".format(
            e1, e2, self.vertices.shape[0]
          )
        )
      dist += np.linalg.norm(v2 - v1)

    return dist

  def downsample(self, factor, preserve_endpoints=True):
    """
    Compute a downsampled version of the skeleton by striding while 
    preserving endpoints.

    factor: stride length for downsampling the saved skeleton paths.
    preserve_endpoints: ensure that regardless of the downsample factor, 
      the final vertex and edge on each tree branch is preserved.

    Returns: downsampled PrecomputedSkeleton
    """
    if int(factor) != factor or factor < 1:
      raise ValueError("Argument `factor` must be a positive integer greater than or equal to 1. Got: <{}>({})", type(factor), factor)

    paths = self.paths()

    for i, path in enumerate(paths):
      if preserve_endpoints:
        paths[i] = np.concatenate(
          (path[0::factor, :], path[-1:, :])
        )
      else:
        paths[i] = path[0::factor, :]

    ds_skel = PrecomputedSkeleton.simple_merge(
      [ PrecomputedSkeleton.from_path(path) for path in paths ]
    ).consolidate()
    ds_skel.id = self.id

    # TODO: I'm sure this could be sped up if need be.
    index = {}
    for i, vert in enumerate(self.vertices):
      vert = tuple(vert)
      index[vert] = i

    for i, vert in enumerate(ds_skel.vertices):
      vert = tuple(vert)
      ds_skel.radii[i] = self.radii[index[vert]]
      ds_skel.vertex_types[i] = self.vertex_types[index[vert]]

    return ds_skel

  def _single_tree_paths(self, tree):
    """Get all traversal paths from a single tree."""
    skel = tree.consolidate()

    tree = defaultdict(list)

    for edge in skel.edges:
      svert = edge[0]
      evert = edge[1]
      tree[svert].append(evert)
      tree[evert].append(svert)

    def dfs(path, visited, paths):
      vertex = path[-1]
      children = tree[vertex]
      
      visited[vertex] = True

      children = [ child for child in children if not visited[child] ]

      if len(children) == 0:
        paths.append(path)

      for child in children:
        dfs(path + [child], copy.deepcopy(visited), paths)

      return paths
      
    root = skel.edges[0,0]
    paths = dfs([root], defaultdict(bool), [])

    root = np.argmax([ len(_) for _ in paths ])
    root = paths[root][-1]
  
    paths = dfs([ root ], defaultdict(bool), [])
    
    return [ np.flip(skel.vertices[path], axis=0) for path in paths ]    

  def paths(self):
    """
    Assuming the skeleton is structured as a single tree, return a 
    list of all traversal paths across all components. For each component, 
    start from the first vertex, find the most distant vertex by 
    hops and set that as the root. Then use depth first traversal 
    to produce paths.

    Returns: [ [(x,y,z), (x,y,z), ...], path_2, path_3, ... ]
    """
    paths = []
    for tree in self.components():
      paths += self._single_tree_paths(tree)
    return paths

  def _compute_components(self):
    skel = self.consolidate()
    if skel.edges.size == 0:
      return skel, []

    index = defaultdict(list)
    visited = defaultdict(bool)
    for e1, e2 in skel.edges:
      index[e1].append(e2)
      index[e2].append(e1)

    def extract_component(start):
      tree = set()
      stack = [ start ]

      while stack:
        node = int(stack.pop(0))
        visited[node] = True
        tree.add(node)
        for child in index[node]:
          if not visited[child]:
            stack.append(child)

      return tree

    forest = []
    for edge in np.unique(skel.edges.flatten()):
      if visited[edge]:
        continue

      component = extract_component(edge)
      forest.append(component)

    return skel, forest

  def components(self):
    """
    Extract connected components from graph. 
    Useful for ensuring that you're working with a single tree.

    Returns: [ PrecomputedSkeleton, PrecomputedSkeleton, ... ]
    """
    skel, forest = self._compute_components()

    if len(forest) == 0:
      return []
    elif len(forest) == 1:
      return [ skel ]

    orig_verts = {}
    for i, coord in enumerate(skel.vertices):
      orig_verts[tuple(coord)] = i

    skeletons = []
    for component in forest:
      edge_list = []
      for e1, e2 in skel.edges:
        if e1 in component:
          edge_list.append( (e1,e2) )

      edge_list = np.array(edge_list, dtype=np.uint32)
      vert_idx = np.unique(edge_list.flatten())
      vert_list = skel.vertices[vert_idx]
      radii = skel.radii[vert_idx]
      vtypes = skel.vertex_types[vert_idx]

      new_verts = {}
      for i, coord in enumerate(vert_list):
        new_verts[orig_verts[tuple(coord)]] = i

      for i in range(edge_list.shape[0]):
        edge_list[i, 0] = new_verts[edge_list[i, 0]]
        edge_list[i, 1] = new_verts[edge_list[i, 1]]

      skeletons.append(
        PrecomputedSkeleton(vert_list, edge_list, radii, vtypes, skel.id)
      )

    return skeletons

  @classmethod
  def from_swc(self, swcstr):
    lines = swcstr.split("\n")
    while re.match(r'[#\s]', lines[0][0]):
      lines.pop(0)

    vertices = []
    edges = []
    radii = []
    vertex_types = []

    vertex_index = {}
    label_index = {}
    for i, line in enumerate(lines):
      (vid, vtype, x, y, z, radius, parent_id) = line.split(" ")
      
      coord = tuple([ float(_) for _ in (x,y,z) ])
      vid = int(vid)
      parent_id = int(parent_id)

      vertex_index[coord] = i 
      label_index[vid] = coord

      vertices.append(coord)

      if parent_id >= 0:
        edges.append( (i, vertex_index[label_index[parent_id]]) )

      vertex_types.append(int(vtype))
      radii.append(float(radius))

    return PrecomputedSkeleton(vertices, edges, radii, vertex_types)

  def to_swc(self):
    """
    Prototype SWC file generator. 

    c.f. http://research.mssm.edu/cnic/swc.html
    """
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

    skel = self.clone()

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

  def __repr__(self):
    return str(self)

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
    