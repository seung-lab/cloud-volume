from collections import defaultdict
import copy
import datetime
from io import BytesIO
import re
import os
import networkx as nx

import fastremap
import numpy as np
import struct

from . import lib
from .exceptions import (
  SkeletonDecodeError, SkeletonEncodeError, 
  SkeletonUnassignedEdgeError, SkeletonTransformError,
  SkeletonAttributeMixingError
)
from .lib import red, Bbox

IDENTITY = np.array([
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
], dtype=np.float32)

class Skeleton(object):
  """
  A stick figure representation of a 3D object. 

  vertices: [[x,y,z], ...] float32
  edges: [[v1,v2], ...] uint32
  radii: [r1,r2,...] float32 distance from vertex to nearest boudary
  vertex_types: [t1,t2,...] uint8 SWC vertex types
  segid: numerical ID
  transform: 3x4 scaling and translation matrix (ie homogenous coordinates) 
    that represents the transformaton from voxel to physical coordinates.
    
    Example Identity Matrix:
    [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0]
    ]

  space: 'voxel', 'physical', or user choice (but other choices 
    make .physical_space() and .voxel_space() stop working as they
    become meaningless.)
  
  extra_attributes: You can specify additional per vertex
    data attributes (the most common are radii and vertex_type) 
    that are present in reading Precomputed binary skeletons using
    the following format:
    [
        {
          "id": "radius",
          "data_type": "uint8",
          "num_components": 1,
        }
    ]

    These attributes will become object properties. i.e. skel.radius

    Note that for backwards compatibility, skel.radius is treated 
    specially and is synonymous with skel.radii.
  """
  def __init__(self, 
    vertices=None, edges=None, 
    radii=None, vertex_types=None, 
    segid=None, transform=None,
    space='voxel', extra_attributes=None
  ):
    self.id = segid
    self.space = space

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
      self.radius = -1 + np.zeros(shape=self.vertices.shape[0], dtype=np.float32)
    elif type(radii) is list:
      self.radius = np.array(radii, dtype=np.float32)
    else:
      self.radius = radii

    if vertex_types is None:
      # 0 = undefined in SWC (http://research.mssm.edu/cnic/swc.html)
      self.vertex_types = np.zeros(shape=self.vertices.shape[0], dtype=np.uint8)
    elif type(vertex_types) is list:
      self.vertex_types = np.array(vertex_types, dtype=np.uint8)
    else:
      self.vertex_types = vertex_types.astype(np.uint8)

    if extra_attributes is None:
      self.extra_attributes = self._default_attributes()
    else:
      self.extra_attributes = extra_attributes

    if transform is None:
      self.transform = np.copy(IDENTITY)
    else:
      self.transform = np.array(transform).reshape( (3, 4) )

  @classmethod
  def _default_attributes(self):
    return [
      {
        "id": "radius",
        "data_type": "float32",
        "num_components": 1,
      }, 
      {
        "id": "vertex_types",
        "data_type": "uint8",
        "num_components": 1,
      }
    ]

  def _check_space(self):
    if self.space not in ('physical', 'voxel'):
      raise SkeletonTransformError(
        """
        Loss of coordinate frame information. If the space is not 'physical' or 'voxel',
        the meaning of applying this transform matrix is unknown.

        space: {}
        """.format(self.space)
      )

  def physical_space(self, copy=True):
    """
    Convert skeleton vertices into a physical space 
    representation if it's not already there.

    copy: if False, don't copy if already in the correct
      coordinate frame.

    Returns: skeleton in physical coordinates
    """
    self._check_space()

    if self.space == 'physical':
      if copy:
        return self.clone()
      else:
        return self

    skel = self.clone()
    skel.apply_transform()
    skel.space = 'physical'
    return skel

  def voxel_space(self, copy=True):
    """
    Convert skeleton vertices into a voxel space 
    representation if it's not already there.

    copy: if False, don't copy if already in the correct
      coordinate frame.

    Returns: skeleton in voxel coordinates
    """
    self._check_space()

    if self.space == 'voxel':
      if copy:
        return self.clone()
      else:
        return self

    skel = self.clone()
    skel.apply_inverse_transform()
    skel.space = 'voxel'
    return skel

  @property
  def transform(self):
    return self._transform

  @transform.setter 
  def transform(self, val):
    self._transform = np.array(val, dtype=np.float32).reshape( (3,4) )

  def transform_vertices(self, vertices, transform):
    verts = np.append(
      vertices,
      np.ones( (vertices.shape[0], 1), dtype=vertices.dtype), 
      axis=1
    )
    verts = transform.dot(verts.T).T
    return verts[:,0:3]    

  def apply_transform(self):
    self.vertices = self.transform_vertices(self.vertices, self.transform)

  def apply_inverse_transform(self, transform=None):
    if transform is None:
      transform = self.transform

    verts = np.append(
      self.vertices, 
      np.ones( (self.vertices.shape[0], 1), dtype=self.vertices.dtype), 
      axis=1
    )
    
    transform = np.zeros( (3,4), dtype=np.float32 )
    transform[:3,:3] = np.linalg.inv(self.transform[:3,:3])
    transform[:,3] = -self.transform[:,3]

    verts = transform.dot(verts.T).T
    self.vertices = verts[:,0:3]    

  @property 
  def radii(self):
    return self.radius

  @radii.setter 
  def radii(self, val):
    self.radius = val

  @classmethod
  def from_path(kls, vertices):
    """
    Given an Nx3 array of vertices that constitute a single path, 
    generate a skeleton with appropriate edges.
    """
    if vertices.shape[0] == 0:
      return Skeleton()

    skel = Skeleton(vertices)
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
      return Skeleton()

    if type(skeletons[0]) is np.ndarray:
      skeletons = [ skeletons ]

    ct = 0
    edges = []
    for skel in skeletons:
      edge = skel.edges + ct
      edges.append(edge)
      ct += skel.vertices.shape[0]

    skel = Skeleton(
      vertices=np.concatenate([ skel.vertices for skel in skeletons ], axis=0),
      edges=np.concatenate(edges, axis=0),
      segid=skeletons[0].id,
    )

    if len(skeletons) == 0:
      return skel

    first_extra_attr = skeletons[0].extra_attributes
    for skl in skeletons[1:]:
      if first_extra_attr != skl.extra_attributes:
        raise SkeletonAttributeMixingError("""
          The skeletons were unable to be merged because
          the extended vertex attributes were not uniformly
          defined.

          Template being matched against:
          {}

          Non-matching skeleton:
          {}
        """.format(first_extra_attr, skl.extra_attributes))

    for attr in skeletons[0].extra_attributes:
      setattr(skel, attr['id'], np.concatenate([
        getattr(skl, attr['id']) for skl in skeletons
      ], axis=0))

    return skel

  def terminals(self):
    """
    Returns vertex indices that correspond to terminal 
    nodes of the skeleton defined as having only one edge.
    """
    unique_nodes, unique_counts = np.unique(self.edges, return_counts=True)
    return unique_nodes[ unique_counts == 1 ]

  def branches(self):
    """
    Returns vertex indices that correspond to branch points
    in the skeleton defined as having three or more edges.
    """
    unique_nodes, unique_counts = np.unique(self.edges, return_counts=True)
    return unique_nodes[ unique_counts >= 3 ]

  def merge(self, skel):
    """Combine with an additional skeleton and consolidate."""
    return Skeleton.simple_merge((self, skel)).consolidate()

  def empty(self):
    return self.vertices.size == 0 or self.edges.size == 0

  def encode(self):
    print(lib.yellow(
      "WARNING: Skeleton.encode() is deprecated in favor of Skeleton.to_precomputed() and will be removed in a future release."
    ))
    return self.to_precomputed()

  def decode(self, binary):
    print(lib.yellow(
      "WARNING: Skeleton.decode(bytes) is deprecated in favor of Skeleton.from_precomputed(bytes) and will be removed in a future release."
    ))
    return self.from_precomputed(binary)

  def to_precomputed(self):
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

    for attr in self.extra_attributes:
      arr = getattr(self, attr['id'])
      writeattr(arr, np.dtype(attr['data_type']), attr['id'])

    return result.getvalue()

  @classmethod
  def from_precomputed(kls, skelbuf, segid=None, vertex_attributes=None):
    """
    Convert a buffer into a Skeleton object.

    Format:
    num vertices (Nv) (uint32)
    num edges (Ne) (uint32)
    XYZ x Nv (float32)
    edge x Ne (2x uint32)

    Default Vertex Attributes:

      radii x Nv (optional, float32)
      vertex_type x Nv (optional, req radii, uint8) (SWC definition)

    Specify your own:

    vertex_attributes = [
      {
        'id': name of attribute,
        'num_components': int,
        'data_type': dtype,
      },
    ]

    More documentation: 
    https://github.com/seung-lab/cloud-volume/wiki/Advanced-Topic:-Skeletons-and-Point-Clouds
    """
    if len(skelbuf) < 8:
      raise SkeletonDecodeError("{} bytes is fewer than needed to specify the number of verices and edges.".format(len(skelbuf)))

    num_vertices, num_edges = struct.unpack('<II', skelbuf[:8])
    min_format_length = 8 + 12 * num_vertices + 8 * num_edges

    if len(skelbuf) < min_format_length:
      raise SkeletonDecodeError("The input skeleton was {} bytes but the format requires {} bytes.".format(
        len(skelbuf), min_format_length
      ))

    vstart = 2 * 4 # two uint32s in
    vend = vstart + num_vertices * 3 * 4 # float32s
    vertbuf = skelbuf[ vstart : vend ]

    estart = vend
    eend = estart + num_edges * 4 * 2 # 2x uint32s

    edgebuf = skelbuf[ estart : eend ]

    vertices = np.frombuffer(vertbuf, dtype='<f4').reshape( (num_vertices, 3) )
    edges = np.frombuffer(edgebuf, dtype='<u4').reshape( (num_edges, 2) )

    skeleton = Skeleton(vertices, edges, segid=segid)

    if len(skelbuf) == min_format_length:
      return skeleton

    if vertex_attributes is None:
      vertex_attributes = kls._default_attributes()

    start = eend
    end = -1
    for attr in vertex_attributes:
      num_components = int(attr['num_components'])
      data_type = np.dtype(attr['data_type'])
      end = start + num_vertices * num_components * data_type.itemsize
      attrbuf = np.frombuffer(skelbuf[start : end], dtype=data_type)

      if num_components > 1:
        attrbuf = attrbuf.reshape( (num_vertices, num_components) )

      setattr(skeleton, attr['id'], attrbuf)
      start = end

    skeleton.extra_attributes = vertex_attributes

    return skeleton

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

    vertex1, ct1 = np.unique(first.vertices, axis=0, return_counts=True)
    vertex2, ct2 = np.unique(second.vertices, axis=0, return_counts=True)
    
    vertex_match = np.all(np.abs(vertex1 - vertex2) < EPSILON)
    ct_match = np.all(ct1 == ct2)
    if not (vertex_match and ct_match):
      return False

    g1 = nx.Graph()
    g1.add_edges_from(first.edges)
    g2 = nx.Graph()
    g2.add_edges_from(second.edges)
    edges_match = nx.is_isomorphic(g1, g2)
    del g1 
    del g2

    if not edges_match:
      return False

    second_verts = {}
    for i, vert in enumerate(second.vertices):
      second_verts[tuple(vert)] = i
    
    attrs = [ attr['id'] for attr in first.extra_attributes ]
    for attr in attrs:
      buf1 = getattr(first, attr)
      buf2 = getattr(second, attr)
      if len(buf1) != len(buf2):
        return False

      for i in range(len(buf1)):
        i2 = second_verts[tuple(first.vertices[i])]
        if buf1[i] != buf2[i2]:
          return False

    return True

  def crop(self, bbox):
    """
    Crop away all vertices and edges that lie outside of the given bbox.
    The edge counts as inside.

    Returns: new Skeleton
    """
    skeleton = self.clone()
    bbox = Bbox.create(bbox)

    if skeleton.empty():
      return skeleton

    nodes_valid_mask = np.array(
      [ bbox.contains(vtx) for vtx in skeleton.vertices ], dtype=bool
    )
    nodes_valid_idx = np.where(nodes_valid_mask)[0]

    # Set invalid vertices to be duplicates
    # so they'll be removed during consolidation
    if nodes_valid_idx.shape[0] == 0:
      return Skeleton()

    first_node = nodes_valid_idx[0]
    skeleton.vertices[~nodes_valid_mask] = skeleton.vertices[first_node]
  
    edges_valid_mask = np.isin(skeleton.edges, nodes_valid_idx)
    edges_valid_idx = edges_valid_mask[:,0] * edges_valid_mask[:,1] 
    skeleton.edges = skeleton.edges[edges_valid_idx,:]
    return skeleton.consolidate()

  def consolidate(self, remove_disconnected_vertices=True):
    """
    Remove duplicate vertices and edges from this skeleton without
    side effects.

    Optional:
      remove_disconnected_vertices: delete vertices that have no edges
        associated with them. This does not preserve index order.

    Returns: new consolidated Skeleton 
    """
    nodes = self.vertices
    edges = self.edges 

    if self.empty():
      return Skeleton(segid=self.id)
    
    eff_vertices, uniq_idx, idx_representative = np.unique(
      nodes, axis=0, return_index=True, return_inverse=True
    )

    eff_edges = idx_representative[ edges ]
    eff_edges = np.sort(eff_edges, axis=1) # sort each edge [2,1] => [1,2]
    eff_edges = eff_edges[np.lexsort(eff_edges[:,::-1].T)] # Sort rows 
    eff_edges = np.unique(eff_edges, axis=0)
    eff_edges = eff_edges[ eff_edges[:,0] != eff_edges[:,1] ] # remove trivial loops

    skel = Skeleton(eff_vertices, eff_edges, segid=self.id)

    for attr in self.extra_attributes:
      name = attr['id']
      buf = getattr(self, name)
      eff_name = buf[ uniq_idx ]
      setattr(skel, name, eff_name)

    if remove_disconnected_vertices:
      return skel.remove_disconnected_vertices()

    return skel

  def remove_disconnected_vertices(self):
    """
    Delete vertices that have no edges associated with them. 
    This does not preserve index order.

    Returns: new Skeleton
    """
    if self.empty():
      return Skeleton(segid=self.id)

    idx_map = {}
    for i, vert in enumerate(self.vertices):
      idx_map[tuple(vert)] = i

    connected_verts = np.unique(self.vertices[ self.edges.flatten() ], axis=0)

    edge_map = np.zeros( (len(self.vertices),), dtype=self.edges.dtype)
    vertex_remap = np.zeros( (len(self.vertices),), dtype=np.int32) - 1
    for i, vert in enumerate(connected_verts):
      reverse_idx = idx_map[tuple(vert)]
      edge_map[reverse_idx] = i
      vertex_remap[i] = reverse_idx

    edges = np.sort(edge_map[self.edges], axis=1)
    vertex_remap = vertex_remap[ vertex_remap > -1 ]

    skel = Skeleton(connected_verts, edges, segid=self.id)

    if len(self.extra_attributes) == 0:
      return skel

    for attr in self.extra_attributes:
      name = attr['id']
      self_buf = getattr(self, name)
      skel_buf = self_buf[vertex_remap]
      setattr(skel, name, skel_buf)
        
    return skel

  def clone(self):
    vertices = np.copy(self.vertices)
    edges = np.copy(self.edges)
    radii = np.copy(self.radii)
    vertex_types = np.copy(self.vertex_types)

    skel = Skeleton(
      vertices, edges, radii, vertex_types, 
      segid=self.id, 
      space=self.space, 
      extra_attributes=self.extra_attributes,
      transform=np.copy(self.transform)
    )
    for attr in skel.extra_attributes:
      setattr(skel, attr['id'], np.copy(getattr(self, attr['id'])))

    return skel

  def cable_length(self):
    """
    Returns cable length of connected skeleton vertices in the same
    metric that this volume uses (typically nanometers).
    """
    skel = self.physical_space(copy=False)

    v1 = skel.vertices[skel.edges[:,0]]
    v2 = skel.vertices[skel.edges[:,1]]

    delta = (v2 - v1)
    delta *= delta
    dist = np.sum(delta, axis=1)
    dist = np.sqrt(dist)

    return np.sum(dist)

  def downsample(self, factor):
    """
    Compute a downsampled version of the skeleton by striding while 
    preserving endpoints.

    factor: stride length for downsampling the saved skeleton paths.

    Returns: downsampled Skeleton
    """
    if int(factor) != factor or factor < 1:
      raise ValueError("Argument `factor` must be a positive integer greater than or equal to 1. Got: <{}>({})", type(factor), factor)

    if factor == 1:
      return self.clone()

    paths = self.interjoint_paths()

    for i, path in enumerate(paths):
      paths[i] = np.concatenate(
        (path[0::factor, :], path[-1:, :]) # preserve endpoints
      )

    ds_skel = Skeleton.simple_merge(
      [ Skeleton.from_path(path) for path in paths ]
    ).consolidate()
    ds_skel.id = self.id

    # TODO: I'm sure this could be sped up if need be.
    index = {}
    for i, vert in enumerate(self.vertices):
      vert = tuple(vert)
      index[vert] = i

    bufs = [ getattr(ds_skel, attr['id']) for attr in ds_skel.extra_attributes ]
    orig_bufs = [ getattr(self, attr['id']) for attr in ds_skel.extra_attributes ]

    for i, vert in enumerate(ds_skel.vertices):
      reverse_i = index[tuple(vert)]
      for buf, buf_rev in zip(bufs, orig_bufs):
        buf[i] = buf_rev[reverse_i]
    
    return ds_skel

  def _single_tree_paths(self, tree, return_indices):
    """Get all traversal paths from a single tree."""
    skel = tree.consolidate()

    tree = defaultdict(list)

    for edge in skel.edges:
      svert = edge[0]
      evert = edge[1]
      tree[svert].append(evert)
      tree[evert].append(svert)

    def dfs(path, visited):
      paths = []
      stack = [ (path, visited) ]
      
      while stack:
        path, visited = stack.pop(0)

        vertex = path[-1]
        children = tree[vertex]
        
        visited[vertex] = True

        children = [ child for child in children if not visited[child] ]

        if len(children) == 0:
          paths.append(path)

        for child in children:
          stack.append( 
            (path + [child], copy.deepcopy(visited))
          )

      return paths
      
    root = skel.edges[0,0]
    paths = dfs([root], defaultdict(bool))

    root = np.argmax([ len(_) for _ in paths ])
    root = paths[root][-1]
  
    paths = dfs([ root ], defaultdict(bool))

    if return_indices:
      return [ np.flip(path) for path in paths ]

    return [ np.flip(skel.vertices[path], axis=0) for path in paths ]

  def paths(self, return_indices=False):
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
      paths += self._single_tree_paths(tree, return_indices=return_indices)
    return paths

  def _single_tree_interjoint_paths(self, skeleton, return_indices):
    vertices = skeleton.vertices
    edges = skeleton.edges

    unique_nodes, unique_counts = fastremap.unique(edges, return_counts=True)
    terminal_nodes = unique_nodes[ unique_counts == 1 ]
    branch_nodes = set(unique_nodes[ unique_counts >= 3 ])
    
    critical_points = set(terminal_nodes)
    critical_points.update(branch_nodes)

    tree = defaultdict(set)

    for e1, e2 in edges:
      tree[e1].add(e2)
      tree[e2].add(e1)

    # The below depth first search would be
    # more elegantly implemented as recursion,
    # but it quickly blows the stack, mandating
    # an iterative implementation.

    paths = []

    stack = [ terminal_nodes[0] ]
    criticals = [ terminal_nodes[0] ]
    # Saving the path stack is memory intensive
    # There might be a way to do it more linearly
    # via a DFS rather than BFS strategy.
    path_stack = [ [] ] 
    
    visited = defaultdict(bool)

    while stack:
      node = stack.pop()
      root = criticals.pop() # "root" is used v. loosely here
      path = path_stack.pop()

      path.append(node)
      visited[node] = True

      if node != root and node in critical_points:
        paths.append(path)
        path = [ node ]
        root = node

      for child in tree[node]:
        if not visited[child]:
          stack.append(child)
          criticals.append(root)
          path_stack.append(list(path))

    if return_indices:
      return paths

    return [ vertices[path] for path in paths ]

  def interjoint_paths(self, return_indices=False):
    """
    Returns paths between the adjacent critical points
    in the skeleton, where a critical point is the set of
    terminal and branch points.
    """
    paths = []
    for tree in self.components():
      subpaths = self._single_tree_interjoint_paths(
        tree, return_indices=return_indices
      )
      paths.extend(subpaths)

    return paths

  def _compute_components(self, skel):
    if skel.edges.size == 0:
      return skel, []

    index = defaultdict(set)
    visited = defaultdict(bool)
    for e1, e2 in skel.edges:
      index[e1].add(e2)
      index[e2].add(e1)

    def extract_component(start):
      edge_list = []
      stack = [ start ]
      parents = [ -1 ]

      while stack:
        node = stack.pop()
        parent = parents.pop()

        if node < parent:
          edge_list.append( (node, parent) )
        else:
          edge_list.append( (parent, node) )

        if visited[node]:
          continue

        visited[node] = True
        
        for child in index[node]:
          stack.append(child)
          parents.append(node)

      return np.unique(edge_list[1:], axis=0)

    forest = []
    for edge in fastremap.unique(skel.edges.flatten()):
      if visited[edge]:
        continue

      forest.append(
        extract_component(edge)
      )

    return forest
  
  def components(self):
    """
    Extract connected components from graph. 
    Useful for ensuring that you're working with a single tree.

    Returns: [ Skeleton, Skeleton, ... ]
    """
    skel = self.clone()
    forest = self._compute_components(skel)
    
    if len(forest) == 0:
      return []
    elif len(forest) == 1:
      return [ skel ]

    skeletons = []
    for edge_list in forest:
      edge_list = np.array(edge_list, dtype=np.uint32)
      vert_idx = fastremap.unique(edge_list)

      vert_list = skel.vertices[vert_idx]
      radii = skel.radii[vert_idx]
      vtypes = skel.vertex_types[vert_idx]

      remap = { vid: i for i, vid in enumerate(vert_idx) }
      edge_list = fastremap.remap(edge_list, remap, in_place=True)

      skeletons.append(
        Skeleton(vert_list, edge_list, radii, vtypes, skel.id)
      )

    return skeletons

  @classmethod
  def from_swc(self, swcstr):
    """
    The SWC format was first defined in 
    
    R.C Cannona, D.A Turner, G.K Pyapali, H.V Wheal. 
    "An on-line archive of reconstructed hippocampal neurons".
    Journal of Neuroscience Methods
    Volume 84, Issues 1-2, 1 October 1998, Pages 49-54
    doi: 10.1016/S0165-0270(98)00091-0

    This website is also helpful for understanding the format:

    https://web.archive.org/web/20180423163403/http://research.mssm.edu/cnic/swc.html

    Returns: Skeleton
    """
    lines = swcstr.split("\n")
    while len(lines) and (lines[0] == '' or re.match(r'[#\s]', lines[0][0])):
      l = lines.pop(0)

    if len(lines) == 0:
      return Skeleton()

    vertices = []
    edges = []
    radii = []
    vertex_types = []

    label_index = {}
    
    N = 0

    for line in lines:
      if line.replace(r"\s", '') == '':
        continue
      (vid, vtype, x, y, z, radius, parent_id) = line.split(" ")
      
      coord = tuple([ float(_) for _ in (x,y,z) ])
      vid = int(vid)
      parent_id = int(parent_id)

      label_index[vid] = N

      if parent_id >= 0:
        if vid < parent_id:
          edge = [vid, parent_id]
        else:
          edge = [parent_id, vid]

        edges.append(edge)

      vertices.append(coord)
      vertex_types.append(int(vtype))

      try:
        radius = float(radius)
      except ValueError:
        radius = -1 # e.g. radius = NA or N/A

      radii.append(radius)

      N += 1

    for edge in edges:
      edge[0] = label_index[edge[0]]
      edge[1] = label_index[edge[1]]

    return Skeleton(vertices, edges, radii, vertex_types)

  def to_swc(self, contributors=""):
    """
    Prototype SWC file generator. 

    The SWC format was first defined in 
    
    R.C Cannona, D.A Turner, G.K Pyapali, H.V Wheal. 
    "An on-line archive of reconstructed hippocampal neurons".
    Journal of Neuroscience Methods
    Volume 84, Issues 1-2, 1 October 1998, Pages 49-54
    doi: 10.1016/S0165-0270(98)00091-0

    This website is also helpful for understanding the format:

    https://web.archive.org/web/20180423163403/http://research.mssm.edu/cnic/swc.html

    Returns: swc as a string
    """
    from . import __version__
    sx, sy, sz = np.diag(self.transform)[:3]

    swc_header = f"""# ORIGINAL_SOURCE CloudVolume {__version__}
# CREATURE 
# REGION
# FIELD/LAYER
# TYPE
# CONTRIBUTOR {contributors}
# REFERENCE
# RAW 
# EXTRAS 
# SOMA_AREA
# SHINKAGE_CORRECTION 
# VERSION_NUMBER {__version__}
# VERSION_DATE {datetime.datetime.utcnow().isoformat()}
# SCALE {sx:.6f} {sy:.6f} {sz:.6f}
"""

    def generate_swc(skel, offset):
      if skel.edges.size == 0:
        return ""

      index = defaultdict(set)
      visited = defaultdict(bool)
      for e1, e2 in skel.edges:
        index[e1].add(e2)
        index[e2].add(e1)

      stack = [ skel.edges[0,0] ]
      parents = [ -1 ]

      swc = ""

      while stack:
        node = stack.pop()
        parent = parents.pop()

        if visited[node]:
          continue

        swc += "{n} {T} {x:0.6f} {y:0.6f} {z:0.6f} {R:0.6f} {P}\n".format(
          n=(node + 1 + offset),
          T=skel.vertex_types[node],
          x=skel.vertices[node][0],
          y=skel.vertices[node][1],
          z=skel.vertices[node][2],
          R=skel.radii[node],
          P=parent if parent == -1 else (parent + 1 + offset),
        )

        visited[node] = True
        
        for child in index[node]:
          stack.append(child)
          parents.append(node)

      return swc

    skels = self.components()

    swc = swc_header + "\n"
    offset = 0
    for skel in skels:
      swc += generate_swc(skel, offset) + "\n"
      offset += skel.vertices.shape[0]

    return swc

  def viewer(
    self, units='nm', 
    draw_edges=True, draw_vertices=True,
    color_by='radius'
  ):
    """
    View the skeleton with a radius heatmap. 

    Requires the matplotlib library which is 
    not installed by default.

    units: label axes with these units
    draw_edges: draw lines between vertices (more useful when skeleton is sparse)
    draw_vertices: draw each vertex colored by its radius.
    color_by: 
      'radius': color each vertex according to its radius attribute
        aliases: 'r', 'radius', 'radii'
      'component': color connected components seperately
        aliases: 'c', 'component', 'components'
      anything else: draw everything black
    """
    try:
      import matplotlib.pyplot as plt
      from mpl_toolkits.mplot3d import Axes3D 
      from matplotlib import cm
    except ImportError:
      print("Skeleton.viewer requires matplotlib. Try: pip install matplotlib --upgrade")
      return

    RADII_KEYWORDS = ('radius', 'radii', 'r')
    COMPONENT_KEYWORDS = ('component', 'components', 'c')

    fig = plt.figure(figsize=(10,10))
    ax = Axes3D(fig)
    ax.set_xlabel(units)
    ax.set_ylabel(units)
    ax.set_zlabel(units)

    # Set plot axes equal. Matplotlib doesn't have an easier way to
    # do this for 3d plots.
    X = self.vertices[:,0]
    Y = self.vertices[:,1]
    Z = self.vertices[:,2]

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ### END EQUALIZATION CODE ###

    component_colors = ['k', 'deeppink', 'dodgerblue', 'mediumaquamarine', 'gold' ]

    def draw_component(i, skel):
      component_color = component_colors[ i % len(component_colors) ]

      if draw_vertices:
        xs = skel.vertices[:,0]
        ys = skel.vertices[:,1]
        zs = skel.vertices[:,2]

        if color_by in RADII_KEYWORDS:
          colmap = cm.ScalarMappable(cmap=cm.get_cmap('rainbow'))
          colmap.set_array(skel.radii)

          normed_radii = skel.radii / np.max(skel.radii)
          yg = ax.scatter(xs, ys, zs, c=cm.rainbow(normed_radii), marker='o')
          cbar = fig.colorbar(colmap)
          cbar.set_label('radius (' + units + ')', rotation=270)
        elif color_by in COMPONENT_KEYWORDS:
          yg = ax.scatter(xs, ys, zs, color=component_color, marker='.')
        else:
          yg = ax.scatter(xs, ys, zs, color='k', marker='.')

      if draw_edges:
        for e1, e2 in skel.edges:
          pt1, pt2 = skel.vertices[e1], skel.vertices[e2]
          ax.plot(  
            [ pt1[0], pt2[0] ],
            [ pt1[1], pt2[1] ],
            zs=[ pt1[2], pt2[2] ],
            color=(component_color if not draw_vertices else 'silver'),
            linewidth=1,
          )

    if color_by in COMPONENT_KEYWORDS:
      for i, skel in enumerate(self.components()):
        draw_component(i, skel)
    else:
      draw_component(0, self)

    plt.show()

  def __eq__(self, other):
    if self.id != other.id:
      return False
    elif self.vertices.shape[0] != other.vertices.shape[0]:
      return False
    elif self.edges.shape[0] != other.edges.shape[0]:
      return False
    elif self.extra_attributes != other.extra_attributes:
      return False

    attrs = [ attr['id'] for attr in self.extra_attributes ]
    for attr in attrs:
      buf1 = getattr(self, attr)
      buf2 = getattr(other, attr)
      if np.all(buf1 != buf2):
        return False

    return (np.all(self.vertices == other.vertices)
      and np.all(self.edges == other.edges) \
      and np.all(self.radii == other.radii) \
      and np.all(self.vertex_types == other.vertex_types))

  def __str__(self):
    template = "{}=({}, {})"
    attr_strings = []
    for attr in self.extra_attributes:
      attr = attr['id']
      buf = getattr(self, attr)
      attr_strings.append(
        template.format(attr, buf.shape[0], buf.dtype)
      )

    return "Skeleton(segid={}, vertices=(shape={}, {}), edges=(shape={}, {}), {}, space='{}' transform={})".format(
      self.id,
      self.vertices.shape[0], self.vertices.dtype,
      self.edges.shape[0], self.edges.dtype,
      ', '.join(attr_strings),
      self.space, self.transform.tolist()
    )

  def __repr__(self):
    return str(self)


PrecomputedSkeleton = Skeleton # backwards compatibility
