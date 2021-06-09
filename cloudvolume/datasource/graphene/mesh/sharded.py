import six

from collections import defaultdict
import itertools
import json
import os
import posixpath
import re
import requests

import numpy as np
from tqdm import tqdm

from cloudfiles import CloudFiles

from ....lib import red, toiter, Bbox, Vec, first, nvl
from ....mesh import Mesh
from .... import paths

from ..sharding import GrapheneShardReader
from ...precomputed.sharding import ShardingSpecification
from ....mesh import is_draco_chunk_aligned
from .unsharded import GrapheneUnshardedMeshSource

class GrapheneShardedMeshSource(GrapheneUnshardedMeshSource):
  def __init__(self, mesh_meta, cache, config, readonly):
    super(GrapheneShardedMeshSource, self).__init__(mesh_meta, cache, config, readonly)

    self.readers = {}
    for level, sharding in self.meta.info['sharding'].items(): # { level: std sharding, ... }
      spec = ShardingSpecification.from_dict(sharding)
      self.readers[int(level)] = GrapheneShardReader(self.meta, self.cache, spec)

  def initial_path(self, level):
    return self.meta.join(self.meta.mesh_path, 'initial', str(level))

  def dynamic_path(self):
    return self.meta.join(self.meta.mesh_path, 'dynamic')

  # 1. determine if the segid is before or after the shard time point
  # 2. assuming it is sharded, fetch the draco encoded file from the
  #    correct level

  def dynamic_exists(self, labels, progress=None):
    """
    Checks for dynamic mesh existence.
  
    Returns: { label: path or None, ... }
    """
    labels = toiter(labels)

    checks = [ self.compute_filename(label) for label in labels ]
    
    cloudpath = self.meta.join(self.meta.meta.cloudpath, self.meta.mesh_path, 'dynamic') 
    progress = progress if progress is not None else self.config.progress

    results = CloudFiles(
      cloudpath, progress=progress, 
      green=self.config.green, secrets=self.config.secrets
    ).exists(checks)

    output = {}
    for filepath, exists in results.items():
      label = int(os.path.basename(filepath)[:-2]) # strip :0
      output[label] = filepath if exists else None

    return output

  def initial_exists(self, labels, return_byte_range=False, progress=None):
    """
    Checks for initial mesh existence.
  
    Returns: 
      If return_byte_range:
        { label: [ path, byte offset, byte size ] or None, ... }
      Else:
        { label: path or None, ... }
    """
    labels = toiter(labels)
    progress = progress if progress is not None else self.config.progress

    layers = defaultdict(list)
    for label in labels:
      if label == 0:
        continue
      layer = self.meta.meta.decode_layer_id(label)
      layers[layer].append(label)

    all_results = {}
    for layer, layer_labels in layers.items():
      path = self.initial_path(layer)
      results = self.readers[int(layer)].exists(
        layer_labels, 
        path=path, 
        return_byte_range=return_byte_range, 
        progress=progress
      )
      all_results.update(results)

    return all_results

  def exists(self, labels, progress=None):
    """
    Checks dynamic then initial meshes for existence.

    Returns: { label: path or None, ... }
    """
    labels = set(toiter(labels))

    dynamic_labels = self.dynamic_exists(labels, progress)
    remainder_labels = set([ label for label, path in dynamic_labels.items() if path ])

    initial_labels = self.initial_exists(remainder_labels, progress=progress, return_byte_range=False)

    dynamic_labels.update(initial_labels)
    return dynamic_labels

  def parse_manifest_filenames(self, manifest):
    lists = defaultdict(list)
    initial_regexp = re.compile(r'~(\d+)/([\d\-]+\.shard):(\d+):(\d+)')

    filenames, segids = manifest['fragments'], manifest['seg_ids']

    for filename, segid in zip(filenames, segids):
      if not filename:
        continue

      # eg. ~2/344239114-0.shard:224659:442 
      # tilde means initial, missing tilde means dynamic
      initial = filename[0] == '~'

      if initial:
        (layer_id, parsed_filename, byte_start, size) = re.search(
          initial_regexp, filename
        ).groups()
        lists['initial'].append((layer_id, parsed_filename, int(byte_start), int(size), int(segid)))
      else:
        lists['dynamic'].append(filename)
        
    return lists

  def get_meshes_via_manifest_byte_offsets(self, seg_id, bounding_box):
    """    
    The manifest for sharded is a bit strange in that exists(..., return_byte_offset=True)
    is being called on the server side. To avoid duplicative delay by recomputing the offset
    locations, the manifest breaks encapsulation by returning the shard filename and byte
    offsets. This breaks enapsulation of the shard fetching logic rather severely but 
    it is probably worth it.
    """
    level = self.meta.meta.decode_layer_id(seg_id)
    dynamic_cloudpath = self.meta.join(self.meta.meta.cloudpath, self.dynamic_path())
    if bounding_box is not None:
      level=2
    manifest = self.fetch_manifest(seg_id, level=level, bbox=bounding_box, return_segids=True)
    lists = self.parse_manifest_filenames(manifest)

    files = []
    if lists['dynamic']:
      files = CloudFiles(
        dynamic_cloudpath, 
        green=self.config.green, 
        secrets=self.config.secrets,
        parallel=self.config.parallel,
      ).get(lists['dynamic'])
    
    dynamic_meshes = []
    while files:
      f = files.pop()
      mesh = Mesh.from_draco(f['content'])
      mesh.segid = int(os.path.basename(f['path']).split(':')[0])
      dynamic_meshes.append(mesh)

    fetches = []
    segid_map = {}
    for layer_id, filename, byte_start, size, segid in lists['initial']:
      path = self.meta.join(layer_id, filename)
      byte_end = byte_start + size
      fetches.append({
        'path': path,
        'start': byte_start,
        'end': byte_end,
      })
      segid_map[(path, byte_start, byte_end)] = segid

    cloudpath = self.meta.join(self.meta.meta.cloudpath, self.meta.mesh_path, 'initial')
    files = CloudFiles(
      cloudpath, 
      green=self.config.green, 
      secrets=self.config.secrets,
      parallel=self.config.parallel,
    ).get(fetches)
    initial_meshes = []
    while files:
      f = files.pop()
      mesh = Mesh.from_draco(f['content'])
      start, end = f['byte_range']
      key = (f['path'], start, end)
      mesh.segid = segid_map[key]
      initial_meshes.append(mesh)    

    return dynamic_meshes + initial_meshes

  def get_meshes_via_manifest_labels(self, seg_id, bounding_box):
    level = None
    if bounding_box is not None:
      level=2
    labels = self.get_fragment_labels(seg_id, level=level, bbox=bounding_box)
    meshes = self.get_meshes_on_bypass(labels, allow_missing=True) # sometimes a tiny label won't get meshed
    return list(meshes.values())

  def get_meshes_via_manifest(self, seg_id, bounding_box, use_byte_offsets):
    if use_byte_offsets:
      return self.get_meshes_via_manifest_byte_offsets(seg_id, bounding_box)
    return self.get_meshes_via_manifest_labels(seg_id, bounding_box)

  def get_chunk_aligned_mask(self, meshes):
    meta = self.meta.meta
    first_mesh = first(meshes)
    if first_mesh is None:
      raise IndexError("No meshes found.")

    draco_grid_size = meta.get_draco_grid_size(
      meta.decode_layer_id(first_mesh.segid)
    )
    base_resolution = meta.resolution(self.config.mip)
    lvl2_resolution = meta.resolution(self.meta.mip)

    voxel_offset = Vec(0,0,0)
    if meta.chunks_start_at_voxel_offset:
      voxel_offset = meta.voxel_offset(self.meta.mip)
      
    offset = voxel_offset * lvl2_resolution
    lvl_2_size_nm = meta.chunk_size(self.meta.mip) * base_resolution

    chunk_aligned_masks = []
    for mesh in meshes:
      level = meta.decode_layer_id(mesh.segid)
      chunk_size = (lvl_2_size_nm * (2 ** (level-2))).astype(np.int32)
      verts = mesh.vertices - offset
      # find all vertices that are exactly on chunk_size boundaries
      is_chunk_aligned = is_draco_chunk_aligned(
        verts, chunk_size, draco_grid_size=draco_grid_size
      )
      chunk_aligned_masks.append(is_chunk_aligned)
    
    return np.concatenate(chunk_aligned_masks)

  def stitch_multi_level_draco_mesh_fragments(self, meshes, segid):
    chunk_aligned_mask = self.get_chunk_aligned_mask(meshes)
    mesh = Mesh.concatenate(*meshes)
    mesh.segid = segid
    return mesh.deduplicate_vertices(chunk_aligned_mask)

  def get_meshes_on_bypass(self, segids, allow_missing=False):
    """
    Attempt to fetch a mesh directly from storage without going through
    the chunk graph server. This capability should only be used in special
    circumstances.
    """
    segids = toiter(segids)

    dynamic_cloudpath = self.meta.join(self.meta.meta.cloudpath, self.dynamic_path())
    filenames = [ self.compute_filename(segid) for segid in segids ]

    cf = CloudFiles(
      dynamic_cloudpath, 
      progress=self.config.progress, 
      green=self.config.green,
      secrets=self.config.secrets,
      parallel=self.config.parallel,
    )
    raw_binaries = cf.get(filenames)

    # extract the label ID from the mesh manifest.
    # e.g. 387463568301300850:0:24576-25088_17920-18432_2048-3072
    label_regexp = re.compile(r'(\d+):\d:[\d_-]+$')

    output = {}
    remaining = []
    for res in raw_binaries:
      if res['error']:
        raise res['error']

      (label,) = re.search(label_regexp, res['path']).groups()
      label = int(label)

      if res['content'] is None:
        remaining.append(label)
      else:
        output[label] = res['content']

    layers = defaultdict(list)
    for segid in remaining:
      layer_id = self.meta.meta.decode_layer_id(segid)
      layers[layer_id].append(segid)

    for layer_id, labels in layers.items():
      subdirectory = self.meta.join(self.meta.mesh_path, 'initial', str(layer_id))
      initial_output = self.readers[layer_id].get_data(
        labels, path=subdirectory, 
        progress=self.config.progress,
        parallel=self.config.parallel,
      )
      for label, raw_binary in initial_output.items():
        if raw_binary is None:
          if allow_missing:
            continue
          else:
            raise IndexError('No mesh found for segment {}'.format(label))
        else:
          output[label] = raw_binary

    return { 
      label: Mesh.from_draco(raw_binary, segid=label) 
      for label, raw_binary in output.items() 
    }

  def download_segid(self, seg_id, bounding_box, bypass=False, use_byte_offsets=False):    
    """See GrapheneUnshardedMeshSource.get for the user facing function."""
    level = self.meta.meta.decode_layer_id(seg_id)
    if level not in self.readers:
      raise KeyError("There is no shard configuration in the mesh info file for level {}.".format(level))

    if bypass:
      mesh = self.get_meshes_on_bypass(seg_id)[seg_id]
    else:
      meshes = self.get_meshes_via_manifest(seg_id, bounding_box, use_byte_offsets=use_byte_offsets)
      mesh = self.stitch_multi_level_draco_mesh_fragments(meshes, seg_id)

    if mesh is None:
      raise IndexError('No mesh found for segment {}'.format(seg_id))

    mesh.segid = seg_id
    return mesh, True
