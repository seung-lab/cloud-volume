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

from ....lib import red, toiter, Bbox, Vec
from ....mesh import Mesh
from .... import paths
from ....storage import SimpleStorage, Storage, GreenStorage

from ..sharding import GrapheneShardReader
from ...precomputed.sharding import ShardingSpecification

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
    StorageClass = GreenStorage if self.config.green else Storage
    progress = progress if progress is not None else self.config.progress

    with StorageClass(cloudpath, progress=progress) as stor:
      results = stor.files_exist(checks)

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
    labels = toiter(labels)
    labels = set(labels)

    dynamic_labels = self.dynamic_exists(labels, progress)
    remainder_labels = set([ label for label, path in dynamic_labels.items() if path ])

    initial_labels = self.initial_exists(remainder_labels, progress=progress, return_byte_range=False)

    dynamic_labels.update(initial_labels)
    return dynamic_labels

  def parse_manifest_filenames(self, filenames):
    lists = defaultdict(list)
    initial_regexp = re.compile(r'~(\d+)/([\d\-]+\.shard):(\d+):(\d+)')

    for filename in filenames:
      if not filename:
        continue

      # eg. ~2/344239114-0.shard:224659:442 
      # tilde means initial, missing tilde means dynamic
      initial = filename[0] == '~'

      if initial:
        (layer_id, parsed_filename, byte_start, size) = re.search(
          initial_regexp, filename
        ).groups()
        lists['initial'].append((layer_id, parsed_filename, int(byte_start), int(size)))
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
    StorageClass = GreenStorage if self.config.green else Storage

    filenames = self.get_fragment_filenames(seg_id, level=level, bbox=bounding_box)
    lists = self.parse_manifest_filenames(filenames)

    files = []
    if lists['dynamic']:
      with StorageClass(dynamic_cloudpath) as stor:
        files = stor.get_files(lists['dynamic'])

    meshes = [ 
      f['content'] for f in files 
    ]

    filenames = []
    starts = []
    ends = []
    for layer_id, filename, byte_start, size in lists['initial']:
      filenames.append(self.meta.join(layer_id, filename))
      starts.append(byte_start)
      ends.append(byte_start + size)

    cloudpath = self.meta.join(self.meta.meta.cloudpath, self.meta.mesh_path, 'initial')

    raw_binaries = []
    
    with StorageClass(cloudpath) as stor:
      initial_meshes = stor.get_files(filenames, starts, ends)

    meshes += initial_meshes

    return [ Mesh.from_draco(mesh['content']) for mesh in meshes ]

  def get_meshes_via_manifest_labels(self, seg_id, bounding_box):
    level = self.meta.meta.decode_layer_id(seg_id)
    labels = self.get_fragment_labels(seg_id, level=level, bbox=bounding_box)
    meshes = self.get_meshes_on_bypass(labels)
    return list(meshes.values())

  def get_meshes_via_manifest(self, seg_id, bounding_box, use_byte_offsets):
    if use_byte_offsets:
      return self.get_meshes_via_manifest_byte_offsets(seg_id, bounding_box)
    return self.get_meshes_via_manifest_labels(seg_id, bounding_box)

  def get_meshes_on_bypass(self, segids):
    """
    Attempt to fetch a mesh directly from storage without going through
    the chunk graph server. This capability should only be used in special
    circumstances.
    """
    segids = toiter(segids)
    StorageClass = GreenStorage if self.config.green else Storage
    dynamic_cloudpath = self.meta.join(self.meta.meta.cloudpath, self.dynamic_path())
    filenames = [ self.compute_filename(segid) for segid in segids ]
    with StorageClass(dynamic_cloudpath, progress=self.config.progress) as stor:
      raw_binaries = stor.get_files(filenames)

    # extract the label ID from the mesh manifest.
    # e.g. 387463568301300850:0:24576-25088_17920-18432_2048-3072
    label_regexp = re.compile(r'(\d+):\d:[\d_-]+$')

    output = {}
    remaining = []
    for res in raw_binaries:
      if res['error']:
        raise res['error']

      (label,) = re.search(label_regexp, res['filename']).groups()
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
      initial_output = self.readers[layer_id].get_data(labels, path=subdirectory, progress=self.config.progress)
      for label, raw_binary in initial_output.items():
        if raw_binary is None:
          raise IndexError('No mesh found for segment {}'.format(label))
      output.update(initial_output)

    return { label: Mesh.from_draco(raw_binary) for label, raw_binary in output.items() }

  def download_segid(self, seg_id, bounding_box, bypass=False, use_byte_offsets=True):    
    """See GrapheneUnshardedMeshSource.get for the user facing function."""
    level = self.meta.meta.decode_layer_id(seg_id)
    if level not in self.readers:
      raise KeyError("There is no shard configuration in the mesh info file for level {}.".format(level))

    if bypass:
      mesh = self.get_meshes_on_bypass(seg_id)[seg_id]
    else:
      meshes = self.get_meshes_via_manifest(seg_id, bounding_box, use_byte_offsets=use_byte_offsets)
      mesh = Mesh.concatenate(*meshes)

    if mesh is None:
      raise IndexError('No mesh found for segment {}'.format(seg_id))

    mesh.segid = seg_id
    return mesh, True
