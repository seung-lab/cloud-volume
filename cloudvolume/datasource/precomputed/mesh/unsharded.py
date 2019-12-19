import six

from collections import defaultdict
import itertools
import json
import re
import os

import struct
import numpy as np
from tqdm import tqdm

from .... import exceptions
from ....lib import yellow, red, toiter
from ....mesh import Mesh
from ....storage import Storage
from ..spatial_index import CachedSpatialIndex

SEGIDRE = re.compile(r'\b(\d+):0.*?$')

def filename_to_segid(filename):
  matches = SEGIDRE.search(filename)
  if matches is None:
    raise ValueError("There was an issue with the fragment filename: " + filename)

  segid, = matches.groups()
  return int(segid)

class UnshardedLegacyPrecomputedMeshSource(object):
  def __init__(self, meta, cache, config, readonly=False):
    self.meta = meta
    self.cache = cache
    self.config = config

    self.readonly = bool(readonly)

    self.spatial_index = None
    if self.meta.spatial_index:
      self.spatial_index = CachedSpatialIndex(
        self.cache,
        cloudpath=self.meta.layerpath, 
        bounds=self.meta.meta.bounds(0) * self.meta.meta.resolution(0),
        chunk_size=self.meta.info['spatial_index']['chunk_size'],
      )

  @property
  def path(self):
    return self.meta.mesh_path

  def manifest_path(self, segid):
    mesh_json_file_name = str(segid) + ':0'
    return self.meta.join(self.path, mesh_json_file_name)

  def _get_manifests(self, segids):
    segids = toiter(segids)    
    paths = [ self.manifest_path(segid) for segid in segids ]
    fragments = self.cache.download(paths)

    contents = {}
    for filename, content in fragments.items():
      content = content.decode('utf8')
      content = json.loads(content)
      segid = filename_to_segid(filename)
      contents[segid] = content['fragments']

    return contents

  def _get_mesh_fragments(self, paths):
    paths = [ self.meta.join(self.path, path) for path in paths ]

    compress = self.config.compress
    if compress is None:
      compress = True

    fragments = self.cache.download(paths, compress=compress)
    fragments = [ (filename, content) for filename, content in fragments.items() ]
    fragments = sorted(fragments, key=lambda frag: frag[0]) # make decoding deterministic
    return fragments

  def _check_missing_manifests(self, segids):
    """Check if there are any missing mesh manifests prior to downloading."""
    manifest_paths = [ self.manifest_path(segid) for segid in segids ]
    with Storage(self.meta.cloudpath, progress=self.config.progress) as stor:
      exists = stor.files_exist(manifest_paths)

    dne = []
    for path, there in exists.items():
      if not there:
        (segid,) = re.search(r'(\d+):0$', path).groups()
        dne.append(segid)
    return dne

  def get(
      self, segids, 
      remove_duplicate_vertices=True, 
      fuse=True,
      chunk_size=None
    ):
    """
    Merge fragments derived from these segids into a single vertex and face list.

    Why merge multiple segids into one mesh? For example, if you have a set of
    segids that belong to the same neuron.

    segids: (iterable or int) segids to render into a single mesh

    Optional:
      remove_duplicate_vertices: bool, fuse exactly matching vertices
      fuse: bool, merge all downloaded meshes into a single mesh
      chunk_size: [chunk_x, chunk_y, chunk_z] if passed only merge at chunk boundaries
    
    Returns: Mesh object if fused, else { segid: Mesh, ... }
    """
    segids = toiter(segids)
    dne = self._check_missing_manifests(segids)

    if dne:
      missing = ', '.join([ str(segid) for segid in dne ])
      raise ValueError(red(
        'Segment ID(s) {} are missing corresponding mesh manifests.\nAborted.' \
        .format(missing)
      ))

    fragments = self._get_manifests(segids)
    fragments = fragments.values()
    fragments = list(itertools.chain.from_iterable(fragments)) # flatten
    fragments = self._get_mesh_fragments(fragments)

    # decode all the fragments
    meshdata = defaultdict(list)
    for frag in tqdm(fragments, disable=(not self.config.progress), desc="Decoding Mesh Buffer"):
      segid = filename_to_segid(frag[0])
      try:
        mesh = Mesh.from_precomputed(frag[1])
      except Exception:
        print(frag[0], 'had a problem.')
        raise
      meshdata[segid].append(mesh)

    if not fuse:
      return { segid: Mesh.concatenate(*meshes) for segid, meshes in six.iteritems(meshdata) }

    meshdata = [ (segid, mesh) for segid, mesh in six.iteritems(meshdata) ]
    meshdata = sorted(meshdata, key=lambda sm: sm[0])
    meshdata = [ mesh for segid, mesh in meshdata ]
    meshdata = list(itertools.chain.from_iterable(meshdata)) # flatten
    mesh = Mesh.concatenate(*meshdata)

    if not remove_duplicate_vertices:
      return mesh 

    if not chunk_size:
      return mesh.consolidate()

    if self.meta.mip is not None:
      mip = self.meta.mip
    else:
      # This will usually be wrong, but it's backwards compatible.
      # Throwing an exception instead would probably break too many
      # things.
      mip = self.config.mip

    if mip not in self.meta.meta.available_mips:
      raise exceptions.ScaleUnavailableError("mip {} is not available.".format(mip))

    resolution = self.meta.meta.resolution(mip)
    chunk_offset = self.meta.meta.voxel_offset(mip)

    return mesh.deduplicate_chunk_boundaries(
      chunk_size * resolution, is_draco=False,
      offset=(chunk_offset * resolution)
    )

  def save(self, segids, filepath=None, file_format='ply'):
    """
    Save one or more segids into a common mesh format as a single file.

    segids: int, string, or list thereof
    filepath: string, file-like, or None (optional)
    file_format: string (optional)
    
    Supported Formats: 'obj', 'ply', 'precomputed'
    """
    if type(segids) != list:
      segids = [segids]

    mesh = self.get(segids, fuse=True, remove_duplicate_vertices=True)

    if file_format == 'obj':
      data = mesh.to_obj()
    elif file_format == 'ply':
      data = mesh.to_ply()
    elif file_format == 'precomputed':
      data = mesh.to_precomputed()
    else:
      raise NotImplementedError('Only .obj, .ply, and precomputed are currently supported.')

    if not filepath:
      filepath = str(segids[0]) + "." + file_format
      if len(segids) > 1:
        filepath = "{}_{}.{}".format(segids[0], segids[-1], file_format)

    try:
      filepath.write(data)
    except AttributeError:
      with open(filepath, 'wb') as f:
        f.write(data)

  def get_bbox(self, bbox):
    if self.spatial_index is None:
      raise IndexError("A spatial index has not been created.")

    segids = self.spatial_index.query(bbox)
    return self.get(segids, fuse=False)
