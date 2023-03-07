from typing import Union, Iterable, Optional

from collections import defaultdict
import itertools
import simdjson
import re
import os

import struct
import numpy as np
from tqdm import tqdm

from cloudfiles import CloudFiles

from .... import exceptions
from ....lib import yellow, red, toiter, sip
from ....mesh import Mesh
from ....types import CompressType
from ..spatial_index import CachedSpatialIndex
from .common import apply_transform

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

    if 'transform' not in self.meta.info:
      self.transform = np.eye(4)
    else:
      self.transform = np.array(self.meta.info['transform'] + [0,0,0,1]).reshape(4,4)

    self.spatial_index = None
    if self.meta.spatial_index:
      mip = self.meta.mip or 0
      self.spatial_index = CachedSpatialIndex(
        self.cache, self.config,
        cloudpath=self.meta.layerpath, 
        bounds=self.meta.meta.bounds(mip), 
        resolution=self.meta.info['spatial_index'].get(
          'resolution', self.meta.meta.resolution(mip)
        ),
        chunk_size=self.meta.info['spatial_index']['chunk_size'],
      )

  @property
  def path(self):
    return self.meta.mesh_path

  def manifest_path(self, segid):
    mesh_json_file_name = str(segid) + ':0'
    return self.meta.join(self.path, mesh_json_file_name)

  def _get_manifests(self, segids, allow_missing=False):
    segids = toiter(segids)    
    paths = [ self.manifest_path(segid) for segid in segids ]
    fragments = self.cache.download(paths)

    contents = {}
    for filename, content in fragments.items():
      segid = filename_to_segid(filename)
      if content is None:
        if allow_missing:
          contents[segid] = None
          continue
        else:
          raise ValueError(f"manifest is missing for {filename}")

      content = content.decode('utf8')
      content = simdjson.loads(content)
      contents[segid] = content['fragments']

    return contents

  def _get_mesh_fragments(self, path_id_map):
    paths = [ self.meta.join(self.path, path) for path in path_id_map.keys() ] 

    compress = self.config.compress
    if compress is None:
      compress = True

    fragments = self.cache.download(paths, compress=compress)
    fragments = [ 
      (filename, content, path_id_map[os.path.basename(filename)]) 
      for filename, content in fragments.items() 
    ]
    fragments = sorted(fragments, key=lambda frag: frag[0]) # make decoding deterministic
    return fragments

  def exists(self, segids, progress=None):
    """
    Checks if the mesh exists.

    Returns: { label: path or None, ... }
    """
    manifest_paths = [ self.manifest_path(segid) for segid in segids ]
    progress = progress if progress is not None else self.config.progress

    cf = CloudFiles(
      self.meta.cloudpath, 
      progress=progress, 
      green=self.config.green, 
      secrets=self.config.secrets
    )
    exists = cf.exists(manifest_paths)

    segid_regexp = re.compile(r'(\d+):0$')

    output = {}
    for path, there in exists.items():
      (segid,) = re.search(segid_regexp, path).groups()
      output[segid] = path if there else None
  
    return output
  
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
    dne = self.exists(segids)
    dne = [ label for label, path in dne.items() if path is None ]

    if dne:
      missing = ', '.join([ str(segid) for segid in dne ])
      raise ValueError(red(
        'Segment ID(s) {} are missing corresponding mesh manifests.\nAborted.' \
        .format(missing)
      ))

    fragments = self._get_manifests(segids)
    path_id_map = {}
    for segid, paths in fragments.items():
      for path in paths:
        path_id_map[path] = segid
    fragments = self._get_mesh_fragments(path_id_map)

    # decode all the fragments
    meshdata = defaultdict(list)
    for filename, contents, segid in tqdm(fragments, disable=(not self.config.progress), desc="Decoding Mesh Buffer"):
      try:
        mesh = Mesh.from_precomputed(contents)
      except Exception:
        print(filename, 'had a problem.')
        raise
      meshdata[segid].append(mesh)

    if not fuse:
      meshdata = { 
          segid: Mesh.concatenate(*meshes, segid=segid) 
          for segid, meshes in meshdata.items() 
      }
      for mesh in meshdata.values():
        mesh.vertices = apply_transform(mesh.vertices, self.transform)
      return meshdata

    meshdata = [ (segid, mesh) for segid, mesh in meshdata.items() ]
    meshdata = sorted(meshdata, key=lambda sm: sm[0])
    meshdata = [ mesh for segid, mesh in meshdata ]
    meshdata = list(itertools.chain.from_iterable(meshdata)) # flatten
    mesh = Mesh.concatenate(*meshdata)
    mesh.vertices = apply_transform(mesh.vertices, self.transform)

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

  def put(
    self, 
    meshes:Union[Mesh,Iterable[Mesh]], 
    batch_size:int = 200,
    compress:CompressType = "gzip", 
    compression_level:int = 6,
    cdn_cache:Optional[str] = None,
    skip_delete:bool = False,
  ):
    """
    Upload a pre-existing mesh. 

    batch_size: affects performance, controls size of each upload batch
    skip_delete: Since meshes consist of an arbitrary number of files, 
      a simple upload won't necessarily replace the old mesh. Therefore,
      we read the manifest and delete the old mesh if it exists before
      uploading. You can skip this step by stetting this flag to True.
    """
    if isinstance(meshes, Mesh):
      meshes = [ meshes ]

    # using this odd structuring to ensure generators will
    # work correctly
    toupload = ( 
      (
        f"{m.segid}:0", { "fragments": [ f"{m.segid}:0:1" ] }, # manifest
        f"{m.segid}:0:1", m.to_precomputed() # fragment file
      )
      for m in meshes
    )

    # need to clear out pre-existing meshes
    if not skip_delete:
      self.delete(( m.segid for m in meshes ))

    cf = CloudFiles(self.meta.layerpath)
    for mshs in sip(toupload, batch_size):
      cf.put_jsons([ m[:2] for m in mshs ])
      cf.puts(
        [ m[2:4] for m in mshs ],
        content_type="application/octet-stream",
        compress=compress,
        compression_level=compression_level,
        cache_control=cdn_cache,
      )

  def delete(self, segids):
    """
    Removes fragment and manifest files for each segid specified.
    """
    manifests = self._get_manifests(segids, allow_missing=True)

    cf = CloudFiles(self.meta.layerpath)
    for segid, filenames in manifests.items():
      if segid is None:
        raise ValueError("Cannot delete segid None")

      if filenames is None:
        continue
      filenames += [ f"{segid}:0" ]
      cf.delete(filenames)

      if self.cache.enabled:
        self.cache.delete(filenames)

  def save(self, segids, filepath=None, file_format='ply'):
    """
    Save one or more segids into a common mesh format as a single file.

    segids: int, string, or list thereof
    filepath: string, file-like, or None (optional)
    file_format: string (optional)
    
    Supported Formats: 'obj', 'ply', 'precomputed'
    """
    segids = toiter(segids)

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
