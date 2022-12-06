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

from ....lib import red, toiter, Bbox, Vec, jsonify
from ....mesh import Mesh
from .... import paths
from ....scheduler import schedule_jobs

from ...precomputed.mesh import UnshardedLegacyPrecomputedMeshSource, PrecomputedMeshMetadata


class GrapheneUnshardedMeshSource(UnshardedLegacyPrecomputedMeshSource):

  def compute_filename(self, label):
    layer_id = self.meta.meta.decode_layer_id(label)
    chunk_block_shape = Vec(*self.meta.meta.mesh_chunk_size, dtype=np.int64)
    chunk_block_shape *= np.int64(self.meta.meta.fan_out ** max(0, layer_id - 2))
    start = self.meta.meta.decode_chunk_position(label)
    start *= chunk_block_shape
    bbx = Bbox(start, start + chunk_block_shape)
    return "{}:0:{}".format(label, bbx.to_filename())

  def exists(self, labels, progress=None):
    """
    Checks for dynamic mesh existence.
  
    Returns: { label: boolean, ... }
    """
    labels = toiter(labels)
    filenames = [
      self.compute_filename(label) for label in labels
    ]

    cloudpath = self.meta.join(self.meta.cloudpath, self.meta.mesh_path)
    return CloudFiles(cloudpath, secrets=self.config.secrets).exists(filenames)

  def get_fragment_labels(self, segid, lod=0, level=2, bbox=None, bypass=False):
    if bypass:
      return [ segid ]

    manifest = self.fetch_manifest(segid, lod, level, bbox, return_segids=True, verify=False)
    return manifest["seg_ids"]

  def get_fragment_filenames(self, segid, lod=0, level=2, bbox=None, bypass=False):
    if bypass:
      return [ self.compute_filename(segid) ]

    manifest = self.fetch_manifest(segid, lod, level, bbox, verify=True)
    return manifest["fragments"]

  def fetch_manifest(self, segid, lod=0, level=2, bbox=None, return_segids=False, verify=True):
    """
    verify: the server calls exists and returns byte ranges. this is intended 
      to take advantage of the lower latency on the server side, but can be a
      bottleneck.
    """
    # TODO: add lod to endpoint
    cacheable = (bbox is None and verify)
    cache_path = self.meta.join(self.path, f"{segid}:{lod}")

    if self.cache.enabled and cacheable:
      manifest = self.cache.get_json(cache_path)
      if manifest is not None:
        if "seg_ids" in manifest and not return_segids:
          del manifest["seg_ids"]
          return manifest
        elif "seg_ids" not in manifest and return_segids:
          pass
        else:
          return manifest

    manifest = self.fetch_manifest_remote(segid, lod, level, bbox, return_segids, verify)

    if self.cache.enabled and cacheable:
      self.cache.put_json(cache_path, manifest)

    return manifest

  def fetch_manifest_remote(self, segid, lod=0, level=2, bbox=None, return_segids=False, verify=True):
    query_d = {
      'verify': bool(verify),
    }
    if return_segids:
      query_d['return_seg_ids'] = 1

    if bbox is not None:
      bbox = Bbox.create(bbox)
      query_d['bounds'] = bbox.to_filename()

    level = min(level, self.meta.meta.max_meshed_layer)

    url = "%s/%s:%s" % (self.meta.meta.manifest_endpoint, segid, lod)
    if level is not None:
      res = requests.get(
        url,
        data=jsonify({ "start_layer": level }),
        params=query_d,
        headers=self.meta.meta.auth_header
      )
    else:
      res = requests.get(url, params=query_d, headers=self.meta.meta.auth_header)

    res.raise_for_status()

    return json.loads(res.content.decode('utf8'))

  def download_segid(self, seg_id, bounding_box, bypass, use_byte_offsets=True):
    """
    Download a mesh for a single segment ID.

    seg_id: Download the mesh for this segid.
    bounding_box: Limit the query for child meshes to this bounding box.
    bypass: Don't fetch the manifest, precompute the filename instead. Use this
      only when you know the actual mesh labels in advance.
    use_byte_offsets: Applicable only for the sharded format. Reuse the byte_offsets
      into the sharded format that the server precalculated to accelerate download.
      A time when you might want to switch this off is when you're working on a new
      meshing job with different sharding parameters but are keeping the existing 
      meshes for visualization while it runs.
    allow_missing: If set to True, return None if segid missing. If set to False, throw
      an error.
    """
    import DracoPy
    if bounding_box is not None:
      level = 2
    else:
      level = self.meta.meta.decode_layer_id(seg_id)
    fragment_filenames = self.get_fragment_filenames(
      seg_id, level=level, bbox=bounding_box, bypass=bypass
    )
    fragments = self._get_mesh_fragments({ fname: seg_id for fname in fragment_filenames })
    fragments = sorted(fragments, key=lambda frag: frag[0])  # make decoding deterministic

    fragiter = tqdm(fragments, disable=(not self.config.progress), desc="Decoding Mesh Buffer")
    is_draco = False
    for i, (filename, frag, _) in enumerate(fragiter):
      mesh = None
      
      if frag is not None:
        try:
          # Easier to ask forgiveness than permission
          mesh = Mesh.from_draco(frag)
          is_draco = True
        except DracoPy.FileTypeException:
          mesh = Mesh.from_precomputed(frag)
          
      fragments[i] = mesh
    
    fragments = [ f for f in fragments if f is not None ] 
    if len(fragments) == 0:
      raise IndexError('No mesh fragments found for segment {}'.format(seg_id))

    mesh = Mesh.concatenate(*fragments)
    mesh.segid = seg_id
    return mesh, is_draco

  def get(
      self, segids, 
      remove_duplicate_vertices=False, 
      fuse=False, bounding_box=None,
      bypass=False, use_byte_offsets=True,
      deduplicate_chunk_boundaries=True,
      allow_missing=False,
    ):
    """
    Merge fragments derived from these segids into a single vertex and face list.

    Why merge multiple segids into one mesh? For example, if you have a set of
    segids that belong to the same neuron.

    segid: (iterable or int) segids to render into a single mesh

    Optional:
      remove_duplicate_vertices: bool, fuse exactly matching vertices within a chunk
      fuse: bool, merge all downloaded meshes into a single mesh
      bounding_box: Bbox, bounding box to restrict mesh download to
      bypass: bypass requesting the manifest and attempt to get the 
        segids from storage directly by testing the dynamic and then the initial mesh. 
        This is an exceptional usage of this tool and should be applied only with 
        an understanding of what that entails.
      use_byte_offsets: For sharded volumes, we can use the output of 
        exists(..., return_byte_offsets) that the server already did in order
        to skip having to query the sharded format again.
      deduplicate_chunk_boundaries: Our meshing is done in chunks and creates duplicate vertices
        at the boundaries of chunks. This parameter will automatically deduplicate these if set
        to True. Superceded by remove_duplicate_vertices.
      allow_missing: If set to True, missing segids will be ignored. If set to False, an error
        is thrown.
    
    Returns: Mesh object if fused, else { segid: Mesh, ... }
    """
    segids = list(set([ int(segid) for segid in toiter(segids) ]))
    meta = self.meta.meta

    exceptions = (IndexError,) if allow_missing else ()

    meshes = []
    for seg_id in tqdm(segids, disable=(not self.config.progress), desc="Downloading Meshes"):
      level = meta.decode_layer_id(seg_id)
      try:
        mesh, is_draco = self.download_segid(
          seg_id, bounding_box, bypass, use_byte_offsets
        )
      except exceptions:
        continue

      resolution = meta.resolution(self.config.mip)
      if meta.chunks_start_at_voxel_offset:
        offset = meta.voxel_offset(self.config.mip)
      else:
        offset = Vec(0,0,0)

      if remove_duplicate_vertices:
        mesh = mesh.consolidate()
      elif is_draco:
        if not deduplicate_chunk_boundaries:
          pass
        elif level == 2:
          # Deduplicate at quantized lvl2 chunk borders
          draco_grid_size = meta.get_draco_grid_size(level)
          mesh = mesh.deduplicate_chunk_boundaries(
            meta.mesh_chunk_size * resolution,
            offset=offset * resolution,
            is_draco=True,
            draco_grid_size=draco_grid_size,
          )
        else:
          # TODO: cyclic draco quantization to properly
          # stitch and deduplicate draco meshes at variable
          # levels (see github issue #299)
          print('Warning: deduplication not currently supported for this layer\'s variable layered draco meshes')
      elif deduplicate_chunk_boundaries:
        mesh = mesh.deduplicate_chunk_boundaries(
            meta.mesh_chunk_size * resolution,
            offset=offset * resolution,
            is_draco=False,
          )
      
      meshes.append(mesh)

    if not fuse:
      return { m.segid: m for m in meshes }

    return Mesh.concatenate(*meshes).consolidate()

