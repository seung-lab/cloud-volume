from collections import defaultdict, namedtuple
import itertools
import orjson
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

ACCEPT_HEADER = {
  "Accept": "application/x.cave;manifest_version=2",
}

GrapheneMeshRequestParams = namedtuple("GrapheneMeshRequestParams", ["segid", "key", "byte_start", "length" ])

class GrapheneMeshManifest:
  def __init__(self, content:dict):
    """Parses the manifest from the server into a normalized form"""
    if content.get("manifest_version", 1) >= 2:
      self.manifest = self.parse_v2_manifest(content)
    else:
      self.manifest = self.parse_v1_manifest(content)

  def keys(self) -> list[str]:
    all_segids = []
    for cloudpath, params in self.manifest.items():
      for req in params:
        all_segids.append(req.key)
    return all_segids

  def segids(self) -> list[int]:
    all_segids = []
    for cloudpath, params in self.manifest.items():
      for req in params:
        all_segids.append(req.segid)
    return all_segids

  def to_manifest_v2(self) -> dict:
    """Dumps the contents of the manifest into the v2 format."""
    manifest_v2 = {
      "manifest_version": 2,
      "fragments": defaultdict(list),
    }

    for cloudpath, params in self.manifest.items():
      sub = manifest_v2["fragments"][cloudpath]
      for param in params:
        if param.byte_start is None:
          sub.append(param.key)
        else:
          sub.append(f"~{param.segid}:{param.key}:{param.byte_start}:{param.size}")

    return manifest_v2

  def cloudfiles_requests(self) -> dict[str, list[dict]]:
    """Convert the manifests into the CloudFiles request format."""
    cf_requests = defaultdict(list)
    for cloudpath, params in self.manifest.items():
      req = cf_requests[cloudpath]
      for param in params:
        if param.byte_start is None:
          req.append({
            "path": param.key,
            "tag": param.segid,
          })
        else:
          req.append({
            "path": param.key,
            "start": param.byte_start,
            "end": param.byte_start + param.size,
            "tag": param.segid,
          })
    return cf_requests

  def parse_v2_manifest(self, manifest:dict) -> dict[int,list[GrapheneMeshRequestParams]]:
    """
    {
       "manifest_version": 2,
       "fragments":{
         "gs://pcg_ws/initial_meshes": [
            "~170521060133831369:2/393478156-0.shard:420288:535",
            "~170520991414354512:2/393477132-0.shard:156168:233"
         ],
         "gs://pcg_ws/dynamic_meshes": [
            "~170521060133831369:2/393478156-0.shard:420288:535",
            "~170520991414354512:2/393477132-0.shard:156168:233",
            "182189093902354880:0:34560-34816_17408-17664_2048-2560",
            "182189093902355396:0:34560-34816_17408-17664_2048-2560"
         ]
       }
    }
    """
    cf_requests = defaultdict(list)

    shard_regexp = re.compile(r'~(\d+):(\d+)/([\d\-]+\.shard):(\d+):(\d+)')

    fragments = manifest['fragments']

    for cloudpath in fragments.keys():
      for filename in fragments[cloudpath]:
        if not filename:
          continue

        # eg. ~2/344239114-0.shard:224659:442 
        # tilde means initial (i.e. sharded), missing tilde means dynamic (i.e. unsharded)
        sharded = filename[0] == '~'

        if sharded:
          (segid, layer_id, parsed_filename, byte_start, size) = re.search(
            shard_regexp, filename
          ).groups()

          cf_requests[cloudpath].append(
            GrapheneMeshRequestParams(
              int(segid),
              self.meta.join(str(layer_id), parsed_filename),
              byte_start,
              size
            )
          )

        else:
          segid = int(filename.split(":")[0])
          cf_requests[cloudpath].append(
            GrapheneMeshRequestParams(segid, filename, None, None)
          )

    return cf_requests

  def parse_v1_manifest(self, manifest:dict) -> dict[int,list[GrapheneMeshRequestParams]]:
    """
    {
      "fragments": [
        "~2/344239114-0.shard:224659:442",
        "396809348417946934:0:32768-34816_14336-16384_0-4096",
        "181621745902421420:0:34048-34304_16384-16640_2048-2560",
        "182189093902354880:0:34560-34816_17408-17664_2048-2560",
        "182189093902355396:0:34560-34816_17408-17664_2048-2560",
        "325684415118191036:0:33792-34816_17408-18432_2048-4096"
      ],
      "seg_ids": [
        2383274832232,
        ...
      ]
    }
    """
    cf_requests = {}

    initial_cloudpath = self.meta.join(self.meta.meta.cloudpath, self.meta.mesh_path, self.meta.sharded_mesh_dir)
    dynamic_cloudpath = self.meta.join(self.meta.meta.cloudpath, self.dynamic_path())

    initial_regexp = re.compile(r'~(\d+)/([\d\-]+\.shard):(\d+):(\d+)')

    filenames, segids = manifest['fragments'], manifest['seg_ids']

    for filename, segid in zip(filenames, segids):
      if not filename:
        continue

      # eg. ~2/344239114-0.shard:224659:442 
      # tilde means initial (i.e. sharded), missing tilde means dynamic (i.e. unsharded)
      initial = filename[0] == '~'

      if initial:
        (layer_id, parsed_filename, byte_start, size) = re.search(
          initial_regexp, filename
        ).groups()

        cf_requests[initial_cloudpath].append(
          GrapheneMeshRequestParams(
            segid,
            self.meta.join(str(layer_id), parsed_filename),
            byte_start,
            size
          )
        )
      else:
        segid = int(filename.split(":")[0])
        cf_requests[dynamic_cloudpath].append(
          GrapheneMeshRequestParams(segid, filename, None, None)
        )
        
    return cf_requests

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
    return manifest.segids()

  def get_fragment_filenames(self, segid, lod=0, level=2, bbox=None, bypass=False):
    if bypass:
      return [ self.compute_filename(segid) ]

    manifest = self.fetch_manifest(segid, lod, level, bbox, verify=True)
    return manifest.keys()

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
        return GrapheneMeshManifest(manifest)

    manifest = self.fetch_manifest_remote(segid, lod, level, bbox, return_segids, verify)

    if self.cache.enabled and cacheable:
      self.cache.put_json(cache_path, manifest.to_manifest_v2())

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

    # In July 2026, to avoid egress fees Forrest Collman, Will Silversmith,
    # and Akhilesh Halageri decided to make it possible
    # to store supervoxels, initial meshes, and dynamic meshes in separate
    # locations. The Accept header (not previously present) signals 
    # to the server that the new format is supported by this client. 
    # The new format allows splitting these locations while the old format
    # has a built-in assumption that all three are co-located.

    headers = dict(self.meta.meta.auth_header)
    headers.update(ACCEPT_HEADER)

    url = "%s/%s:%s" % (self.meta.meta.manifest_endpoint, segid, lod)
    if level is not None:
      res = requests.get(
        url,
        data=jsonify({ "start_layer": level }),
        params=query_d,
        headers=headers
      )
    else:
      res = requests.get(url, params=query_d, headers=headers)

    res.raise_for_status()

    content = orjson.loads(res.content.decode('utf8'))
    return GrapheneMeshManifest(content)

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

