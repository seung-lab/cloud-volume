from collections import defaultdict
import re

import numpy as np

from cloudfiles import CloudFiles

from .common import apply_transform
from .unsharded import UnshardedLegacyPrecomputedMeshSource
from ..sharding import ShardingSpecification, ShardReader
from ....mesh import Mesh
from ....lib import yellow, red, toiter, first
from .... import exceptions

def extract_lod_meshes(manifest, lod, lod_binary, vertex_quantization_bits, transform):
  meshdata = defaultdict(list)
  for frag in range(manifest.fragment_offsets[lod].shape[0]):
    frag_binary = lod_binary[
      int(np.sum(manifest.fragment_offsets[lod][0:frag])) :
      int(np.sum(manifest.fragment_offsets[lod][0:frag+1]))
    ]
    if len(frag_binary) == 0:
      # According to @JBMS, empty fragments are used in cases where a child 
      # fragment exists, but its parent does not have a corresponding fragment, 
      # a possible byproduct of running marching cubes and mesh simplification 
      # independently for each level of detail.
      continue

    mesh = Mesh.from_draco(frag_binary)

    # Convert from "stored model" space to "model" space
    mesh.vertices = manifest.grid_origin + manifest.vertex_offsets[lod] + \
            manifest.chunk_shape * (2 ** lod) * \
            (manifest.fragment_positions[lod][:,frag] + \
            (mesh.vertices / (2.0 ** vertex_quantization_bits - 1)))
    
    mesh.vertices = apply_transform(mesh.vertices, transform)
    meshdata[manifest.segment_id].append(mesh)
  return meshdata

class UnshardedMultiLevelPrecomputedMeshSource(UnshardedLegacyPrecomputedMeshSource):
  def __init__(self, meta, cache, config, readonly=False):
    super().__init__(meta, cache, config, readonly)

    self.vertex_quantization_bits = self.meta.info['vertex_quantization_bits']
    self.lod_scale_multiplier = self.meta.info['lod_scale_multiplier']
    self.transform = np.array(self.meta.info['transform'] + [0,0,0,1]).reshape(4,4)
  
  @property
  def path(self):
    return self.meta.mesh_path

  def get_manifest(self, segid, progress=None):
    """Retrieve the manifest for one or more segments."""
    segid, multiple_return = toiter(segid, is_iter=True)
    progress = progress if progress is not None else self.config.progress

    cloudpath = self.meta.join(self.meta.cloudpath, self.path)
    cf = CloudFiles(cloudpath, progress=progress)
    results = cf.get((f"{sid}.index" for sid in segid ), total=len(segid))

    if not multiple_return:
      if not results:
        return None
      binary = results[0]["content"]
      if binary is None:
        return None
      return MultiLevelPrecomputedMeshManifest(binary, segment_id=first(segid), offset=0)

    regexp = re.compile(r'(\d+)\.index$')
    manifests = []
    for res in results:
      key = res["path"]
      sid = int(re.match(regexp, key).groups()[0])
      binary = res["content"]
      if binary is None:
        manifests.append(None)
      manifest = MultiLevelPrecomputedMeshManifest(binary, segment_id=sid, offset=0)
      manifests.append(manifest)

    return manifests

  def exists(self, segids, progress=None):
    """
    Checks if the mesh exists

    Returns: { MultiLevelPrecomputedMeshManifest or None, ... }
    """
    cf = CloudFiles(self.path)
    return cf.exists(( f"{segid}.index" for segid in segids ))

  def get(self, segids, lod=0, concat=True, progress=None):
    """Fetch meshes at a given level of detail (lod).

    Parameters:
    segids: (iterable or int) segids to render

    lod: int, default 0
      Level of detail to retrieve.  0 is highest level of detail.

    Optional:
      concat: bool, concatenate fragments (per segment per lod)

    Returns:
    { segid: { Mesh } }
    ... or if concatenate=False: { segid: { Mesh, ... } }

    Reference:
      https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/meshes.md
    """
    if lod < 0:
      raise exceptions.ValueError(red(f'lod ({lod}) must be >= 0.'))

    progress = progress if progress is not None else self.config.progress
    segids = toiter(segids)

    # decode all the fragments
    manifests = self.get_manifest(segids)
    for manifest in manifests:
      if manifest is None:
        raise exceptions.MeshDecodeError(red(
          f'Manifest not found for segment {manifest.segment_id}.'
        ))
      if lod >= manifest.num_lods:
        raise exceptions.MeshDecodeError(red(
          f'LOD value ({lod}) out of range (0 - {manifest.num_lods - 1}) for segment {manifest.segment_id}.'
        ))

    full_path = self.meta.join(self.meta.cloudpath, self.path)

    meshdata = defaultdict(list)
    for manifest in manifests:
      # Read the manifest (with a tweak to sharding.py to get the offset)
      fragment_sizes = [ 
        np.sum(lod_fragment_sizes) for lod_fragment_sizes in manifest.fragment_offsets 
      ]
      
      lod_binary = CloudFiles(
        full_path, progress=progress, 
        green=self.config.green, secrets=self.config.secrets
      ).get({
        'path': str(manifest.segment_id),
        'start': np.sum(fragment_sizes[0:lod]),
        'end': np.sum(fragment_sizes[0:lod+1]),  
      })

      meshes = extract_lod_meshes(
        manifest, lod, lod_binary, 
        self.vertex_quantization_bits, self.transform
      )
      meshdata.update(meshes)

    if concat:
      for segid in meshdata:
        meshdata[segid] = Mesh.concatenate(*meshdata[segid])

    return meshdata

class ShardedMultiLevelPrecomputedMeshSource(UnshardedLegacyPrecomputedMeshSource):
  def __init__(self, meta, cache, config, readonly=False):
    super(ShardedMultiLevelPrecomputedMeshSource, self).__init__(meta, cache, config, readonly)

    spec = ShardingSpecification.from_dict(self.meta.info['sharding'])
    self.reader = ShardReader(meta, cache, spec)

    self.vertex_quantization_bits = self.meta.info['vertex_quantization_bits']
    self.lod_scale_multiplier = self.meta.info['lod_scale_multiplier']
    self.transform = np.array(self.meta.info['transform'] + [0,0,0,1]).reshape(4,4)


  @property
  def path(self):
    return self.meta.mesh_path

  def exists(self, segids, progress=None):
    """
    Checks if the mesh exists

    Returns: { MultiLevelPrecomputedMeshManifest or None, ... }
    """
    return [ self.get_manifest(segid) for segid in segids ]

  def get_manifest(self, segid, progress=None):
    """Retrieve the manifest for a single segment.

    Returns:
      { MultiLevelPrecomputedMeshManifest or None }
    """
    manifest_info = self.reader.exists(segid, self.path, return_byte_range=True)
    if manifest_info is None:
      # Manifest not found
      return None
    shard_filepath, byte_start, num_bytes = tuple(manifest_info)
    binary = self.reader.get_data(segid, self.path)
    if binary is None:
      return None
    return MultiLevelPrecomputedMeshManifest(binary, segment_id=segid, offset=byte_start, path=shard_filepath)
  
  def get(self, segids, lod=0, concat=True, progress=None):
    """Fetch meshes at a given level of detail (lod).

    Parameters:
    segids: (iterable or int) segids to render

    lod: int, default 0
      Level of detail to retrieve.  0 is highest level of detail.

    Optional:
      concat: bool, concatenate fragments (per segment per lod)

    Returns:
    { segid: { Mesh } }
    ... or if concatenate=False: { segid: { Mesh, ... } }

    Reference:
      https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/meshes.md
    """
    progress = progress if progress is not None else self.config.progress
    segids = toiter(segids)

    # decode all the fragments
    meshdata = defaultdict(list)
    for segid in segids:
      # Read the manifest (with a tweak to sharding.py to get the offset)
      manifest = self.get_manifest(segid)
      if manifest == None:
        raise exceptions.MeshDecodeError(red(
          'Manifest not found for segment {}.'.format(segid)
        ))

      if lod < 0 or lod >= manifest.num_lods:
        raise exceptions.MeshDecodeError(red(
          'LOD value ({}) out of range (0 - {}) for segment {}.'.format(lod, manifest.num_lods - 1, segid)
        ))

      # Read the data for all LODs
      fragment_sizes = [ 
        np.sum(lod_fragment_sizes) for lod_fragment_sizes in manifest.fragment_offsets 
      ]
      total_fragment_size = np.sum(fragment_sizes)
      full_path = self.reader.meta.join(self.reader.meta.cloudpath)
      lod_binary = CloudFiles(full_path, progress=progress, secrets=self.config.secrets).get({
        'path': manifest.path,
        'start': (manifest.offset - total_fragment_size) + np.sum(fragment_sizes[0:lod]),
        'end': (manifest.offset - total_fragment_size) + np.sum(fragment_sizes[0:lod+1]),  
      })

      meshes = extract_lod_meshes(
        manifest, lod, lod_binary, 
        self.vertex_quantization_bits, self.transform
      )
      meshdata.update(meshes)

    if concat:
      for segid in meshdata:
        meshdata[segid] = Mesh.concatenate(*meshdata[segid])

    return meshdata

class MultiLevelPrecomputedMeshManifest:
  # Parse the multi-resolution mesh manifest file format:
  # https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/meshes.md
  # https://github.com/google/neuroglancer/blob/master/src/neuroglancer/mesh/multiscale.ts

  def __init__(self, binary, segment_id, offset, path=None):
    self._segment = segment_id
    self._binary = binary
    self._offset = offset
    self._path = path

    # num_loads is the 7th word
    num_lods = int(np.frombuffer(self._binary[6*4:7*4], dtype=np.uint32)[0])

    header_dt = np.dtype([
      ('chunk_shape', np.float32, (3,)),
      ('grid_origin', np.float32, (3,)),
      ('num_lods', np.uint32),
      ('lod_scales', np.float32, (num_lods,)),
      ('vertex_offsets', np.float32, (num_lods,3)),
      ('num_fragments_per_lod', np.uint32, (num_lods,))
    ])
    self._header = np.frombuffer(self._binary[0:header_dt.itemsize], dtype=header_dt)
    offset = header_dt.itemsize

    self._fragment_positions = []
    self._fragment_offsets = []
    for lod in range(num_lods):
      # Read fragment positions
      pos_size =  3 * 4 * self.num_fragments_per_lod[lod]
      self._fragment_positions.append(
        np.frombuffer(self._binary[offset:offset + pos_size], dtype=np.uint32).reshape((3,self.num_fragments_per_lod[lod]))
      )
      offset += pos_size

      # Read fragment sizes
      off_size = 4 * self.num_fragments_per_lod[lod]
      self._fragment_offsets.append(
        np.frombuffer(self._binary[offset:offset + off_size], dtype=np.uint32)
      )
      offset += off_size

    # Make sure we read the entire manifest
    if offset != len(binary):
      raise exceptions.MeshDecodeError(red(
        'Error decoding mesh manifest for segment {}'.format(segment_id)
      ))

  def data_size(self):
    fragment_sizes = [ 
      np.sum(lod_fragment_sizes) for lod_fragment_sizes in self.fragment_offsets 
    ]
    return np.sum(fragment_sizes)

  @property
  def segment_id(self):
    return self._segment

  @property
  def chunk_shape(self):
    return self._header['chunk_shape'][0]

  @property
  def grid_origin(self):
    return self._header['grid_origin'][0]

  @property
  def num_lods(self):
    return self._header['num_lods'][0]

  @property
  def lod_scales(self):
    return self._header['lod_scales'][0]

  @property
  def vertex_offsets(self):
    return self._header['vertex_offsets'][0]

  @property
  def num_fragments_per_lod(self):
    return self._header['num_fragments_per_lod'][0]

  @property
  def fragment_positions(self):
    return self._fragment_positions

  @property
  def fragment_offsets(self):
    return self._fragment_offsets

  @property
  def length(self):
    return len(self._binary)

  @property
  def offset(self):
    """Manifest offset within the shard file. Used as a base when calculating fragment offsets."""
    return self._offset

  @property
  def path(self):
    return self._path
