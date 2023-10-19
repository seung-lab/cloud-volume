from collections import defaultdict
from functools import partial
import re
import struct

import numpy as np

from cloudfiles import CloudFiles

from .common import apply_transform
from .unsharded import UnshardedLegacyPrecomputedMeshSource
from ..sharding import ShardingSpecification, ShardReader
from ....scheduler import schedule_jobs
from ....mesh import Mesh
from ....lib import yellow, red, toiter, first
from .... import exceptions

import fastremap

def extract_lod_meshes(manifest, lod, lod_binary, vertex_quantization_bits, transform):
  meshdata = defaultdict(list)
  for frag in range(manifest.fragment_offsets[lod].shape[0]):
    start = int(np.sum(manifest.fragment_offsets[lod][0:frag]))
    end = start + int(manifest.fragment_offsets[lod][frag])
    frag_binary = lod_binary[start:end]
    if len(frag_binary) == 0:
      # According to @JBMS, empty fragments are used in cases where a child 
      # fragment exists, but its parent does not have a corresponding fragment, 
      # a possible byproduct of running marching cubes and mesh simplification 
      # independently for each level of detail.
      continue

    mesh = Mesh.from_draco(frag_binary)

    # "stored model" to "model" coordinates
    mesh.vertices = from_stored_model_space(
      mesh.vertices, manifest, lod, vertex_quantization_bits, frag
    )
    # "model" to physical coordinates (usually scaling by resolution)
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
      return MultiLevelPrecomputedMeshManifest.from_binary(binary, segment_id=first(segid), shard_offset=0)

    regexp = re.compile(r'(\d+)\.index$')
    manifests = []
    for res in results:
      key = res["path"]
      sid = int(re.match(regexp, key).groups()[0])
      binary = res["content"]
      if binary is None:
        manifests.append(None)
      manifest = MultiLevelPrecomputedMeshManifest.from_binary(binary, segment_id=sid, shard_offset=0)
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
        'start': int(np.sum(fragment_sizes[0:lod])),
        'end': int(np.sum(fragment_sizes[0:lod+1])),  
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

  def put(self, *args, **kwargs):
    raise NotImplementedError("put is not implemented for multi-res meshes.")

  def delete(self, segids):
    """
    Removes fragment and manifest files for each segid specified.
    """
    segids = toiter(segids)

    def filenames(segids):
      for segid in segids:
        yield self.meta.join(full_path, f"{segid}.index")
        yield self.meta.join(full_path, f"{segid}")

    full_path = self.meta.join(self.meta.cloudpath, self.path)
    progress = progress if progress is not None else self.config.progress
    CloudFiles(
      full_path, progress=progress,
      green=self.config.green, secrets=self.config.secrets
    ).delete(filenames(segids), total=len(segids) * 2)

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

    return MultiLevelPrecomputedMeshManifest.from_binary(
      binary, segment_id=segid, shard_offset=byte_start, path=shard_filepath
    )
  
  def get(self, segids, lod=0, concat=True, progress=None):
    """Fetch meshes at a given level of detail (lod).

    Parameters:
    segids: (iterable or int) segids to render

    lod: int, default 0
      Level of detail to retrieve.  0 is highest level of detail.
      Use -1 to indicate the lowest resolution available.

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

    def get_segid(segid):
      nonlocal lod
      # Read the manifest (with a tweak to sharding.py to get the offset)
      manifest = self.get_manifest(segid)
      if manifest == None:
        raise exceptions.MeshDecodeError(red(
          'Manifest not found for segment {}.'.format(segid)
        ))

      if lod < -1 or lod >= manifest.num_lods:
        raise exceptions.MeshDecodeError(red(
          f'LOD value ({lod}) out of range (-1 - {manifest.num_lods - 1}) for segment {segid}.'
        ))
      
      if lod == -1:
        lod = manifest.num_lods - 1

      # Read the data for all LODs
      fragment_sizes = [ 
        np.sum(lod_fragment_sizes) for lod_fragment_sizes in manifest.fragment_offsets 
      ]
      total_fragment_size = np.sum(fragment_sizes)
      full_path = self.reader.meta.join(self.reader.meta.cloudpath)

      manifest_byte_start = (manifest.shard_offset - total_fragment_size) + np.sum(fragment_sizes[0:lod])
      lod_binary = CloudFiles(full_path, progress=progress, secrets=self.config.secrets).get({
        'path': manifest.path,
        'start': int(manifest_byte_start),
        'end': int(manifest_byte_start + fragment_sizes[lod]),
      })

      return extract_lod_meshes(
        manifest, lod, lod_binary, 
        self.vertex_quantization_bits, self.transform
      )

    # decode all the fragments
    meshdata = defaultdict(list)
    def get_meshes_and_update(segid):
      nonlocal meshdata
      meshes = get_segid(segid)
      meshdata.update(meshes)

    schedule_jobs(
      fns=[ partial(get_meshes_and_update, segid) for segid in segids ],
      progress=progress,
      total=len(segids),
      green=self.config.green,
    )

    if concat:
      for segid in meshdata:
        meshdata[segid] = Mesh.concatenate(*meshdata[segid])

    return meshdata

  def put(self, *args, **kwargs):
    raise NotImplementedError("put is not implemented for multi-res meshes.")

  def delete(self, *args, **kwargs):
    raise NotImplementedError("delete is not implemented for individual sharded multi-res meshes.")

class MultiLevelPrecomputedMeshManifest:
  # Parse the multi-resolution mesh manifest file format:
  # https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/meshes.md
  # https://github.com/google/neuroglancer/blob/master/src/neuroglancer/mesh/multiscale.ts

  def __init__(
    self, segment_id, chunk_shape, grid_origin, 
    num_lods, lod_scales, vertex_offsets, num_fragments_per_lod, 
    fragment_positions, fragment_offsets, shard_offset=0, 
    path=None
  ):
    # core specification
    self.chunk_shape = chunk_shape
    self.grid_origin = grid_origin
    self.num_lods = int(num_lods)
    self.lod_scales = lod_scales
    self.vertex_offsets = vertex_offsets
    self.num_fragments_per_lod = num_fragments_per_lod
    self.fragment_positions = fragment_positions
    self.fragment_offsets = fragment_offsets

    # custom metadata
    self.segment_id = int(segment_id)
    self.shard_offset = shard_offset
    self.path = path

    # normalize attributes
    self.fragment_positions = [ 
      np.array(fpos) for fpos in self.fragment_positions
    ]

  @classmethod
  def from_binary(cls, binary, segment_id, shard_offset=0, path=None):
    # num_loads is the 7th word
    num_lods = int(np.frombuffer(binary[6*4:7*4], dtype=np.uint32)[0])

    header_dt = cls._header_dtype(cls, num_lods)
    header = np.frombuffer(binary[0:header_dt.itemsize], dtype=header_dt)
    offset = header_dt.itemsize
    num_fragments_per_lod = header["num_fragments_per_lod"][0]

    fragment_positions = []
    fragment_offsets = []
    for lod in range(num_lods):
      # Read fragment positions
      pos_size =  3 * 4 * num_fragments_per_lod[lod]
      fragment_positions.append(
        np.frombuffer(binary[offset:offset + pos_size], dtype=np.uint32).reshape((num_fragments_per_lod[lod],3), order="F")
      )
      offset += pos_size

      # Read fragment sizes
      off_size = 4 * num_fragments_per_lod[lod]
      fragment_offsets.append(
        np.frombuffer(binary[offset:offset + off_size], dtype=np.uint32)
      )
      offset += off_size

    # Make sure we read the entire manifest
    if offset != len(binary):
      raise exceptions.MeshDecodeError(red(
        'Error decoding mesh manifest for segment {}'.format(segment_id)
      ))

    return MultiLevelPrecomputedMeshManifest(
      segment_id, 
      chunk_shape=header['chunk_shape'][0],
      grid_origin=header['grid_origin'][0],
      num_lods=header['num_lods'][0],
      lod_scales=header['lod_scales'][0], 
      vertex_offsets=header['vertex_offsets'][0],
      num_fragments_per_lod=header['num_fragments_per_lod'][0],
      fragment_positions=fragment_positions,
      fragment_offsets=fragment_offsets,
      shard_offset=shard_offset,
      path=path
    )

  def to_binary(self):
    """Render the manifest in its serialized binary representation."""
    chunk_shape = np.array(self.chunk_shape, dtype=np.float32).reshape((3,))
    grid_origin = np.array(self.grid_origin, dtype=np.float32).reshape((3,))
    vertex_offsets = np.array(self.vertex_offsets, dtype=np.float32).reshape(
      (self.num_lods, 3), order="C"
    )
    num_fragments_per_lod = np.array(
      self.num_fragments_per_lod, dtype=np.uint32
    ).reshape((self.num_lods,), order="C")

    # frag positions and offsets must be provided in morton order
    fragment_positions = [ 
      np.array(fpos, dtype="<I").tobytes(order='F') 
      for fpos in self.fragment_positions 
    ]
    fragment_offsets = np.array(self.fragment_offsets, dtype=np.uint32)
    lod_scales = np.array(self.lod_scales, dtype=np.float32)

    manifest = [
      chunk_shape.astype('<f').tobytes(),
      grid_origin.astype('<f').tobytes(),
      struct.pack('<I', self.num_lods),
      lod_scales.astype('<f').tobytes(),
      vertex_offsets.astype('<f').tobytes(order='C'),
      num_fragments_per_lod.astype('<I').tobytes(),
    ]

    offset = 0
    for lod in range(self.num_lods):
      manifest.append(
        fragment_positions[lod]
      )
      manifest.append(
        fragment_offsets[offset:offset+num_fragments_per_lod[lod]]
          .astype('<I').tobytes(order='C')
      )
      offset += num_fragments_per_lod[lod]

    return b''.join(manifest)

  def header_dtype(self):
    return self._header_dtype(self.num_lods)

  def _header_dtype(cls, num_lods):
    return np.dtype([
      ('chunk_shape', np.float32, (3,)),
      ('grid_origin', np.float32, (3,)),
      ('num_lods', np.uint32),
      ('lod_scales', np.float32, (num_lods,)),
      ('vertex_offsets', np.float32, (num_lods,3)),
      ('num_fragments_per_lod', np.uint32, (num_lods,))
    ])

  def __len__(self):
    fixed_header_size = self.header_dtype().itemsize
    lod_frags = 0
    for lod in range(self.num_lods):
      lod_frags += self.num_fragments_per_lod[lod]

    variable_header_size = (3*4 + 4) * lod_frags # frag pos + frag offsets
    return fixed_header_size + variable_header_size

def from_stored_model_space(
  vertices:np.ndarray, 
  manifest:MultiLevelPrecomputedMeshManifest, 
  lod:int, 
  vertex_quantization_bits:int, 
  frag:int
) -> np.ndarray:
  """
  Neuroglancer Specification:
  https://github.com/google/neuroglancer/blob/8432f531c4d8eb421556ec36926a29d9064c2d3c/src/neuroglancer/datasource/precomputed/meshes.md#multi-resolution-mesh-fragment-data-file-format

  The mesh fragment data files consist of the concatenation of the 
  encoded mesh data for all octree nodes specified in the manifest file,
  in the same order the nodes are specified in the index file, starting
  with lod 0. Each mesh fragment is a Draco-encoded triangular mesh with
  a 3-component integer vertex position attribute. Each position component j
  must be a value x in the range [0, 2**vertex_quantization_bits), which
  corresponds to a "stored model" coordinate of:

  grid_origin[j] +
  vertex_offsets[lod,j] +
  chunk_shape[j] * (2**lod) * (fragmentPosition[j] +
                               x / ((2**vertex_quantization_bits)-1))
  """
  return np.array(
    manifest.grid_origin + 
    manifest.vertex_offsets[lod] + (
      manifest.chunk_shape * (2 ** lod) * (
        manifest.fragment_positions[lod][frag,:] + 
        (vertices / (2.0 ** vertex_quantization_bits - 1))
      )
    )
  )

def to_stored_model_space(  
  vertices:np.ndarray, 
  manifest:MultiLevelPrecomputedMeshManifest, 
  lod:int, 
  vertex_quantization_bits:int, 
  frag:int
) -> np.ndarray:
  """Inverse of from_stored_model_space (see explaination there)."""
  vertices = vertices.astype(np.float32, copy=False)
  quant_factor = ((2 ** vertex_quantization_bits) - 1) 

  stored_model = vertices - manifest.grid_origin - manifest.vertex_offsets[lod]
  stored_model /= manifest.chunk_shape * (2 ** lod)
  stored_model -= manifest.fragment_positions[lod][frag,:]
  stored_model *= quant_factor
  stored_model = np.round(stored_model, out=stored_model)
  stored_model = np.clip(
    stored_model, 0, quant_factor, 
    out=stored_model
  )

  dtype = fastremap.fit_dtype(np.uint64, value=quant_factor)
  return stored_model.astype(dtype)




