from collections import defaultdict

import numpy as np

from ..sharding import ShardingSpecification, ShardReader
from ....storage import SimpleStorage
from ....mesh import Mesh
from ....lib import yellow, red, toiter
from .... import exceptions

class ShardedMultiLevelPrecomputedMeshSource:
    def __init__(self, meta, cache, config, readonly=False):
        self.meta = meta
        self.cache = cache
        self.config = config
        self.readonly = bool(readonly)

        spec = ShardingSpecification.from_dict(self.meta.info['sharding'])
        self.reader = ShardReader(meta, cache, spec)

        self.vertex_quantization_bits = self.meta.info['vertex_quantization_bits']
        self.lod_scale_multiplier = self.meta.info['lod_scale_multiplier']
        self.transform = np.array(self.meta.info['transform'] + [0,0,0,1]).reshape(4,4)
    
        if np.any(self.transform * np.array([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]])):
            raise exceptions.MeshDecodeError(red("Non-scale homogeneous transforms are not implemented"))


    @property
    def path(self):
        return self.meta.mesh_path


    def exists(self, segids, progress=None):
        """
        Checks if the mesh exists

        Returns: { MultiLevelPrecomputedMeshManifest or None }
        """
        return [ self.get_manifest(segid) for segid in segids ]


    def get_manifest(self, segid):
        """Retrieve the manifest for a single segment.

        Returns:
            { MultiLevelPrecomputedMeshManifest or None }
        """
        result = self.reader.get_data(segid, self.meta.mesh_path, return_offset=True)
        if result == None:
            return None
        binary, shard_file_offset = result
        return MultiLevelPrecomputedMeshManifest(binary, segment_id=segid, offset=shard_file_offset)
    

    def get(self, segids, lods=None, concat=True):
        """Fetch meshes at all levels of details.

        Parameters:
        segids: (iterable or int) segids to render

        lods: int, [int, ..] or None
            Level of detail(s) to retrieve.  0 is highest level of detail.
            None will fetch meshes at all levels.

        Optional:
          concat: bool, concatenate fragments (per segment per lod)

        Returns:
        { lod: { segid: { Mesh } } }
        ... or if concatenate=False: { lod: { segid: { Mesh, ... } } }

        Reference:
            https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/meshes.md
        """

        segids = toiter(segids)

        # decode all the fragments
        meshdata = defaultdict(lambda: defaultdict(list))
        for segid in segids:
            # Read the manifest (with a tweak to sharding.py to get the offset)
            result = self.reader.get_data(segid, self.meta.mesh_path, return_offset=True)
            manifest = self.get_manifest(segid)
            if manifest == None:
                raise exceptions.MeshDecodeError(red(
                    'Manifest not found for segment {}.'.format(segid)
                ))
            shard_file_offset = manifest.offset
            if lods == None:
                lods = list(range(manifest.num_lods))
            lods = toiter(lods)

            if any(lod >= manifest.num_lods for lod in lods):
                raise exceptions.MeshDecodeError(red(
                    'LOD value out of range ({} > {}) for segment {}.'.format(max(lods), manifest.num_lods, segid)
                ))

            # Read the data for all LODs
            fragment_sizes = [ np.sum(lod_fragment_sizes) for lod_fragment_sizes in manifest.fragment_offsets ]
            total_fragment_size = np.sum(fragment_sizes)

            # Kludge to hijack sharding.py to read the data
            shard_file_name = self.reader.get_filename(segid)
            full_path = self.reader.meta.join(self.reader.meta.cloudpath, self.path)
            stor =  SimpleStorage(full_path)

            for lod in lods:
                lod_binary = stor.get_file(shard_file_name,
                        start=(shard_file_offset - total_fragment_size) + np.sum(fragment_sizes[0:lod]),
                        end=(shard_file_offset - total_fragment_size) + np.sum(fragment_sizes[0:lod+1]))
                lod_meshes = []
                for frag in range(manifest.fragment_offsets[lod].shape[0]):
                    frag_binary = lod_binary[
                                            int(np.sum(manifest.fragment_offsets[lod][0:frag])) :
                                            int(np.sum(manifest.fragment_offsets[lod][0:frag+1]))
                                            ]
                    if len(frag_binary) == 0:
                        # According to @JBMS, empty fragments are used in cases where a child fragment exists,
                        # but its parent does not have a corresponding fragment, a possible byproduct of running
                        # marching cubes and mesh simplification independently for each level of detail.
                        continue
                    mesh = Mesh.from_draco(frag_binary)

                    # Conversion references:
                    # https://github.com/google/neuroglancer/blob/master/src/neuroglancer/mesh/draco/neuroglancer_draco.cc
                    # Treat the draco result as integers in the range [0, 2**vertex_quantization_bits)
                    mesh.vertices = mesh.vertices.view(dtype=np.int32)
                    
                    # Convert from "stored model" space to "model" space
                    mesh.vertices = manifest.grid_origin + manifest.vertex_offsets[lod] + \
                                    manifest.chunk_shape * (2 ** lod) * \
                                    (manifest.fragment_positions[lod][:,frag] + \
                                    (mesh.vertices / (2.0 ** self.vertex_quantization_bits - 1)))

                    # Scale to native (nm) space
                    mesh.vertices =  mesh.vertices * (self.transform[0,0], self.transform[1,1], self.transform[2,2])
                    
                    meshdata[lod][segid].append(mesh)

        if concat:
            for lod in meshdata:
                for segid in meshdata[lod]:
                    meshdata[lod][segid] = Mesh.concatenate(*meshdata[lod][segid])

        return meshdata

class MultiLevelPrecomputedMeshManifest:
    # Parse the multi-resolution mesh manifest file format:
    # https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/meshes.md
    # https://github.com/google/neuroglancer/blob/master/src/neuroglancer/mesh/multiscale.ts

    def __init__(self, binary, segment_id, offset):
        self._segment = segment_id
        self._binary = binary
        self._offset = offset

        # num_loads is the 7th word
        num_lods = int.from_bytes(self._binary[6*4:7*4], byteorder='little', signed=False)

        header_dt = np.dtype([('chunk_shape', np.float32, (3,)),
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
