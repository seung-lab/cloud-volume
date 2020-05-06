import numpy as np

from ..sharding import ShardingSpecification, ShardReader
from ....storage import SimpleStorage
from ....mesh import Mesh
from ....lib import red

class ShardedMultiLevelPrecomputedMeshSource:
    def __init__(self, meta, cache, config, readonly=False):
        self.meta = meta
        self.cache = cache
        self.config = config
        self.readonly = bool(readonly)

        spec = ShardingSpecification.from_dict(self.meta.info['sharding'])
        self.reader = ShardReader(meta, cache, spec)

        print(spec)

    @property
    def path(self):
        return self.meta.mesh_path
    
    def get(self, segids):
        list_return = True
        if type(segids) in (int, float):
            list_return = False
            segids = [ int(segids) ]

        results = []
        for segid in segids:
            # Read the manifest (with a tweak to sharding.py to get the offset)
            binary, shard_file_offset = self.reader.get_data(segid, self.meta.mesh_path, return_offset=True)
            manifest = MultiLevelPrecomputedMeshManifest(binary)

            #print(manifest)
            #print(manifest.lod_scales)
            #print(manifest.fragment_positions)
            print(manifest.fragment_offsets)

            # Read the data for all LODs
            fragment_sizes = [ np.sum(lod) for lod in manifest.fragment_offsets ]
            total_fragment_size = np.sum(fragment_sizes)
            # Kludge to hijack sharding.py to read the data
            shard_file_name = self.reader.get_filename(segid)
            full_path = self.reader.meta.join(self.reader.meta.cloudpath, self.path)
            with SimpleStorage(full_path) as stor:
                binary = stor.get_file(shard_file_name,
                                    start=shard_file_offset - total_fragment_size,
                                    end=shard_file_offset)
            print("Read %d bytes" % len(binary))
            for lod in range(manifest.num_lods):
                print("Extracting LOD %d" % lod)
                lod_binary = binary[int(np.sum(fragment_sizes[0:lod])) : int(np.sum(fragment_sizes[0:lod+1]))]
                print("LOD data size is: ", len(lod_binary))

            results.append([])



        if list_return:
            return results
        else:
            return results[0]

class MultiLevelPrecomputedMeshManifest:
    # Parse the multi-resolution mesh manifest file format:
    # https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/meshes.md#multi-resolution-mesh-format
    # https://github.com/google/neuroglancer/blob/master/src/neuroglancer/mesh/multiscale.ts

    def __init__(self, binary):
        self._binary = binary

        # num_loads is the 7th word
        num_lods = int.from_bytes(self._binary[24:28], byteorder='little', signed=False)

        header_dt = np.dtype([('chunk_shape', np.float32, (3,)),
                        ('grid_origin', np.float32, (3,)),
                        ('num_lods', np.uint32),
                        ('lod_scales', np.float32, (num_lods,)),
                        ('vertex_offsets', np.float32, (num_lods,3)),
                        ('num_fragments_per_lod', np.uint32, (num_lods,))
                        ])
        self._header = np.frombuffer(self._binary[0:header_dt.itemsize], dtype=header_dt)

        frag_pos_dt = np.dtype([
            ('', np.uint32, (count, 3))
            for count in np.nditer(self.num_fragments_per_lod)
        ])

        self._fragment_positions = np.frombuffer(
            self._binary[header_dt.itemsize:header_dt.itemsize+frag_pos_dt.itemsize],
            dtype=frag_pos_dt)

        frag_off_dt = np.dtype([
            ('', np.uint32, (count,))
            for count in np.nditer(self.num_fragments_per_lod)
        ])

        self._fragment_offsets = np.frombuffer(
            self._binary[header_dt.itemsize+frag_pos_dt.itemsize:
                   header_dt.itemsize+frag_pos_dt.itemsize+frag_off_dt.itemsize],
            dtype=frag_off_dt)

    @property
    def chunk_shape(self):
        return self._header['chunk_shape']

    @property
    def grid_origin(self):
        return self._header['grid_origin']

    @property
    def num_lods(self):
        return int(self._header['num_lods'])

    @property
    def lod_scales(self):
        return self._header['lod_scales']

    @property
    def vertex_offsets(self):
        return self._header['vertex_offsets']

    @property
    def num_fragments_per_lod(self):
        return self._header['num_fragments_per_lod']

    @property
    def fragment_positions(self):
        return [ self._fragment_positions[field] for field in self._fragment_positions.dtype.names ]

    @property
    def fragment_offsets(self):
        return [ self._fragment_offsets[field] for field in self._fragment_offsets.dtype.names ]

    @property
    def length(self):
        return len(self._binary)
