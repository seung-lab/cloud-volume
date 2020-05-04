import numpy as np

from ..sharding import ShardingSpecification, ShardReader
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
            binary = self.reader.get_data(segid, self.meta.mesh_path)
            print("Binary length: %d" % len(binary))
            manifest = MultiLevelPrecomputedMeshManifest(binary)
            print(manifest)

            print(manifest.lod_scales)
            print(manifest.fragment_positions)
            print(manifest.fragment_offsets)

            results.append([])
        if list_return:
            return results
        else:
            return results[0]

class MultiLevelPrecomputedMeshManifest:
    # Parse the multi-resolution mesh manifest file format:
    # https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/meshes.md#multi-resolution-mesh-format

    def __init__(self, binary):
        self._binary = binary

        # num_loads is at the 7th word
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
            ('%d' % level, np.uint32, (count, 3))
            for level, count in zip(
                range(num_lods),
                np.nditer(self.num_fragments_per_lod)
            )
        ])

        self._fragment_positions = np.frombuffer(
            self._binary[header_dt.itemsize:header_dt.itemsize+frag_pos_dt.itemsize],
            dtype=frag_pos_dt)

        frag_off_dt = np.dtype([
            ('%d' % level, np.uint32, (count,))
            for level, count in zip(
                range(num_lods),
                np.nditer(self.num_fragments_per_lod)
            )
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
        return self._header['num_lods']

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
        return self._fragment_positions

    @property
    def fragment_offsets(self):
        return self._fragment_offsets
