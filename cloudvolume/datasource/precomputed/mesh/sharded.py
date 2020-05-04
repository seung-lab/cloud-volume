import numpy as np

from ..sharding import ShardingSpecification, ShardReader
from ....mesh import Mesh
from ....lib import red

class ShardedMultiLevelPrecomputedMeshSource(object):
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
            #binary = self.reader.get_multi_resolution_mesh(segid, self.meta.mesh_path)
            binary = self.reader.get_data(segid, self.meta.mesh_path)
            manifest = MultiLevelPrecomputedMeshManifest(binary)
            print(manifest)

        return
        if list_return:
            return results
        else:
            return results[0]

class MultiLevelPrecomputedMeshManifest(object):
    # Parse the multi-resolution mesh manifest file format:
    # https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/meshes.md#multi-resolution-mesh-format

    def __init__(self, binary):
        # num_loads is at the 7th word
        self.num_lods = np.fromstring(binary[24:28], np.uint32)[0]
        print(self.num_lods)

        header_dt = np.dtype([('chunk_shape', np.float32, (3,)),
                        ('grid_origin', np.float32, (3,)),
                        ('num_lods', np.uint32),
                        ('lod_scales', np.float32, (self.num_lods,)),
                        ('vertex_offsets', np.float32, (self.num_lods,3)),
                        ('num_fragments_per_lod', np.uint32, (self.num_lods,))
                        ])
        self.header = np.frombuffer(binary[0:header_dt.itemsize], dtype=header_dt)
        print(self.header)

        frag_pos_dt = np.dtype([
            ('%d' % level, np.uint32, (count, 3))
            for level, count in zip(
                range(self.num_lods),
                np.nditer(self.header['num_fragments_per_lod'])
            )
        ])
        print(frag_pos_dt)

        self.fragment_positions = np.frombuffer(
            binary[header_dt.itemsize:header_dt.itemsize+frag_pos_dt.itemsize],
            dtype=frag_pos_dt)

        print(self.fragment_positions)

        frag_off_dt = np.dtype([
            ('%d' % level, np.uint32, (count,))
            for level, count in zip(
                range(self.num_lods),
                np.nditer(self.header['num_fragments_per_lod'])
            )
        ])
        print(frag_off_dt)

        self.fragment_offsets = np.frombuffer(
            binary[header_dt.itemsize+frag_pos_dt.itemsize:
                   header_dt.itemsize+frag_pos_dt.itemsize+frag_off_dt.itemsize],
            dtype=frag_off_dt)

        print(self.fragment_offsets)

        return
