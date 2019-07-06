from __future__ import print_function

from functools import partial
import itertools
import collections
import json
import os
import re
import requests
import sys
import weakref
import fastremap
import numpy as np
import multiprocessing as mp


from ..cacheservice import CacheService
from ..storage import SimpleStorage, Storage, reset_connection_pools

# Set the interpreter bool
try:
    INTERACTIVE = bool(sys.ps1)
except AttributeError:
    INTERACTIVE = bool(sys.flags.interactive)


def warn(text):
    print(colorize('yellow', text))


class CloudVolumeGraphene(object):
    """ This is CloudVolumeGraphene
    """

    def __init__(self, cloud_url, mip=0, bounded=True, autocrop=False,
                 fill_missing=False,
                 cache=False, compress_cache=None, cdn_cache=True,
                 progress=INTERACTIVE, provenance=None,
                 compress=None, parallel=1,
                 map_gs_to_https=False,
                 output_to_shared_memory=False):

        # Read info from chunkedgraph endpoint
        self._cloud_url = cloud_url
        self._info_dict = self.read_info()

        self._cv = CloudVolume(cloudpath=self.cloudpath,
                               info=self._info_dict,
                               mip=mip,
                               bounded=bounded,
                               autocrop=autocrop,
                               fill_missing=fill_missing,
                               cache=cache,
                               compress_cache=compress_cache,
                               cdn_cache=cdn_cache,
                               progress=progress,
                               provenance=provenance,
                               compress=compress,
                               non_aligned_writes=False,
                               parallel=parallel,
                               map_gs_to_https=map_gs_to_https,
                               output_to_shared_memory=output_to_shared_memory)

        # Init other parameters
        self.autocrop = bool(autocrop)
        self.bounded = bool(bounded)
        self.fill_missing = bool(fill_missing)
        self.progress = bool(progress)
        if type(output_to_shared_memory) == str:
            self.shared_memory_id = str(output_to_shared_memory)

        if type(parallel) == bool:
            self.parallel = mp.cpu_count() if parallel == True else 1
        else:
            self.parallel = int(parallel)

        if self.parallel <= 0:
            raise ValueError(
                'Number of processes must be >= 1. Got: ' + str(self.parallel))

        self.init_submodules(cache)
        self.cache.compress = compress_cache

        self.read_info()

        self._mip = mip
        self.pid = os.getpid()

### Graphene specific properties:

    @property
    def info(self):
        return self._info_dict

    @property
    def cloud_url(self):
        return self._cloud_url

    @property
    def info_endpoint(self):
        return "%s/info"%self.cloud_url

    @property
    def manifest_endpoint(self):
        return "%s/manifest"%self.cloud_url.replace('segmentation', 'meshing')

    @property
    def cloudpath(self):
        return self._info_dict["data_dir"]

    @property
    def graph_chunk_size(self):
        return self._info_dict["graph"]["chunk_size"]

    @property
    def mesh_chunk_size(self):
        #TODO: add this as new parameter to the info as it can be different from the chunkedgraph chunksize
        return self.graph_chunk_size

    @property
    def _storage(self):
        return self._cv._storage

    @property
    def dataset_name(self):
        return self.info_endpoint.split("/")[-1]

### CloudVolume properties:

    @property
    def layer_cloudpath(self):
        return self._cv.layer_cloudpath

    @property
    def mip(self):
        return self._cv.mip

    @property
    def scales(self):
        return self._cv.scales

    @property
    def scale(self):
        return self._cv.scale

    @property
    def shape(self):
        return self._cv.shape

    @property
    def volume_size(self):
        return self._cv.volume_size

    @property
    def available_mips(self):
        return self._cv.available_mips

    @property
    def available_resolutions(self):
        return self._cv.available_resolutions

    @property
    def layer_type(self):
        return self._cv.layer_type

    @property
    def dtype(self):
        return self._cv.dtype

    @property
    def data_type(self):
        return self._cv.data_type

    @property
    def encoding(self):
        return self._cv.encoding

    @property
    def compressed_segmentation_block_size(self):
        return self._cv.compressed_segmentation_block_size

    @property
    def num_channels(self):
        return self._cv.num_channels

    @property
    def voxel_offset(self):
        return self._cv.voxel_offset

    @property
    def resolution(self):
        return self._cv.resolution

    @property
    def downsample_ratio(self):
        return self._cv.downsample_ratio

    @property
    def chunk_size(self):
        return self._cv.chunk_size

    @property
    def underlying(self):
        return self._cv.underlying

    @property
    def key(self):
        return self._cv.key

    @property
    def bounds(self):
        return self._cv.bounds

    def __setstate__(self, d):
        """Called when unpickling which is integral to multiprocessing."""
        self.__dict__ = d

        if 'cache' in d:
            self.init_submodules(d['cache'].enabled)
        else:
            self.init_submodules(False)

        pid = os.getpid()
        if 'pid' in d and d['pid'] != pid:
            # otherwise the pickle might have references to old connections
            reset_connection_pools()
            self.pid = pid

    def read_info(self):
        """
        Reads info from chunkedgraph endpoint and extracts relevant information
        """

        r = requests.get(os.path.join(self._cloud_url, "info"))
        assert r.status_code == 200
        info_dict = json.loads(r.content)
        return info_dict

    def init_submodules(self, cache):
        """cache = path or bool"""

        self.cache = CacheService(cache, weakref.proxy(self))
        self.mesh = GrapheneMeshService(weakref.proxy(self))
        self.skeleton = PrecomputedSkeletonService(weakref.proxy(self))

    def mip_bounds(self, mip):
        self._cv.mip_bounds(mip)

    def bbox_to_mip(self, bbox, mip, to_mip):
        return self._cv.bbox_to_mip(bbox, mip, to_mip)

    def slices_to_global_coords(self, slices):
        return self._cv.slices_to_global_coords(slices)

    def slices_from_global_coords(self, slices):
        return self._cv.slices_from_global_coords(slices)

    def exists(self, bbox_or_slices):
        return self._cv.exists(bbox_or_slices)

    @staticmethod
    def _convert_root_id_list(root_ids):
        if isinstance(root_ids, int):
            return [root_ids]
        if isinstance(root_ids, list):
            return np.array(root_ids, dtype=np.uint64)
        if isinstance(root_ids, (np.ndarray, np.generic)):
            return np.array(root_ids.ravel(), dtype=np.uint64)
        return root_ids

    def __interpret_slices(self, slices):
        """
        Convert python slice objects into a more useful and computable form:

        - requested_bbox: A bounding box representing the volume requested
        - steps: the requested stride over x,y,z
        - channel_slice: A python slice object over the channel dimension

        Returned as a tuple: (requested_bbox, steps, channel_slice)
        """
        maxsize = list(self.bounds.maxpt) + [ 1 ]
        minsize = list(self.bounds.minpt) + [ 0 ]

        slices = generate_slices(slices, minsize, maxsize, bounded=self.bounded)
        channel_slice = slices.pop()

        minpt = Vec(*[ slc.start for slc in slices ])
        maxpt = Vec(*[ slc.stop for slc in slices ]) 
        steps = Vec(*[ slc.step for slc in slices ])

        return Bbox(minpt, maxpt), steps, channel_slice
        
    def __getitem__(self, slices):
        assert(len(slices) == 4)
        seg_cutout = self._cv.__getitem__(slices[:-1])
        root_ids = slices[-1]
        root_ids = self._convert_root_id_list(root_ids)
        bbox, steps, channel_slice = self.__interpret_slices(slices[:-1])
        bbox = bbox * [2**self.mip, 2**self.mip, 1]
        print(bbox, self.bounds)
        bbox=bbox.intersection(self.bounds* [2**self.mip, 2**self.mip, 1], bbox)
        print(bbox)
        remap_d = {}
        for root_id in root_ids:
            leaves = self._get_leaves(root_id, bbox)
            remap_d.update(dict(zip(leaves, [root_id]*len(leaves))))
        seg_cutout = fastremap.remap(seg_cutout, remap_d, preserve_missing_labels=True)
            # seg_cutout[np.isin(sup_voxels_cutout, leaves)] = root_id
        return seg_cutout

    def _get_leaves(self, root_id, bbox):
        """
        get the supervoxels for this root_id

        params
        ------
        root_id: uint64 root id to find supervoxels for
        bbox: cloudvolume.lib.Bbox 3d bounding box for segmentation
        """
        url = "%s/segment/%d/leaves" % (self._cloud_url, root_id)
        bounds_str = []
        for sl in bbox.to_slices():
            bounds_str.append(f"{sl.start}-{sl.stop}")
        bounds_str = "_".join(bounds_str)
        query_d = {
            'bounds': bounds_str
        }

        response = requests.post(url, json=[int(root_id)], params=query_d)

        assert(response.status_code == 200)
        return np.frombuffer(response.content, dtype=np.uint64)
