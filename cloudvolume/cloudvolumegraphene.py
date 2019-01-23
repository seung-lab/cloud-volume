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

import numpy as np
import multiprocessing as mp


from .cacheservice import CacheService
from .lib import (
    toabs, colorize, red, yellow,
    mkdir, clamp, xyzrange, Vec,
    Bbox, min2, max2, check_bounds,
    jsonify, generate_slices
)
from cloudvolume import CloudVolume
from .meshservice import GrapheneMeshService
from .skeletonservice import PrecomputedSkeletonService
from .storage import SimpleStorage, Storage, reset_connection_pools

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
                               output_to_shared_memory=output_to_shared_memory)

        # Init other parameters
        self.autocrop = bool(autocrop)
        self.bounded = bool(bounded)
        self.fill_missing = bool(fill_missing)
        self.progress = bool(progress)
        self.shared_memory_id = self.generate_shared_memory_location()
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
    def cloud_url(self):
        return self._cloud_url

    @property
    def info_endpoint(self):
        return f"{self.cloud_url}/info"

    @property
    def manifest_endpoint(self):
        return f"{self.cloud_url.replace('segmentation', 'meshing')}/manifest"

    @property
    def cloudpath(self):
        return self._info_dict["data_dir"]

    @property
    def graph_chunk_size(self):
        return self._info_dict["graph"]["chunk_size"]


    @property
    def _storage(self):
        return self._cv._storage

    @property
    def dataset_name(self):
        return self.info_endpoint.split("/")[-1]

### CloudVolume properties:

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

    def _get_leaves(self, root_id, bounds=None):
        """
        get the supervoxels for this root_id

        params
        ------
        root_id: uint64 root id to find supervoxels for
        bounds: 3x2 numpy array of bounds [[minx,maxx],[miny,maxy],[minz,maxz]]
        """
        url = f"{self._info_endpoint}/segment/{root_id}/leaves"
        query_d = {}
        if bounds is not None:
            bounds_str = []
            for b in bounds:
                bounds_str.append("-".join(str(b2) for b2 in b))
            bounds_str = "_".join(bounds_str)
            query_d['bounds'] = bounds_str

        response = self.session.post(url, json=[root_id], params=query_d)

        assert(response.status_code == 200)
        return np.frombuffer(response.content, dtype=np.uint64)
