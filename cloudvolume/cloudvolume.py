from __future__ import print_function

import itertools
import collections
import gevent.socket
import json
import os
import sys
import uuid
import weakref
import socket
import traceback

from six.moves import range
import numpy as np
from tqdm import tqdm
from six import string_types
import multiprocessing as mp
from time import strftime

from . import lib
from .cacheservice import CacheService
from . import exceptions 
from .lib import ( 
  colorize, red, mkdir, Vec, Bbox,  
  jsonify, generate_random_string
)

from .datasource import autocropfn
from .datasource.boss.metadata import BossMetadata
from .datasource.boss.image import BossImageSource 
from .datasource.precomputed.image import PrecomputedImageSource
from .datasource.precomputed.metadata import PrecomputedMetadata
from .datasource.precomputed.mesh import PrecomputedMeshSource
from .datasource.precomputed.skeleton import PrecomputedSkeletonSource
from .provenance import DataLayerProvenance
from .storage import SimpleStorage, Storage, reset_connection_pools
from .volumecutout import VolumeCutout
from . import sharedmemory

# Set the interpreter bool
try:
  INTERACTIVE = bool(sys.ps1)
except AttributeError:
  INTERACTIVE = bool(sys.flags.interactive)

def warn(text):
  print(colorize('yellow', text))

class SharedConfiguration(object):
  def __init__(
    self, cdn_cache, compress, green, 
    mip, parallel, progress,
    *args, **kwargs
  ):
    if type(parallel) == bool:
      parallel = mp.cpu_count() if parallel == True else 1
    elif parallel <= 0:
      raise ValueError('Number of processes must be >= 1. Got: ' + str(parallel))
    else:
      parallel = int(parallel)

    self.cdn_cache = cdn_cache
    self.compress = compress 
    self.green = bool(green)
    self.mip = mip
    self.parallel = parallel 
    self.progress = bool(progress)
    self.args = args 
    self.kwargs = kwargs

class CloudVolume(object):
  """
  A "serverless" Python client for reading and writing arbitrarily large 
  Neuroglancer Precomputed volumes both locally and on cloud services using.  
  A CloudVolume instance represents a dataset interface at a given mip level 
  (it doesn't load the entire dataset into memory).  

  Neuroglancer datasets specify metadata in an `info` file located at the root 
  of a data layer. It contains, among other things, the bounds of the 
  volume described as a 3D "voxel_offset" and 3D "size" in voxels, and the
  resolution of the dataset.

  Example:

    from cloudvolume import CloudVolume

    vol = CloudVolume('gs://mylab/mouse/image', progress=True)
    image = vol[:,:,:] # Download an image stack as a numpy array
    vol[:,:,:] = image # Upload an image stack from a numpy array
    
    label = 1
    mesh = vol.mesh.get(label) 
    skel = vol.skeletons.get(label)

  Required:
    cloudpath: Path to the dataset layer. This should match storage's supported
      providers.

      e.g. Google: gs://$BUCKET/$DATASET/$LAYER/
           S3    : s3://$BUCKET/$DATASET/$LAYER/
           Lcl FS: file:///tmp/$DATASET/$LAYER/
           Boss  : boss://$COLLECTION/$EXPERIMENT/$CHANNEL
           HTTP/S: http(s)://.../$CHANNEL
           matrix: matrix://$BUCKET/$DATASET/$LAYER/
  Optional:
    autocrop: (bool) If the specified retrieval bounding box exceeds the
        volume bounds, process only the area contained inside the volume. 
        This can be useful way to ensure that you are staying inside the 
        bounds when `bounded=False`.
    bounded: (bool) If a region outside of volume bounds is accessed:
        True: Throw an error
        False: Allow accessing the region. If no files are present, an error 
            will still be thrown. Consider combining this option with 
            `fill_missing=True`. However, this can be dangrous as it allows
            missing files and potentially network errors to be intepreted as 
            zeros.
    cache: (bool or str) Store downs and uploads in a cache on disk
          and preferentially read from it before redownloading.
        - falsey value: no caching will occur.
        - True: cache will be located in a standard location.
        - non-empty string: cache is located at this file path

        After initialization, you can adjust this setting via:
        `cv.cache.enabled = ...` which accepts the same values.

    cdn_cache: (int, bool, or str) Sets Cache-Control HTTP header on uploaded 
      image files. Most cloud providers perform some kind of caching. As of 
      this writing, Google defaults to 3600 seconds. Most of the time you'll 
      want to go with the default. 
      - int: number of seconds for cache to be considered fresh (max-age)
      - bool: True: max-age=3600, False: no-cache
      - str: set the header manually
    compress: (bool, str, None) pick which compression method to use. 
        None: (default) gzip for raw arrays and no additional compression
          for compressed_segmentation and fpzip.
        bool: 
          True=gzip, 
          False=no compression, Overrides defaults
        str: 
          'gzip': Extension so that we can add additional methods in the future 
                  like lz4 or zstd. 
          '': no compression (same as False).
    compress_cache: (None or bool) If not None, override default compression 
        behavior for the cache.
    delete_black_uploads: (bool) If True, on uploading an entirely black chunk,
        issue a DELETE request instead of a PUT. This can be useful for avoiding storing
        tiny files in the region around an ROI. Some storage systems using erasure coding 
        don't do well with tiny file sizes.
    fill_missing: (bool) If a chunk file is unable to be fetched:
        True: Use a block of zeros
        False: Throw an error
    green_threads: (bool) Use green threads instead of preemptive threads. This
      can result in higher download performance for some compression types. Preemptive
      threads seem to reduce performance on multi-core machines that aren't densely
      loaded as the CPython threads are assigned to multiple cores and the thrashing
      + GIL reduces performance. You'll need to add the following code to the top
      of your program to use green threads:

          import gevent.monkey
          gevent.monkey.patch_all(threads=False)

    info: (dict) In lieu of fetching a neuroglancer info file, use this one.
        This is useful when creating new datasets and for repeatedly initializing
        a new cloudvolume instance.
    non_aligned_writes: (bool) Enable non-aligned writes. Not multiprocessing 
        safe without careful design. When not enabled, a 
        cloudvolume.exceptions.AlignmentError is thrown for non-aligned writes. 
        
        https://github.com/seung-lab/cloud-volume/wiki/Advanced-Topic:-Non-Aligned-Writes

    mip: (int or iterable) Which level of downsampling to read and write from.
        0 is the highest resolution. You can also specify the voxel resolution
        like mip=[6,6,30] which will search for the appropriate mip level.
    parallel (int: 1, bool): Number of extra processes to launch, 1 means only 
        use the main process. If parallel is True use the number of CPUs 
        returned by multiprocessing.cpu_count(). When parallel > 1, shared
        memory (Linux) or emulated shared memory via files (other platforms) 
        is used by the underlying download.
    progress: (bool) Show progress bars. 
        Defaults to True in interactive python, False in script execution mode.
    provenance: (string, dict) In lieu of fetching a provenance 
        file, use this one. 
  """
  def __init__(self, 
    cloudpath, mip=0, bounded=True, autocrop=False, 
    fill_missing=False, cache=False, compress_cache=None, 
    cdn_cache=True, progress=INTERACTIVE, info=None, provenance=None, 
    compress=None, non_aligned_writes=False, parallel=1,
    delete_black_uploads=False, green_threads=False
  ):

    path = lib.extract_path(cloudpath)

    # hack around python's inability to 
    # pass primatives by reference. 
    # We would like updates to e.g. mip or parallel
    # to be passively absorbed by all listening 
    # data sources rather than work hard to actively
    # synchronize them.
    self.config = SharedConfiguration(
      cdn_cache=cdn_cache, 
      compress=compress, 
      green=green_threads,
      mip=mip, 
      parallel=parallel, 
      progress=progress,
    )
    self.green_threads = green_threads # trigger warning message

    self.cache = CacheService(
      cloudpath=(cache if type(cache) == str else cloudpath),
      enabled=bool(cache), 
      config=self.config, 
      compress=compress_cache,
    )

    if path.protocol != 'boss':
      self.meta = PrecomputedMetadata(
        cloudpath, cache=self.cache, 
        info=info, provenance=provenance, 
      )

      self.image = PrecomputedImageSource(
        self.config, self.meta, self.cache, 
        autocrop=bool(autocrop),
        bounded=bool(bounded),
        non_aligned_writes=bool(non_aligned_writes), 
        fill_missing=bool(fill_missing), 
        delete_black_uploads=bool(delete_black_uploads), 
      )
    else:
      self.meta = BossMetadata(
        cloudpath, cache=self.cache, 
        info=info, 
      )

      self.image = BossImageSource(
        self.config, self.meta, self.cache, 
        autocrop=bool(autocrop),
        bounded=bool(bounded),
        non_aligned_writes=bool(non_aligned_writes), 
      )

    self.mesh = PrecomputedMeshSource(self.meta, self.cache, self.config)
    self.skeleton = PrecomputedSkeletonSource(self.meta, self.cache, self.config)

    # needs to be set after info is defined since
    # its setter is based off of scales
    self.mip = mip
    self.pid = os.getpid()

  @property 
  def autocrop(self):
    return self.image.autocrop

  @autocrop.setter
  def autocrop(self, val):
    self.image.autocrop = val

  @property 
  def bounded(self):
    return self.image.bounded

  @bounded.setter 
  def bounded(self, val):
    self.image.bounded = val

  @property
  def fill_missing(self):
    return self.image.fill_missing
  
  @fill_missing.setter
  def fill_missing(self, val):
    self.image.fill_missing = val
  
  @property
  def green_threads(self):
    return self.config.green
  
  @green_threads.setter 
  def green_threads(self, val):
    if val and socket.socket is not gevent.socket.socket:
      warn("""
      WARNING: green_threads is set but this process is 
      not monkey patched. This will cause severely degraded
      performance.
      
      CloudVolume uses gevent for cooperative (green) 
      threading but it requires patching the Python standard 
      library to perform asynchronous IO. Add this code to
      the top of your program (before any other imports):

        import gevent.monkey
        gevent.monkey.patch_all(threads=False)

      More Information:

      http://www.gevent.org/intro.html#monkey-patching
      """)

    self.config.green = bool(val)

  @property
  def non_aligned_writes(self):
    return self.image.non_aligned_writes

  @non_aligned_writes.setter
  def non_aligned_writes(self, val):
    self.image.non_aligned_writes = val

  @property
  def delete_black_uploads(self):
    return self.image.delete_black_uploads

  @delete_black_uploads.setter
  def delete_black_uploads(self, val):
    self.image.delete_black_uploads = val

  @property
  def parallel(self):
    return self.config.parallel

  @parallel.setter
  def parallel(self, num_processes):
    if type(num_processes) == bool:
      num_processes = mp.cpu_count() if num_processes == True else 1
    elif num_processes <= 0:
      raise ValueError('Number of processes must be >= 1. Got: ' + str(num_processes))
    else:
      num_processes = int(num_processes)

    self.config.parallel = num_processes

  @property
  def cdn_cache(self):
    return self.config.cdn_cache

  @cdn_cache.setter 
  def cdn_cache(self, val):
    self.config.cdn_cache = val

  @property 
  def compress(self):
    return self.config.compress

  @compress.setter 
  def compress(self, val):
    self.config.compress = val 

  @property 
  def progress(self):
    return self.config.progress 

  @progress.setter 
  def progress(self, val):
    self.config.progress = bool(val)

  @property 
  def info(self):
    return self.meta.info

  @info.setter
  def info(self, val):
    self.meta.info = val
  
  @property
  def provenance(self):
    return self.meta.provenance

  @provenance.setter
  def provenance(self, val):
    self.meta.provenance = val

  @classmethod
  def from_numpy(cls, 
      arr, 
      vol_path='file:///tmp/image/'+generate_random_string(),
      resolution=(4,4,40), voxel_offset=(0,0,0), 
      chunk_size=(128,128,64), layer_type=None, max_mip=0,
      encoding='raw', compress=None
    ):
    """
    Create a new dataset from a numpy array.

    max_mip: (int) the maximum mip level id in the info file. 
    Note that currently the numpy array can only sit in mip 0,
    the max_mip was only created in info file.
    the numpy array itself was not downsampled. 
    """
    if not layer_type:
      if arr.dtype in (np.bool, np.uint32, np.uint64, np.uint16):
        layer_type = 'segmentation'
      elif np.issubdtype(arr.dtype, np.integer) \
                        or np.issubdtype(arr.dtype, np.floating):
        layer_type = 'image'
      else:
        raise NotImplementedError

    if arr.ndim == 3:
      num_channels = 1
    elif arr.ndim == 4:
      num_channels = arr.shape[-1]
    else:
      raise NotImplementedError

    info = cls.create_new_info(
      num_channels, layer_type, arr.dtype.name, 
      encoding, resolution, 
      voxel_offset, arr.shape[:3], 
      chunk_size=chunk_size, max_mip=max_mip
    )
    vol = CloudVolume(vol_path, info=info, bounded=True, compress=compress) 
    # save the info file
    vol.commit_info()
    vol.provenance.processing.append({
      'method': 'from_numpy',
      'date': strftime('%Y-%m-%d %H:%M %Z')
    })
    vol.commit_provenance()
    # save the numpy array
    vol[:,:,:] = arr
    return vol 

  def __setstate__(self, d):
    """Called when unpickling which is integral to multiprocessing."""
    self.__dict__ = d 

    # if 'cache' in d:
    #   self.init_submodules(d['cache'].enabled)
    # else:
    #   self.init_submodules(False)
    
    pid = os.getpid()
    if 'pid' in d and d['pid'] != pid:
      # otherwise the pickle might have references to old connections
      reset_connection_pools() 
      self.pid = pid
  
  @classmethod
  def create_new_info(cls, 
    num_channels, layer_type, data_type, encoding, 
    resolution, voxel_offset, volume_size, 
    mesh=None, skeletons=None, chunk_size=(64,64,64),
    compressed_segmentation_block_size=(8,8,8),
    max_mip=0, factor=Vec(2,2,1), *args, **kwargs
  ):
    """
    Create a new neuroglancer Precomputed info file.

    Required:
      num_channels: (int) 1 for grayscale, 3 for RGB 
      layer_type: (str) typically "image" or "segmentation"
      data_type: (str) e.g. "uint8", "uint16", "uint32", "float32"
      encoding: (str) "raw" for binaries like numpy arrays, "jpeg"
      resolution: int (x,y,z), x,y,z voxel dimensions in nanometers
      voxel_offset: int (x,y,z), beginning of dataset in positive cartesian space
      volume_size: int (x,y,z), extent of dataset in cartesian space from voxel_offset
    
    Optional:
      mesh: (str) name of mesh directory, typically "mesh"
      skeletons: (str) name of skeletons directory, typically "skeletons"
      chunk_size: int (x,y,z), dimensions of each downloadable 3D image chunk in voxels
      compressed_segmentation_block_size: (x,y,z) dimensions of each compressed sub-block
        (only used when encoding is 'compressed_segmentation')
      max_mip: (int), the maximum mip level id.
      factor: (Vec), the downsampling factor for each mip level

    Returns: dict representing a single mip level that's JSON encodable
    """
    return PrecomputedMetadata.create_info(
      num_channels, layer_type, data_type, encoding, 
      resolution, voxel_offset, volume_size, 
      mesh, skeletons, chunk_size,
      compressed_segmentation_block_size,
      max_mip, factor,
      *args, **kwargs
    )

  def refresh_info(self):
    """Restore the current info from cache or storage."""
    return self.meta.refresh_info()

  def commit_info(self):
    return self.meta.commit_info()

  def refresh_provenance(self):
    return self.meta.refresh_provenance()

  def commit_provenance(self):
    return self.meta.commit_provenance()

  @property
  def dataset_name(self):
    return self.meta.dataset
  
  @property
  def layer(self):
    return self.meta.layer

  @property
  def mip(self):
    return self.config.mip

  @mip.setter
  def mip(self, mip):
    mip = list(mip) if isinstance(mip, collections.Iterable) else int(mip)
    try:
      if isinstance(mip, list):  # mip specified by voxel resolution
        self.config.mip = next((i for (i,s) in enumerate(self.scales)
                          if s["resolution"] == mip))
      else:  # mip specified by index into downsampling hierarchy
        self.config.mip = self.available_mips[mip]
    except Exception:
      if isinstance(mip, list):
        opening_text = "Scale <{}>".format(", ".join(map(str, mip)))
      else:
        opening_text = "MIP {}".format(str(mip))
  
      scales = [ ",".join(map(str, scale)) for scale in self.available_resolutions ]
      scales = [ "<{}>".format(scale) for scale in scales ]
      scales = ", ".join(scales)
      msg = "{} not found. {} available: {}".format(
        opening_text, len(self.available_mips), scales
      )
      raise exceptions.ScaleUnavailableError(msg)

  @property
  def scales(self):
    return self.meta.scales

  @scales.setter
  def scales(self, val):
    self.meta.scales = val

  @property
  def scale(self):
    return self.meta.scale(self.mip)

  @scale.setter
  def scale(self, val):
    self.info['scales'][self.mip] = val

  def mip_scale(self, mip):
    return self.meta.scale(mip)

  @property
  def basepath(self):
    return self.meta.basepath

  @property 
  def layerpath(self):
    return self.meta.layerpath

  @property
  def base_cloudpath(self):
    return self.meta.base_cloudpath

  @property 
  def cloudpath(self):
    return self.layer_cloudpath

  @property
  def layer_cloudpath(self):
    return self.meta.cloudpath

  @property
  def info_cloudpath(self):
    return self.meta.infopath

  @property
  def cache_path(self):
    return self.cache.path

  @property
  def shape(self):
    """Returns Vec(x,y,z,channels) shape of the volume similar to numpy.""" 
    return self.meta.shape(self.mip)

  def mip_shape(self, mip):
    return self.meta.shape(mip)

  @property
  def volume_size(self):
    """Returns Vec(x,y,z) shape of the volume (i.e. shape - channels).""" 
    return self.meta.volume_size(self.mip)

  def mip_volume_size(self, mip):
    return self.meta.volume_size(mip)

  @property
  def available_mips(self):
    """Returns a list of mip levels that are defined."""
    return self.meta.available_mips

  @property
  def available_resolutions(self):
    """Returns a list of defined resolutions."""
    return (s["resolution"] for s in self.scales)

  @property
  def layer_type(self):
    """e.g. 'image' or 'segmentation'"""
    return self.meta.layer_type

  @property
  def dtype(self):
    """e.g. 'uint8'"""
    return self.meta.dtype

  @property
  def data_type(self):
    return self.meta.data_type

  @property
  def encoding(self):
    """e.g. 'raw' or 'jpeg'"""
    return self.meta.encoding(self.mip)

  def mip_encoding(self, mip):
    return self.meta.encoding(mip)

  @property
  def compressed_segmentation_block_size(self):
    return self.mip_compressed_segmentation_block_size(self.mip)

  def mip_compressed_segmentation_block_size(self, mip):
    if 'compressed_segmentation_block_size' in self.info['scales'][mip]:
      return self.info['scales'][mip]['compressed_segmentation_block_size']
    return None

  @property
  def num_channels(self):
    return self.meta.num_channels

  @property
  def voxel_offset(self):
    """Vec(x,y,z) start of the dataset in voxels"""
    return self.meta.voxel_offset(self.mip)

  def mip_voxel_offset(self, mip):
    return self.meta.voxel_offset(mip)

  @property 
  def resolution(self):
    """Vec(x,y,z) dimensions of each voxel in nanometers"""
    return self.meta.resolution(self.mip)

  def mip_resolution(self, mip):
    return self.meta.resolution(mip)

  @property
  def downsample_ratio(self):
    """Describes how downsampled the current mip level is as an (x,y,z) factor triple."""
    return self.meta.downsample_ratio(self.mip)

  @property
  def chunk_size(self):
    """Underlying chunk size dimensions in voxels. Synonym for underlying."""
    return self.meta.chunk_size(self.mip)

  def mip_chunk_size(self, mip):
    return self.meta.chunk_size(mip)

  @property
  def underlying(self):
    """Underlying chunk size dimensions in voxels. Synonym for chunk_size."""
    return self.meta.chunk_size(self.mip)

  def mip_underlying(self, mip):
    return self.meta.chunk_size(mip)

  @property
  def key(self):
    """The subdirectory within the data layer containing the chunks for this mip level"""
    return self.meta.key(self.mip)

  def mip_key(self, mip):
    return self.meta.key(mip)

  @property
  def bounds(self):
    """Returns a bounding box for the dataset with dimensions in voxels"""
    return self.meta.bounds(self.mip)

  def mip_bounds(self, mip):
    offset = self.meta.voxel_offset(mip)
    shape = self.meta.volume_size(mip)
    return Bbox( offset, offset + shape )

  def point_to_mip(self, pt, mip, to_mip):
    return self.meta.point_to_mip(pt, mip, to_mip)

  def bbox_to_mip(self, bbox, mip, to_mip):
    """Convert bbox or slices from one mip level to another."""
    return self.meta.bbox_to_mip(bbox, mip, to_mip)

  def slices_to_global_coords(self, slices):
    """
    Used to convert from a higher mip level into mip 0 resolution.
    """
    bbox = self.meta.bbox_to_mip(slices, self.mip, 0)
    return bbox.to_slices()

  def slices_from_global_coords(self, slices):
    """
    Used for converting from mip 0 coordinates to upper mip level
    coordinates. This is mainly useful for debugging since the neuroglancer
    client displays the mip 0 coordinates for your cursor.
    """
    bbox = self.meta.bbox_to_mip(slices, 0, self.mip)
    return bbox.to_slices()

  def reset_scales(self):
    """Used for manually resetting downsamples if something messed up."""
    self.meta.reset_scales()
    return self.commit_info()

  def add_scale(self, factor, encoding=None, chunk_size=None, info=None):
    """
    Generate a new downsample scale to for the info file and return an updated dictionary.
    You'll still need to call self.commit_info() to make it permenant.

    Required:
      factor: int (x,y,z), e.g. (2,2,1) would represent a reduction of 2x in x and y

    Optional:
      encoding: force new layer to e.g. jpeg or compressed_segmentation
      chunk_size: force new layer to new chunk size

    Returns: info dict
    """
    return self.meta.add_scale(factor, encoding, chunk_size, info)

  def exists(self, bbox_or_slices):
    """
    Produce a summary of whether all the requested chunks exist.

    bbox_or_slices: accepts either a Bbox or a tuple of slices representing
      the requested volume. 
    Returns: { chunk_file_name: boolean, ... }
    """
    return self.image.exists(bbox_or_slices)

  def delete(self, bbox_or_slices):
    """
    Delete the files within the bounding box.

    bbox_or_slices: accepts either a Bbox or a tuple of slices representing
      the requested volume. 
    """
    return self.image.delete(bbox_or_slices)

  def transfer_to(self, cloudpath, bbox, block_size=None, compress=True):
    """
    Transfer files from one storage location to another, bypassing
    volume painting. This enables using a single CloudVolume instance
    to transfer big volumes. In some cases, gsutil or aws s3 cli tools
    may be more appropriate. This method is provided for convenience. It
    may be optimized for better performance over time as demand requires.

    cloudpath (str): path to storage layer
    bbox (Bbox object): ROI to transfer
    block_size (int): number of file chunks to transfer per I/O batch.
    compress (bool): Set to False to upload as uncompressed
    """
    return self.image.transfer_to(cloudpath, bbox, self.mip, block_size, compress)

  def __getitem__(self, slices):
    if type(slices) == Bbox:
      slices = slices.to_slices()

    slices = self.meta.bbox(self.mip).reify_slices(slices, bounded=self.bounded)
    steps = Vec(*[ slc.step for slc in slices ])
    channel_slice = slices.pop()
    requested_bbox = Bbox.from_slices(slices)

    img = self.download(requested_bbox, self.mip)
    return img[::steps.x, ::steps.y, ::steps.z, channel_slice]

  def download(self, bbx, mip=None, parallel=None):
    if self.autocrop:
      bbx = Bbox.intersection(bbx, self.bounds)

    if mip is None:
      mip = self.mip

    if parallel is None:
      parallel = self.parallel

    return self.image.download(bbx, mip, parallel=parallel)

  def download_point(self, pt, size=256, mip=None):
    """
    Download to the right of point given in mip 0 coords.
    Useful for quickly visualizing a neuroglancer coordinate
    at an arbitary mip level.

    pt: (x,y,z)
    size: int or (sx,sy,sz)

    Return: image
    """
    if isinstance(size, int):
      size = Vec(size, size, size)
    else:
      size = Vec(*size)

    if mip is None:
      mip = self.mip

    size2 = size // 2

    pt = self.point_to_mip(pt, mip=0, to_mip=mip)
    bbox = Bbox(pt - size2, pt + size2).astype(np.int64)

    if parallel is None:
      parallel = self.parallel

    return self.image.download(bbox, mip, parallel=parallel)

  def unlink_shared_memory(self):
    """Unlink the current shared memory location from the filesystem."""
    return self.image.unlink_shared_memory()

  def download_to_shared_memory(self, slices, location=None, mip=None):
    """
    Download images to a shared memory array. 

    https://github.com/seung-lab/cloud-volume/wiki/Advanced-Topic:-Shared-Memory

    tip: If you want to use slice notation, np.s_[...] will help in a pinch.

    MEMORY LIFECYCLE WARNING: You are responsible for managing the lifecycle of the 
      shared memory. CloudVolume will merely write to it, it will not unlink the 
      memory automatically. To fully clear the shared memory you must unlink the 
      location and close any mmap file handles. You can use `cloudvolume.sharedmemory.unlink(...)`
      to help you unlink the shared memory file or `vol.unlink_shared_memory()` if you do 
      not specify location (meaning the default instance location is used).

    EXPERT MODE WARNING: If you aren't sure you need this function (e.g. to relieve 
      memory pressure or improve performance in some way) you should use the ordinary 
      download method of img = vol[:]. A typical use case is transferring arrays between 
      different processes without making copies. For reference, this  feature was created 
      for downloading a 62 GB array and working with it in Julia.

    Required:
      slices: (Bbox or list of slices) the bounding box the shared array represents. For instance
        if you have a 1024x1024x128 volume and you're uploading only a 512x512x64 corner
        touching the origin, your Bbox would be `Bbox( (0,0,0), (512,512,64) )`.
    Optional:
      location: (str) Defaults to self.shared_memory_id. Shared memory location 
        e.g. 'cloudvolume-shm-RANDOM-STRING' This typically corresponds to a file 
        in `/dev/shm` or `/run/shm/`. It can also be a file if you're using that for mmap. 
    
    Returns: ndarray backed by shared memory
    """
    if mip is None:
      mip = self.mip

    slices = self.meta.bbox(mip).reify_slices(slices, bounded=self.bounded)
    steps = Vec(*[ slc.step for slc in slices ])
    channel_slice = slices.pop()
    requested_bbox = Bbox.from_slices(slices)

    if self.autocrop:
      requested_bbox = Bbox.intersection(requested_bbox, self.bounds)

    img = self.image.download(
      requested_bbox, mip, parallel=self.parallel,
      location=location, retain=True, use_shared_memory=True
    )
    return img[::steps.x, ::steps.y, ::steps.z, channel_slice]

  def download_to_file(self, path, bbox, mip=None):
    """
    Download images directly to a file.

    Required:
      slices: (Bbox) the bounding box the shared array represents. For instance
        if you have a 1024x1024x128 volume and you're uploading only a 512x512x64 corner
        touching the origin, your Bbox would be `Bbox( (0,0,0), (512,512,64) )`. 
      path: (str) 
    Optional:
      mip: (int; default: self.mip) The current resolution level.

    Returns: ndarray backed by an mmapped file
    """
    if mip is None:
      mip = self.mip

    slices = self.meta.bbox(mip).reify_slices(bbox, bounded=self.bounded)
    steps = Vec(*[ slc.step for slc in slices ])
    channel_slice = slices.pop()
    requested_bbox = Bbox.from_slices(slices)

    if self.autocrop:
      requested_bbox = Bbox.intersection(requested_bbox, self.bounds)

    img = self.image.download(
      requested_bbox, mip, parallel=self.parallel,
      location=lib.toabs(path), retain=True, use_file=True
    )
    return img[::steps.x, ::steps.y, ::steps.z, channel_slice]

  def __setitem__(self, slices, img):
    if type(slices) == Bbox:
      slices = slices.to_slices()

    slices = self.meta.bbox(self.mip).reify_slices(slices, bounded=self.bounded)
    bbox = Bbox.from_slices(slices)
    slice_shape = list(bbox.size())
    bbox = Bbox.from_slices(slices[:3])

    if np.isscalar(img):
      img = np.zeros(slice_shape, dtype=self.dtype) + img

    imgshape = list(img.shape)
    if len(imgshape) == 3:
      imgshape = imgshape + [ self.num_channels ]

    if not np.array_equal(imgshape, slice_shape):
      raise exceptions.AlignmentError("""
        Input image shape does not match slice shape.

        Image Shape: {}  
        Slice Shape: {}
      """.format(imgshape, slice_shape))

    if self.autocrop:
      if not self.bounds.contains_bbox(bbox):
        img, bbox = autocropfn(self.meta, img, bbox, self.mip)

    if bbox.subvoxel():
      return

    self.image.upload(img, bbox.minpt, self.mip, parallel=self.parallel)

  def upload_from_shared_memory(self, location, bbox, order='F', cutout_bbox=None):
    """
    Upload from a shared memory array. 

    https://github.com/seung-lab/cloud-volume/wiki/Advanced-Topic:-Shared-Memory

    tip: If you want to use slice notation, np.s_[...] will help in a pinch.

    MEMORY LIFECYCLE WARNING: You are responsible for managing the lifecycle of the 
      shared memory. CloudVolume will merely read from it, it will not unlink the 
      memory automatically. To fully clear the shared memory you must unlink the 
      location and close any mmap file handles. You can use `cloudvolume.sharedmemory.unlink(...)`
      to help you unlink the shared memory file.

    EXPERT MODE WARNING: If you aren't sure you need this function (e.g. to relieve 
      memory pressure or improve performance in some way) you should use the ordinary 
      upload method of vol[:] = img. A typical use case is transferring arrays between 
      different processes without making copies. For reference, this feature was created
      for uploading a 62 GB array that originated in Julia.

    Required:
      location: (str) Shared memory location e.g. 'cloudvolume-shm-RANDOM-STRING'
        This typically corresponds to a file in `/dev/shm` or `/run/shm/`. It can 
        also be a file if you're using that for mmap.
      bbox: (Bbox or list of slices) the bounding box the shared array represents. For instance
        if you have a 1024x1024x128 volume and you're uploading only a 512x512x64 corner
        touching the origin, your Bbox would be `Bbox( (0,0,0), (512,512,64) )`.
    Optional:
      cutout_bbox: (bbox or list of slices) If you only want to upload a section of the
        array, give the bbox in volume coordinates (not image coordinates) that should 
        be cut out. For example, if you only want to upload 256x256x32 of the upper 
        rightmost corner of the above example but the entire 512x512x64 array is stored 
        in memory, you would provide: `Bbox( (256, 256, 32), (512, 512, 64) )`

        By default, just upload the entire image.

    Returns: void
    """
    bbox = Bbox.create(bbox)
    cutout_bbox = Bbox.create(cutout_bbox) if cutout_bbox else bbox.clone()

    if not bbox.contains_bbox(cutout_bbox):
      raise exceptions.AlignmentError("""
        The provided cutout is not wholly contained in the given array. 
        Bbox:        {}
        Cutout:      {}
      """.format(bbox, cutout_bbox))

    if self.autocrop:
      cutout_bbox = Bbox.intersection(cutout_bbox, self.bounds)

    if cutout_bbox.subvoxel():
      return

    shape = list(bbox.size3()) + [ self.num_channels ]
    mmap_handle, shared_image = sharedmemory.ndarray(
      location=location, shape=shape, 
      dtype=self.dtype, order=order, 
      readonly=True
    )

    delta_box = cutout_bbox.clone() - bbox.minpt
    cutout_image = shared_image[ delta_box.to_slices() ]
    
    self.image.upload(
      cutout_image, cutout_bbox.minpt, self.mip,
      parallel=self.parallel, 
      location=location, 
      location_bbox=bbox,
      order=order,
      use_shared_memory=True,
    )
    mmap_handle.close()

  def upload_from_file(self, location, bbox, order='F', cutout_bbox=None):
    """
    Upload from an mmapped file.

    tip: If you want to use slice notation, np.s_[...] will help in a pinch.

    Required:
      location: (str) Shared memory location e.g. 'cloudvolume-shm-RANDOM-STRING'
        This typically corresponds to a file in `/dev/shm` or `/run/shm/`. It can 
        also be a file if you're using that for mmap.
      bbox: (Bbox or list of slices) the bounding box the shared array represents. For instance
        if you have a 1024x1024x128 volume and you're uploading only a 512x512x64 corner
        touching the origin, your Bbox would be `Bbox( (0,0,0), (512,512,64) )`.
    Optional:
      cutout_bbox: (bbox or list of slices) If you only want to upload a section of the
        array, give the bbox in volume coordinates (not image coordinates) that should 
        be cut out. For example, if you only want to upload 256x256x32 of the upper 
        rightmost corner of the above example but the entire 512x512x64 array is stored 
        in memory, you would provide: `Bbox( (256, 256, 32), (512, 512, 64) )`

        By default, just upload the entire image.

    Returns: void
    """        
    bbox = Bbox.create(bbox)
    cutout_bbox = Bbox.create(cutout_bbox) if cutout_bbox else bbox.clone()

    if not bbox.contains_bbox(cutout_bbox):
      raise exceptions.AlignmentError("""
        The provided cutout is not wholly contained in the given array. 
        Bbox:        {}
        Cutout:      {}
      """.format(bbox, cutout_bbox))

    if self.autocrop:
      cutout_bbox = Bbox.intersection(cutout_bbox, self.bounds)

    if cutout_bbox.subvoxel():
      return

    shape = list(bbox.size3()) + [ self.num_channels ]
    mmap_handle, shared_image = sharedmemory.ndarray_fs(
      location=lib.toabs(location), shape=shape, 
      dtype=self.dtype, order=order, 
      readonly=True, lock=None
    )

    delta_box = cutout_bbox.clone() - bbox.minpt
    cutout_image = shared_image[ delta_box.to_slices() ]
    
    self.image.upload(
      cutout_image, cutout_bbox.minpt, self.mip,
      parallel=self.parallel, 
      location=lib.toabs(location), 
      location_bbox=bbox,
      order=order,
      use_file=True,
    )
    mmap_handle.close()

  def viewer(self, port=1337):
    import cloudvolume.server

    cloudvolume.server.view(self.cloudpath, port=port)



