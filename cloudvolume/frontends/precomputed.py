from typing import Optional, Sequence

import itertools
import gevent.socket
import json
import os
import sys
import uuid
import socket

import fastremap
from six.moves import range
import numpy as np
from tqdm import tqdm
from six import string_types
import multiprocessing as mp

from .. import lib
from ..cacheservice import CacheService
from .. import exceptions 
from ..lib import ( 
  colorize, yellow, red, mkdir, 
  Vec, Bbox, jsonify, BboxLikeType,
)

from ..datasource import autocropfn
from ..datasource.precomputed import PrecomputedMetadata

from ..provenance import DataLayerProvenance
from ..storage import SimpleStorage, Storage, reset_connection_pools
from ..volumecutout import VolumeCutout
from .. import sharedmemory

def warn(text):
  print(colorize('yellow', text))

class CloudVolumePrecomputed(object):
  def __init__(self, 
    meta, cache, config,
    image=None, mesh=None, skeleton=None,
    mip=0
  ):
    self.config = config 
    self.cache = cache 
    self.meta = meta

    self.image = image
    self.mesh = mesh 
    self.skeleton = skeleton

    self.green_threads = self.config.green # display warning message

    # needs to be set after info is defined since
    # its setter is based off of scales
    self.mip = mip
    self.pid = os.getpid()

    is_placeholder = self.meta.check_for_placeholder_scale(self.mip)
    if is_placeholder:
      print(yellow("Warning: The currently selected mip level is marked as a placeholder and likely has no associated image data."))

  @property 
  def autocrop(self):
    return self.image.autocrop

  @autocrop.setter
  def autocrop(self, val):
    self.image.autocrop = val

  @property
  def background_color(self):
    return self.image.background_color 

  @background_color.setter
  def background_color(self, val):
    self.image.background_color = val  

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

  def __getstate__(self):
    # can't pickle a weakref
    if hasattr(self.mesh.meta, "_cv"):
      del self.mesh.meta._cv
    if hasattr(self.skeleton.meta, "_cv"):
      del self.skeleton.meta._cv

    return self.__dict__

  def __setstate__(self, d):
    """Called when unpickling which is integral to multiprocessing."""
    self.__dict__ = d 
    
    self.mesh.meta.cv = self
    self.skeleton.meta.cv = self

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
    max_mip=0, factor=Vec(2,2,1), redirect=None, 
    *args, **kwargs
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
      redirect: If this volume has moved, you can set an automatic redirect
        by specifying a cloudpath here.

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
    self.config.mip = self.meta.to_mip(mip)

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
  def ndim(self):
    return len(self.shape)

  def mip_ndim(self, mip):
    return len(self.meta.shape(mip))

  @property
  def shape(self):
    """Returns Vec(x,y,z,channels) shape of the volume similar to numpy.""" 
    return tuple(self.meta.shape(self.mip))

  def mip_shape(self, mip):
    return tuple(self.meta.shape(mip))

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

  def transfer_to(
    self, cloudpath, bbox, 
    block_size=None, compress=True, encoding=None,
    sharded=None,
  ):
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
    sharded (bool): set whether the destination should be sharded 
      or unsharded by default the state of the source volume is used.
    """
    if isinstance(bbox, Bbox):
      bbox = bbox.convert_units('vx', self.resolution)

    return self.image.transfer_to(
      cloudpath, bbox, self.mip, 
      block_size, compress, 
      encoding=encoding, sharded=sharded,
    )

  def coordinate_indexing(self, slices):
    """
    Limited version of fancy indexing for accepting x,y,z points.

    col = [0,1,2]

    e.g. arr[col,col,col]
    >>> array([129, 122, 11]) # (0,0,0), (1,1,1), and (2,2,2)
    """
    pts = [ (x,y,z) for x,y,z in zip(slices[0],slices[1],slices[2]) ]
    res = self.scattered_points(pts)
    return np.array([ res[tuple(pt)] for pt in pts ], dtype=self.dtype)

  def __getitem__(self, slices):
    if isinstance(slices, Bbox):
      slices = slices.convert_units(
        "vx", self.meta.resolution(self.mip)
      ).astype(int).to_slices()
    elif (
      hasattr(slices, "__len__") 
      and len(slices) == 3
      and all([ isinstance(slc, (list, tuple, np.ndarray)) for slc in slices ])
    ):
        return self.coordinate_indexing(slices)

    slices = self.meta.bbox(self.mip).reify_slices(slices, bounded=self.bounded)
    steps = Vec(*[ slc.step for slc in slices ])
    slices = [ slice(slc.start, slc.stop) for slc in slices ]
    channel_slice = slices.pop()
    requested_bbox = Bbox.from_slices(slices)

    img = self.download(requested_bbox, self.mip)
    return img[::steps.x, ::steps.y, ::steps.z, channel_slice]

  def unique(
    self, 
    bbox:BboxLikeType, 
    mip:Optional[int] = None,
    
    # Absorbing polymorphic Graphene calls
    agglomerate:Optional[bool] = None, 
    timestamp:Optional[int] = None, 
    stop_layer:Optional[int] = None,

    # new download arguments
    coord_resolution:Optional[Sequence[int]] = None,
  ):
    """
    Downloads segmentation and extracts unique
    labels from it without rendering a full image.
    Faster and saves memory.
    """
    if mip is None:
      mip = self.mip

    if isinstance(bbox, Bbox):
      bbox = bbox.convert_units('vx', self.meta.resolution(mip))

    bbox = Bbox.create(
      bbox, context=self.bounds, 
      bounded=(self.bounded and coord_resolution is None), 
      autocrop=self.autocrop
    )

    if coord_resolution is not None:
      factor = self.meta.resolution(mip) / coord_resolution
      bbox /= factor
      if self.bounded and not self.meta.bounds(mip).contains_bbox(bbox):
        raise exceptions.OutOfBoundsError(f"Computed {bbox} is not contained within bounds {self.meta.bounds(mip)}")

    return self.image.unique(
      bbox.astype(np.int64), mip
    )

  def download_files(
    self,
    bbox:BboxLikeType, 
    mip:Optional[int] = None, 
    parallel:Optional[int] = None,
    segids:Optional[Sequence[int]] = None, 
    
    # Absorbing polymorphic Graphene calls
    agglomerate:Optional[bool] = None, 
    timestamp:Optional[int] = None, 
    stop_layer:Optional[int] = None,

    coord_resolution:Optional[Sequence[int]] = None,
    cache_only:bool = False,
    decompress:bool = True,
  ):
    """
    Downloads files without rendering to an image.

    decompress: automatically decompress downloaded
      files. If False, returns the raw bytes.
    cache_only: discard downloaded files to avoid
      inflating memory. (you must enable the cache separately)

    Returns: { filename: binary }
    """
    if mip is None:
      mip = self.mip

    if isinstance(bbox, Bbox):
      bbox = bbox.convert_units('vx', self.meta.resolution(mip))

    bbox = Bbox.create(
      bbox, context=self.bounds, 
      bounded=(self.bounded and coord_resolution is None), 
      autocrop=self.autocrop
    )

    if coord_resolution is not None:
      factor = self.meta.resolution(mip) / coord_resolution
      bbox /= factor
      if self.bounded and not self.meta.bounds(mip).contains_bbox(bbox):
        raise exceptions.OutOfBoundsError(f"Computed {bbox} is not contained within bounds {self.meta.bounds(mip)}")

    if parallel is None:
      parallel = self.parallel

    files = self.image.download_files(
      bbox.astype(np.int64), mip, 
      parallel=parallel,
      decompress=decompress, 
      cache_only=cache_only
    )

    if not segids:
      return files

    for key in files:
      val = files[key]
      labels = set(chunks.labels(val))
      mask_labels = labels - set(segids)
      remap = { lbl: lbl for lbl in segids }
      if preserve_zeros:
        mask_value = np.inf
        if np.issubdtype(self.dtype, np.integer):
          mask_value = np.iinfo(self.dtype).max        
      remap.update({ mask_labels: mask_value for lbl in mask_labels })
      if preserve_zeros:
        remap[0] = 0

      files[key] = chunks.remap(
        files[key], 
        encoding=self.meta.encoding(mip), 
        shape=self.meta.chunk_size(mip),
        dtype=self.meta.dtype,
        block_size=self.meta.compressed_segmentation_block_size(mip),
        mapping=remap,
        preserve_missing_labels=True,
      )

    return files
    
  def download(
    self, 
    bbox:BboxLikeType, 
    mip:Optional[int] = None, 
    parallel:Optional[int] = None,
    segids:Optional[Sequence[int]] = None, 
    preserve_zeros:bool = False,
    
    # Absorbing polymorphic Graphene calls
    agglomerate:Optional[bool] = None, 
    timestamp:Optional[int] = None, 
    stop_layer:Optional[int] = None,

    # new download arguments
    renumber:bool = False, 
    coord_resolution:Optional[Sequence[int]] = None,
    label:Optional[int] = None,
  ) -> VolumeCutout:
    """
    Downloads segmentation from the indicated cutout
    region.

    bbox: specifies cutout to fetch
    mip: which resolution level to get (default self.mip)
    parallel: what parallel level to use (default self.parallel)

    segids: agglomerate the leaves of these segids from the graph 
      server and label them with the given segid.
    preserve_zeros: If segids is not None:
      False: mask other segids with zero
      True: mask other segids with the largest integer value
        contained by the image data type and leave zero as is.
    renumber: dynamically rewrite downloaded segmentation into
      a more compact data type. Only compatible with single-process
      non-sharded download.
    coord_resolution: (rx,ry,rz) the coordinate resolution of the input point.
      Sometimes Neuroglancer is working in the resolution of another
      higher res layer and this can help correct that.
    label: download as a binary image where this label is foreground (True)
      and everything else is background (False)

    agglomerate, timestamp, and stop_layer are just there to 
    absorb arguments to what could be a graphene frontend.

    Returns: img
    """
    if mip is None:
      mip = self.mip
    
    if isinstance(bbox, Bbox):
      bbox = bbox.convert_units('vx', self.meta.resolution(mip))

    bbox = Bbox.create(
      bbox, context=self.bounds, 
      bounded=(self.bounded and coord_resolution is None), 
      autocrop=self.autocrop
    )

    if coord_resolution is not None:
      factor = self.meta.resolution(mip) / coord_resolution
      bbox = bbox / factor
      if self.bounded and not self.meta.bounds(mip).contains_bbox(bbox):
        raise exceptions.OutOfBoundsError(f"Computed {bbox} is not contained within bounds {self.meta.bounds(mip)}")

    if parallel is None:
      parallel = self.parallel

    bbox = bbox.astype(np.int64)

    if bbox.subvoxel():
      raise exceptions.SubvoxelVolumeError(
        f"{bbox} (after adjusting for coord_resolution) has zero or near zero size."
      )

    tup = self.image.download(
      bbox, mip, 
      parallel=parallel, 
      renumber=bool(renumber),
      label=label,
    )
    
    if renumber:
      img, remap = tup
    else:
      remap = {}
      img = tup

    if segids is None:
      return tup

    mask_value = 0
    if preserve_zeros:
      mask_value = np.inf
      if np.issubdtype(self.dtype, np.integer):
        mask_value = np.iinfo(self.dtype).max

      segids.append(0)

    img = fastremap.mask_except(img, segids, in_place=True, value=mask_value)

    img = VolumeCutout.from_volume(
      self.meta, mip, img, bbox
    )
    if renumber:
      return img, remap
    else:
      return img

  def scattered_points(
    self, pts, 
    mip=None, coord_resolution=None
  ):
    """
    Download one or more single voxel values that may be scattered
    across the dataset. You can accelerate this query with an LRU
    if there is some spatial localization.

    pts: iterable of triples
    mip: which resolution level to get (default self.mip)
    coord_resolution: (rx,ry,rz) the coordinate resolution of the input point.
      Sometimes Neuroglancer is working in the resolution of another
      higher res layer and this can help correct that.

    Returns:     
      { (x,y,z): label, ... }
    """
    pts = list(pts)
    if isinstance(pts[0], int):
      pts = [ pts ]

    if mip is None:
      mip = self.mip
    mip = self.meta.to_mip(mip)

    if coord_resolution is not None:
      factor = self.meta.resolution(0) / Vec(*coord_resolution)
      pts = [ Vec(*pt) / factor for pt in pts ]

    pts = set([ tuple(self.point_to_mip(pt, mip=0, to_mip=mip)) for pt in pts ])
    return self.image.download_points(pts, mip)

  def download_point(
    self, pt, size=256, 
    mip=None, parallel=None, 
    coord_resolution=None,
    **kwargs
  ):
    """
    Download to the right of point given in mip 0 coords.
    Useful for quickly visualizing a neuroglancer coordinate
    at an arbitary mip level.

    pt: (x,y,z)
    size: int or (sx,sy,sz)
    mip: int representing resolution level
    parallel: number of processes to launch (0 means all cores)
    coord_resolution: (rx,ry,rz) the coordinate resolution of the input point.
      Sometimes Neuroglancer is working in the resolution of another
      higher res layer and this can help correct that.

    Return: image
    """
    if isinstance(size, int):
      size = Vec(size, size, size)
    else:
      size = Vec(*size)

    if mip is None:
      mip = self.mip

    mip = self.meta.to_mip(mip)
    size2 = size // 2

    if coord_resolution is not None:
      factor = self.meta.resolution(0) / Vec(*coord_resolution)
      pt = Vec(*pt) / factor

    pt = self.point_to_mip(pt, mip=0, to_mip=mip)
    bbox = Bbox(pt - size2, pt + size2).astype(np.int64)
    for i, sz in enumerate(size):
      if sz == 1:
        bbox.minpt[i] = pt[i]
        bbox.maxpt[i] = pt[i] + 1

    if self.autocrop:
      bbox = Bbox.intersection(bbox, self.meta.bounds(mip))

    bbox = bbox.astype(np.int32)
    if parallel is None:
      parallel = self.parallel

    return self.image.download(bbox, mip, parallel=parallel, **kwargs)

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
    if isinstance(slice, Bbox):
      slices = slices.convert_units(
        "vx", self.meta.resolution(self.mip)
      ).astype(int).to_slices()

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
    bbox = bbox.convert_units('vx', self.resolution)
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
    bbox = bbox.convert_units('vx', self.resolution)
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

  def to_dask(self, chunks=None, name=None):
    """Return a dask array for this volume.

    Parameters
    ----------
    chunks: tuple of ints or tuples of ints
      Passed to ``da.from_array``, allows setting the chunks on
      initialisation, if the chunking scheme in the stored dataset is not
      optimal for the calculations to follow. Note that the chunking should
      be compatible with an underlying 4d array.
    name: str, optional
      An optional keyname for the array. Defaults to hashing the input

    Returns
    -------
    Dask array
    """
    import dask.array as da
    from dask.base import tokenize

    if chunks is None:
      chunks = tuple(self.chunk_size) + (self.num_channels, )
    if name is None:
      name = 'to-dask-' + tokenize(self, chunks)
    return da.from_array(self, chunks, name=name)

  def __del__(self):
    pass
