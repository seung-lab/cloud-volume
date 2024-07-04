"""
The Precomputed format is a neuroscience imaging format 
designed for cloud storage. The specification is located
here:

https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed

This datasource contains the code for manipulating images.
"""
import copy
from typing import Dict, Tuple, Sequence, Union, Optional
from functools import reduce, partial
import itertools
import operator
import uuid

import numpy as np
from tqdm import tqdm

from cloudfiles import CloudFiles, compression

from cloudvolume import lib, exceptions
from cloudvolume.lru import LRU
from cloudvolume.scheduler import schedule_jobs, DEFAULT_THREADS
from ....types import CompressType, MipType
from .... import paths
from ....lib import Bbox, Vec, sip, first, BboxLikeType, toiter
from .... import sharedmemory, chunks

from ... import autocropfn, readonlyguard, ImageSourceInterface
from .. import sharding
from .common import chunknames, gridpoints, compressed_morton_code, morton_code_to_bbox
from . import tx, rx, xfer

class PrecomputedImageSource(ImageSourceInterface):
  def __init__(
    self, config, meta, cache,
    autocrop:bool = False, 
    bounded:bool = True,
    non_aligned_writes:bool = False,
    fill_missing:bool = False, 
    delete_black_uploads:bool = False,
    background_color:int = 0,
    readonly:bool = False,
    lru_bytes:int = 0,
  ):
    self.config = config
    self.meta = meta 
    self.cache = cache 

    self.autocrop = bool(autocrop)
    self.bounded = bool(bounded)
    self.fill_missing = bool(fill_missing)
    self.non_aligned_writes = bool(non_aligned_writes)
    self.readonly = bool(readonly)
    
    self.delete_black_uploads = bool(delete_black_uploads)
    self.background_color = background_color

    self.shared_memory_id = self.generate_shared_memory_location()

    self.lru = LRU(lru_bytes, size_in_bytes=True)

  def generate_shared_memory_location(self):
    return 'precomputed-shm-' + str(uuid.uuid4())

  def unlink_shared_memory(self):
    """Unlink the current shared memory location from the filesystem."""
    return sharedmemory.unlink(self.shared_memory_id)
    
  def grid_size(self, mip=None):
    mip = mip if mip is not None else self.config.mip
    return np.ceil(self.meta.volume_size(mip) / self.meta.chunk_size(mip)).astype(np.int64)

  def check_bounded(self, bbox, mip):
    bbox = bbox.convert_units('vx', self.meta.resolution(mip))
    if self.bounded and not self.meta.bounds(mip).contains_bbox(bbox):
      raise exceptions.OutOfBoundsError("""
        Requested cutout not contained within dataset bounds.

        Cloudpath: {}
        Requested: {}
        Bounds: {}
        Mip: {}
        Resolution: {}

        Set bounded=False to disable this warning.
      """.format(
          self.meta.cloudpath, 
          bbox, self.meta.bounds(mip), 
          mip, self.meta.resolution(mip)
        )
      )

  def has_data(self, mip=None):
    """
    Returns whether the specified mip appears to have data 
    by testing whether the "folder" exists.

    Returns: bool
  
    The mip is the index into the returned list. If
    the entry is True, then the data appears to be there.
    If the entry is False, then the data is not there.
    """
    mip = mip if mip is not None else self.config.mip
    mip = self.meta.to_mip(mip)
    
    cf = CloudFiles(self.meta.cloudpath, secrets=self.config.secrets)
    key = self.meta.key(mip)
    return first(cf.list(prefix=key)) is not None

  def download_points(
    self, points, mip:int
  ) -> Dict[Tuple[int,int,int], int]:
    """For accessing a list of individual voxels."""
    total = len(points) if hasattr(points, "__len__") else None
    points = toiter(points)
    bbxs = ( Bbox(pt, Vec(*pt)+1) for pt in points )

    res = {}
    def getpt(bbx):
      nonlocal mip
      value = self.download(bbx, mip)
      res[tuple(bbx.minpt)] = value[0][0][0][0]

    fns = ( partial(getpt, bbx) for bbx in bbxs )
    progress = self.config.progress
    if progress and not isinstance(progress, str):
      progress = "Downloading"

    concurrency = DEFAULT_THREADS
    if self.meta.path.protocol == "file":
      concurrency = 0

    schedule_jobs(
      fns, 
      concurrency=concurrency,
      progress=progress, 
      total=total, 
      green=self.config.green,
    )
    return res

  def download(
    self, bbox, mip, parallel=1, 
    location=None, retain=False,
    use_shared_memory=False, use_file=False,
    order='F', renumber=False, 
    label=None,
  ):
    """
    Download a cutout image from the dataset.

    bbox: a Bbox object describing what region to download
    mip: which resolution to fetch, 0 is the highest resolution
    parallel: how many processes to use for downloading 
    location: if using shared memory or downloading to a file,
      which file location should be used?
    retain: don't delete the shared memory file after download
      completes
    use_shared_memory: download to a shared memory location. 
      This enables efficient inter-process communication and
      efficient parallel operation. mutually exclusive with
      use_file.
    use_file: download image directly to a file named by location. 
      mutually exclusive with use_shared_memory. 
    order: The underlying shared memory or file buffer can use either
      C or Fortran order for storing a multidimensional array.
    renumber: dynamically rewrite downloaded segmentation into
      a more compact data type. Only compatible with single-process
      non-sharded download.
    label: If provided, downloads a binary image where the selected
      label is foreground. This can help reduce memory usage 1-byte
      per voxel instead of the volume's dtype (max: 8-bytes per voxel).

    Returns:
      if renumber:
        (4d ndarray, remap dict)
      else:
        4d ndarray
    """
    if isinstance(bbox, Bbox):
      bbox = bbox.convert_units('vx', self.meta.resolution(mip))

    if self.autocrop:
      bbox = Bbox.intersection(bbox, self.meta.bounds(mip))

    self.check_bounded(bbox, mip)

    if location is None:
      location = self.shared_memory_id

    if self.is_sharded(mip):
      if renumber:
        raise ValueError("renumber is only supported for non-sharded volumes.")

      scale = self.meta.scale(mip)
      spec = sharding.ShardingSpecification.from_dict(scale['sharding'])
      return rx.download_sharded(
        bbox, mip, 
        self.meta, self.cache, self.lru, spec,
        compress=self.config.compress,
        progress=self.config.progress,
        fill_missing=self.fill_missing,
        order=order,
        background_color=int(self.background_color),
        label=label,
      )
    else:
      return rx.download(
        bbox, mip, 
        meta=self.meta,
        cache=self.cache,
        lru=self.lru,
        parallel=parallel,
        location=location,
        retain=retain,
        use_shared_memory=use_shared_memory,
        use_file=use_file,
        fill_missing=self.fill_missing,
        progress=self.config.progress,
        compress=self.config.compress,
        order=order,
        green=self.config.green,
        secrets=self.config.secrets,
        renumber=renumber,
        background_color=int(self.background_color),
        label=label,
      )

  def download_files(
    self, bbox:BboxLikeType, mip:int, 
    decompress:bool = True, 
    parallel:int = 1, 
    cache_only:bool = False
  ):
    """
    Download the files that comprise a cutout image from the dataset
    without rendering them into an image. 

    bbox: a Bbox object describing what region to download
    mip: which resolution to fetch, 0 is the highest resolution
    parallel: how many processes to use for downloading 
    cache_only: write downloaded files to cache and discard
      the result to save memory

    Returns: 
      If sharded:
        { morton_code: binary }
      else:
        { path: binary }
    """
    if isinstance(bbox, Bbox):
      bbox = bbox.convert_units('vx', self.meta.resolution(mip))

    bbox = Bbox.create(bbox, context=self.meta.bounds(mip))

    if self.autocrop:
      bbox = Bbox.intersection(bbox, self.meta.bounds(mip))

    self.check_bounded(bbox, mip)

    if self.is_sharded(mip):
      scale = self.meta.scale(mip)
      spec = sharding.ShardingSpecification.from_dict(scale['sharding'])
      return rx.download_raw_sharded(
        bbox, mip, 
        self.meta, self.cache, spec,
        decompress=decompress,
        progress=self.config.progress,
      )
    else:
      return rx.download_raw_unsharded(
        bbox, mip,
        meta=self.meta,
        cache=self.cache,
        decompress=decompress,
        progress=self.config.progress,
        parallel=parallel, 
        green=self.config.green,
        secrets=self.config.secrets,
        fill_missing=self.fill_missing,
        compress_type=self.config.compress,
        background_color=int(self.background_color),
        cache_only=cache_only,
      )

  def unique(self, bbox:BboxLikeType, mip:int) -> set:
    """Extract unique values in an efficient way."""
    if isinstance(bbox, Bbox):
      bbox = bbox.convert_units('vx', self.meta.resolution(mip))

    bbox = Bbox.create(bbox, context=self.meta.bounds(mip))
    
    if self.autocrop:
      bbox = Bbox.intersection(bbox, self.meta.bounds(mip))

    self.check_bounded(bbox, mip)

    if self.is_sharded(mip):
      scale = self.meta.scale(mip)
      spec = sharding.ShardingSpecification.from_dict(scale['sharding'])
      return rx.unique_sharded(
        bbox, mip, 
        self.meta, self.cache, self.lru, spec,
        compress=self.config.compress,
        progress=self.config.progress,
        fill_missing=self.fill_missing,
        background_color=int(self.background_color),
      )
    else:
      return rx.unique_unsharded(
        bbox, mip, 
        meta=self.meta,
        cache=self.cache,
        lru=self.lru,
        parallel=1,
        fill_missing=self.fill_missing,
        progress=self.config.progress,
        compress=self.config.compress,
        green=self.config.green,
        secrets=self.config.secrets,
        background_color=int(self.background_color),
      )

  @readonlyguard
  def upload(
      self, 
      image, offset, mip, 
      parallel=1,
      location=None, location_bbox=None, order='F',
      use_shared_memory=False, use_file=False      
    ):

    if mip in self.meta.locked_mips():
      raise exceptions.ReadOnlyException(
        "MIP {} is currently write locked. If this should not be the case, run vol.meta.unlock_mips({}).".format(
          mip, mip
        )
      )

    offset = Vec(*offset)
    bbox = Bbox( offset, offset + Vec(*image.shape[:3]) )

    self.check_bounded(bbox, mip)

    if self.autocrop:
      image, bbox = autocropfn(self.meta, image, bbox, mip)
      offset = bbox.minpt

    if location is None:
      location = self.shared_memory_id

    if self.is_sharded(mip):
      return self._upload_shard(image, bbox, mip)

    return tx.upload(
      self.meta, self.cache, self.lru,
      image, offset, mip, 
      compress=self.config.compress,
      compress_level=self.config.compress_level,
      cdn_cache=self.config.cdn_cache,
      parallel=parallel, 
      progress=self.config.progress,
      location=location, 
      location_bbox=location_bbox,
      location_order=order,
      use_shared_memory=use_shared_memory,
      use_file=use_file,
      delete_black_uploads=self.delete_black_uploads,
      background_color=self.background_color,
      non_aligned_writes=self.non_aligned_writes,
      secrets=self.config.secrets,
      green=self.config.green,
      fill_missing=self.fill_missing, # applies only to unaligned writes
    )

  def _upload_shard(self, image, bbox, mip):
    basepath = self.meta.join(self.meta.cloudpath, self.meta.key(mip))
    cf = CloudFiles(basepath, progress=self.config.progress, secrets=self.config.secrets)

    def do_upload():
      (filename, shard) = self.make_shard(image, bbox, mip)
      cf.put(
        filename, shard, 
        compress=self.config.compress, 
        cache_control=self.config.cdn_cache
      )

    def do_delete():
      nonlocal mip
      shard_filename = self.shard_filename(bbox, mip)
      cf.delete(shard_filename)

    testfn = lambda image: np.any(image)
    if self.background_color != 0:
      testfn = lambda image: np.any(image != self.background_color)

    if self.delete_black_uploads:
      if testfn(image):
        do_upload()
      else:
        do_delete()
    else:
      do_upload()

  def exists(self, bbox, mip=None):
    if mip is None:
      mip = self.config.mip

    if isinstance(bbox, Bbox):
      bbox = bbox.convert_units('vx', self.meta.resolution(mip))

    bbox = Bbox.create(bbox, self.meta.bounds(mip), bounded=True)
    realized_bbox = bbox.expand_to_chunk_size(
      self.meta.chunk_size(mip), offset=self.meta.voxel_offset(mip)
    )
    realized_bbox = Bbox.clamp(realized_bbox, self.meta.bounds(mip))

    cloudpaths = chunknames(
      realized_bbox, self.meta.bounds(mip), 
      self.meta.key(mip), self.meta.chunk_size(mip),
      protocol=self.meta.path.protocol
    )

    exists = CloudFiles(
      self.meta.cloudpath, progress=self.config.progress,
      secrets=self.config.secrets
    ).exists(cloudpaths)

    if len(self.lru):
      for k,v in exists.items():
        if v == False:
          self.lru.pop(k, None)

    return exists

  @readonlyguard
  def delete(self, bbox, mip=None):
    if mip is None:
      mip = self.config.mip

    if mip in self.meta.locked_mips():
      raise exceptions.ReadOnlyException(
        "MIP {} is currently write locked. If this should not be the case, run vol.meta.unlock_mips({}).".format(
          mip, mip
        )
      )

    if isinstance(bbox, Bbox):
      bbox = bbox.convert_units('vx', self.meta.resolution(mip))

    bbox = Bbox.create(bbox, self.meta.bounds(mip), bounded=True)
    realized_bbox = bbox.expand_to_chunk_size(
      self.meta.chunk_size(mip), offset=self.meta.voxel_offset(mip)
    )
    realized_bbox = Bbox.clamp(realized_bbox, self.meta.bounds(mip))

    if bbox != realized_bbox:
      raise exceptions.AlignmentError(
        "Unable to delete non-chunk aligned bounding boxes. Requested: {}, Realized: {}".format(
        bbox, realized_bbox
      ))

    cloudpaths = lambda: chunknames(
      realized_bbox, self.meta.bounds(mip),
      self.meta.key(mip), self.meta.chunk_size(mip),
      protocol=self.meta.path.protocol
    ) # need to regenerate so that generator isn't used up

    CloudFiles(self.meta.cloudpath, progress=self.config.progress, secrets=self.config.secrets) \
      .delete(cloudpaths())

    if len(self.lru) > 0:
      for path in cloudpaths():
        self.lru.pop(path, None)

    if self.cache.enabled:
      CloudFiles('file://' + self.cache.path, progress=self.config.progress, secrets=self.config.secrets) \
        .delete(cloudpaths())

  def memory_cutout(
    self, 
    bbox:BboxLikeType, 
    mip:int,
    encoding:Optional[str] = None, 
    compress:CompressType = None, 
    compress_level:Optional[int] = None,
  ):
    """
    Create a disposable in-memory CloudVolume (mem://) containing
    the requested cutout region in the unsharded precomputed
    format. The source volume may be sharded or unsharded.

    You can specify an alternative encoding and compression 
    settings for the new volume.
    """
    if isinstance(bbox, Bbox):
      bbox = bbox.convert_units('vx', self.meta.resolution(mip))

    realized_bbox = bbox.expand_to_chunk_size(
      self.meta.chunk_size(mip), offset=self.meta.voxel_offset(mip)
    )
    realized_bbox = Bbox.clamp(realized_bbox, self.meta.bounds(mip))

    cv = self.transfer_to(
      cloudpath=f"mem://{str(uuid.uuid4())}",
      bbox=realized_bbox, 
      mip=mip, 
      compress=compress, 
      compress_level=compress_level,
      encoding=encoding,
      sharded=False,
    )

    delfn = cv.__del__
    def cleanup(self):
      nonlocal delfn
      self.image.delete_all()
      delfn(self)

    cv.__del__ = cleanup

    return cv

  def delete_all(self, mip):
    cf = CloudFiles(self.meta.join(self.meta.cloudpath, self.meta.key(mip)))
    cf.delete(cf.list())

  def transfer_to(
    self,
    cloudpath:str, 
    bbox:BboxLikeType, 
    mip:MipType, 
    block_size:Optional[int] = None, 
    compress:CompressType = True, 
    compress_level:Optional[int] = None, 
    encoding:Optional[str] = None,
    sharded:Optional[bool] = None,
  ):
    if isinstance(bbox, Bbox):
      bbox = bbox.convert_units('vx', self.meta.resolution(mip))

    if sharded is None:
      sharded = self.is_sharded(mip)

    pth = paths.extract(cloudpath)
    if self.meta.path.format != pth.format:
      return xfer.transfer_by_rerendering(
        self, cloudpath,
        bbox=bbox,
        mip=mip,
        compress=compress,
        compress_level=compress_level,
        encoding=encoding,
      )
    elif not sharded and self.is_sharded(mip):
      return xfer.transfer_any_to_unsharded(
        self, cloudpath,
        bbox=bbox,
        mip=mip,
        compress=compress,
        compress_level=compress_level,
        encoding=encoding,
      )
    elif sharded and self.is_sharded(mip):
      return xfer.transfer_sharded_to_sharded(
        self, cloudpath,
        bbox=bbox,
        mip=mip, 
        block_size=block_size,
        compress=compress, 
        compress_level=compress_level, 
        encoding=encoding,
      )
    elif not sharded and not self.is_sharded(mip):
      return xfer.transfer_unsharded_to_unsharded(
        self, cloudpath,
        bbox=bbox,
        mip=mip,
        block_size=block_size,
        compress=compress,
        compress_level=compress_level, 
        encoding=encoding,
      )
    else:
      raise ValueError(
        "Unsharded to sharded is not implemented in CloudVolume. "
        "Try Igneous! https://github.com/seung-lab/igneous"
      )

  def shard_reader(self, mip=None):
    mip = mip if mip is not None else self.config.mip
    scale = self.meta.scale(mip)
    spec = sharding.ShardingSpecification.from_dict(scale['sharding'])
    return sharding.ShardReader(self.meta, self.cache, spec)

  def shard_spec(self, mip, spec=None):
    if spec is None:
      scale = self.meta.scale(mip)
      if 'sharding' in scale:
        spec = sharding.ShardingSpecification.from_dict(scale['sharding'])
      else:
        raise ValueError("mip {} does not have a sharding specification.".format(mip))
    return spec

  def morton_codes(
    self, bbox, mip=None, spec=None,
    same_shard=True, require_aligned=True
  ):
    mip = mip if mip is not None else self.config.mip
    scale = self.meta.scale(mip)
    spec = self.shard_spec(mip, spec)

    if isinstance(bbox, Bbox):
      bbox = bbox.convert_units('vx', self.meta.resolution(mip))

    bbox = Bbox.create(bbox)
    if bbox.subvoxel():
      raise ValueError("Bounding box is too small to make a shard. Got: {}".format(bbox))

    # Alignment Checks:
    # 1. Aligned to atomic chunks - required for grid point generation
    aligned_bbox = bbox.expand_to_chunk_size(
      self.meta.chunk_size(mip), offset=self.meta.voxel_offset(mip)
    )
    if require_aligned and bbox != aligned_bbox:
      raise exceptions.AlignmentError(
        "Unable to create shard from a non-chunk aligned bounding box. Requested: {}, Aligned: {}".format(
        bbox, aligned_bbox
      ))

    # 2. Covers the dataset at least partially
    aligned_bbox = Bbox.clamp(aligned_bbox, self.meta.bounds(mip))
    if aligned_bbox.subvoxel():
      raise exceptions.OutOfBoundsError("Shard completely outside dataset: Requested: {}, Dataset: {}".format(
        bbox, self.meta.bounds(mip)
      ))

    grid_size = self.grid_size(mip)
    chunk_size = self.meta.chunk_size(mip)
    reader = sharding.ShardReader(self.meta, self.cache, spec)

    # 3. Gridpoints all within this one shard
    gpts = list(gridpoints(aligned_bbox, self.meta.bounds(mip), chunk_size))
    morton_codes = compressed_morton_code(gpts, grid_size)

    if same_shard:
      all_same_shard = bool(reduce(lambda a,b: operator.eq(a,b) and a,
        map(reader.get_filename, morton_codes)
      ))

      if not all_same_shard:
        raise exceptions.AlignmentError(
          "The gridpoints for this image did not all correspond to the same shard. Got: {}".format(bbox)
        )

    return gpts, morton_codes

  def shard_filename(self, bbox:BboxLikeType, mip:int) -> str:
    mip = mip if mip is not None else self.config.mip
    if not self.is_sharded(mip):
      raise ValueError("Unable to compute filename for unsharded image.")

    spec = self.shard_spec(mip)
    gpts, morton_codes = self.morton_codes(bbox, mip=mip, spec=spec)
    reader = self.shard_reader()
    return reader.get_filename(first(morton_codes))

  def make_shard(self, img, bbox, mip=None, spec=None, progress=False):
    """
    Convert an image that represents a single complete shard 
    into a shard file.
  
    img: a volumetric numpy array image
    bbox: the bbox it represents in voxel coordinates
    mip: if specified, use the sharding specification from 
      this mip level, otherwise use the sharding spec from
      the current implicit mip level in config.
    spec: use the provided specification (overrides mip parameter)

    Returns: (filename, shard_file)
    """
    mip = mip if mip is not None else self.config.mip
    
    if isinstance(bbox, Bbox):
      bbox = bbox.convert_units('vx', self.meta.resolution(mip))

    spec = self.shard_spec(mip, spec)
    gpts, morton_codes = self.morton_codes(bbox, mip=mip, spec=spec)
    chunk_size = self.meta.chunk_size(mip)

    testfn = lambda image: np.any(image)
    if self.background_color != 0:
      testfn = lambda image: np.any(image != self.background_color)

    labels = {}
    pt_anchor = gpts[0] * chunk_size
    for pt_abs, morton_code in zip(gpts, morton_codes):
      cutout_bbx = Bbox(pt_abs * chunk_size, (pt_abs + 1) * chunk_size)

      # Neuroglancer expects border chunks not to extend beyond dataset bounds
      cutout_bbx.maxpt = cutout_bbx.maxpt.clip(None, self.meta.volume_size(mip))
      cutout_bbx -= pt_anchor

      chunk = img[ cutout_bbx.to_slices() ]

      if (not self.delete_black_uploads) or testfn(chunk):
        labels[morton_code] = chunks.encode(
          chunk, self.meta.encoding(mip),
          block_size=self.meta.compressed_segmentation_block_size(mip),
          compression_params=self.meta.compression_params(mip),
        )

    reader = self.shard_reader(mip=mip)
    shard_filename = reader.get_filename(first(labels.keys()))

    return (shard_filename, spec.synthesize_shard(labels, progress=progress))

  def to_sharded(
    self,
    uncompressed_shard_bytesize:int = int(3.5e9),
    max_shard_index_bytes:int = 8192, # 2^13
    max_minishard_index_bytes:int = 40000,
    max_labels_per_minishard:int = 4000,
    minishard_index_encoding:str = "gzip",
    data_encoding:str = "gzip",
    mip:Optional[int] = None,
  ):
    mip = mip if mip is not None else self.config.mip

    spec = sharding.compute_shard_params_for_image(
      dataset_size=self.meta.volume_size(mip),
      chunk_size=self.meta.chunk_size(mip),
      encoding=self.meta.encoding(mip),
      dtype=self.meta.dtype,
      uncompressed_shard_bytesize=uncompressed_shard_bytesize, 
      max_shard_index_bytes=max_shard_index_bytes,
      max_minishard_index_bytes=max_minishard_index_bytes,
      max_labels_per_minishard=max_labels_per_minishard,
      minishard_index_encoding=minishard_index_encoding,
      data_encoding=data_encoding,
    )
    self.meta.scale(mip)["sharding"] = spec.to_dict()

  def to_unsharded(self, mip=None):
    mip = mip if mip is not None else self.config.mip
    self.meta.scale(mip).pop("sharding", None)

  def shard_shape(self, mip=None):
    mip = mip if mip is not None else self.config.mip

    if not self.is_sharded(mip):
      raise ValueError("This volume is not sharded.")

    return sharding.image_shard_shape_from_spec(
      self.meta.scale(mip)["sharding"],
      self.meta.volume_size(mip),
      self.meta.chunk_size(mip),
    )

  def is_sharded(self, mip):
    scale = self.meta.scale(mip)
    return 'sharding' in scale and scale['sharding'] is not None

