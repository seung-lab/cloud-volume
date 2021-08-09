"""
The Precomputed format is a neuroscience imaging format 
designed for cloud storage. The specification is located
here:

https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed

This datasource contains the code for manipulating images.
"""
from functools import reduce
import itertools
import operator
import uuid

import numpy as np
from tqdm import tqdm

from cloudfiles import CloudFiles, compression

from cloudvolume import lib, exceptions
from ....lib import Bbox, Vec, sip, first
from .... import sharedmemory, chunks

from ... import autocropfn, readonlyguard, ImageSourceInterface
from .. import sharding
from .common import chunknames, gridpoints, compressed_morton_code
from . import tx, rx

class PrecomputedImageSource(ImageSourceInterface):
  def __init__(
    self, config, meta, cache,
    autocrop=False, bounded=True,
    non_aligned_writes=False,
    fill_missing=False, 
    delete_black_uploads=False,
    background_color=0,
    readonly=False,
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

  def generate_shared_memory_location(self):
    return 'precomputed-shm-' + str(uuid.uuid4())

  def unlink_shared_memory(self):
    """Unlink the current shared memory location from the filesystem."""
    return sharedmemory.unlink(self.shared_memory_id)

  def grid_size(self, mip=None):
    mip = mip if mip is not None else self.config.mip
    return np.ceil(self.meta.volume_size(mip) / self.meta.chunk_size(mip)).astype(np.int64)

  def check_bounded(self, bbox, mip):
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

  def download(
      self, bbox, mip, parallel=1, 
      location=None, retain=False,
      use_shared_memory=False, use_file=False,
      order='F', renumber=False
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

    Returns:
      if renumber:
        (4d ndarray, remap dict)
      else:
        4d ndarray
    """

    if self.autocrop:
      bbox = Bbox.intersection(bbox, self.meta.bounds(mip))

    self.check_bounded(bbox, mip)

    if location is None:
      location = self.shared_memory_id

    scale = self.meta.scale(mip)
    if 'sharding' in scale:
      if renumber:
        raise ValueError("renumber is only supported for non-shared volumes.")

      spec = sharding.ShardingSpecification.from_dict(scale['sharding'])
      return rx.download_sharded(
        bbox, mip, 
        self.meta, self.cache, spec,
        compress=self.config.compress,
        progress=self.config.progress,
        fill_missing=self.fill_missing,
        order=order,
        background_color=int(self.background_color),
      )
    else:
      return rx.download(
        bbox, mip, 
        meta=self.meta,
        cache=self.cache,
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
      (filename, shard) = self.make_shard(image, bbox, mip)
      basepath = self.meta.join(self.meta.cloudpath, self.meta.key(mip))
      CloudFiles(basepath, progress=self.config.progress, secrets=self.config.secrets).put(
        filename, shard, 
        compress=self.config.compress, 
        cache_control=self.config.cdn_cache
      )
      return

    return tx.upload(
      self.meta, self.cache,
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
      green=self.config.green,
      fill_missing=self.fill_missing, # applies only to unaligned writes
    )

  def exists(self, bbox, mip=None):
    if mip is None:
      mip = self.config.mip

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

    return CloudFiles(
      self.meta.cloudpath, progress=self.config.progress,
      secrets=self.config.secrets
    ).exists(cloudpaths)

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

    if self.cache.enabled:
      CloudFiles('file://' + self.cache.path, progress=self.config.progress, secrets=self.config.secrets) \
        .delete(cloudpaths())

  def transfer_to(self, cloudpath, bbox, mip, block_size=None, compress=True, compress_level=None):
    """
    Transfer files from one storage location to another, bypassing
    volume painting. This enables using a single CloudVolume instance
    to transfer big volumes. In some cases, gsutil or aws s3 cli tools
    may be more appropriate. This method is provided for convenience. It
    may be optimized for better performance over time as demand requires.

    cloudpath (str): path to storage layer
    bbox (Bbox object): ROI to transfer
    mip (int): resolution level
    block_size (int): number of file chunks to transfer per I/O batch.
    compress (bool): Set to False to upload as uncompressed
    """
    from cloudvolume import CloudVolume

    if mip is None:
      mip = self.config.mip

    if self.is_sharded(mip):
      raise exceptions.UnsupportedFormatError(f"Sharded sources are not supported. got: {self.meta.cloudpath}")

    bbox = Bbox.create(bbox, self.meta.bounds(mip))
    realized_bbox = bbox.expand_to_chunk_size(
      self.meta.chunk_size(mip), offset=self.meta.voxel_offset(mip)
    )
    realized_bbox = Bbox.clamp(realized_bbox, self.meta.bounds(mip))

    if bbox != realized_bbox:
      raise exceptions.AlignmentError(
        "Unable to transfer non-chunk aligned bounding boxes. Requested: {}, Realized: {}".format(
          bbox, realized_bbox
        ))

    default_block_size_MB = 50 # MB
    chunk_MB = self.meta.chunk_size(mip).rectVolume() * np.dtype(self.meta.dtype).itemsize * self.meta.num_channels
    if self.meta.layer_type == 'image':
      # kind of an average guess for some EM datasets, have seen up to 1.9x and as low as 1.1
      # affinites are also images, but have very different compression ratios. e.g. 3x for kempressed
      chunk_MB /= 1.3 
    else: # segmentation
      chunk_MB /= 100.0 # compression ratios between 80 and 800....
    chunk_MB /= 1024.0 * 1024.0

    if block_size:
      step = block_size
    else:
      step = int(default_block_size_MB // chunk_MB) + 1

    try:
      destvol = CloudVolume(cloudpath, mip=mip)
    except exceptions.InfoUnavailableError: 
      destvol = CloudVolume(cloudpath, mip=mip, info=self.meta.info, provenance=self.meta.provenance.serialize())
      destvol.commit_info()
      destvol.commit_provenance()
    except exceptions.ScaleUnavailableError:
      destvol = CloudVolume(cloudpath)
      for i in range(len(destvol.scales) + 1, len(self.meta.scales)):
        destvol.scales.append(
          self.meta.scales[i]
        )
      destvol.commit_info()
      destvol.commit_provenance()

    if destvol.image.is_sharded(mip):
      raise exceptions.UnsupportedFormatError(f"Sharded destinations are not supported. got: {destvol.cloudpath}")

    num_blocks = np.ceil(self.meta.bounds(mip).volume() / self.meta.chunk_size(mip).rectVolume()) / step
    num_blocks = int(np.ceil(num_blocks))

    cloudpaths = chunknames(
      bbox, self.meta.bounds(mip), 
      self.meta.key(mip), self.meta.chunk_size(mip),
      protocol=self.meta.path.protocol
    )

    pbar = tqdm(
      desc='Transferring Blocks of {} Chunks'.format(step), 
      unit='blocks', 
      disable=(not self.config.progress),
      total=num_blocks,
    )

    cfsrc = CloudFiles(self.meta.cloudpath, secrets=self.config.secrets)
    cfdest = CloudFiles(cloudpath)

    def check(files):
      errors = [
        file for file in files if \
        (file['content'] is None or file['error'] is not None)
      ]
      if errors:
        error_paths = [ f['path'] for f in errors ]
        raise exceptions.EmptyFileException("{} were empty or had IO errors.".format(", ".join(error_paths)))
      return files

    with pbar:
      for srcpaths in sip(cloudpaths, step):
        files = check(cfsrc.get(srcpaths, raw=True))
        cfdest.puts(
          compression.transcode(files, encoding=compress, level=compress_level), 
          compress=compress,
          content_type=tx.content_type(destvol),
          raw=True
        )
        pbar.update()

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
    scale = self.meta.scale(mip)

    if spec is None:
      if 'sharding' in scale:
        spec = sharding.ShardingSpecification.from_dict(scale['sharding'])
      else:
        raise ValueError("mip {} does not have a sharding specification.".format(mip))

    bbox = Bbox.create(bbox)
    if bbox.subvoxel():
      raise ValueError("Bounding box is too small to make a shard. Got: {}".format(bbox))

    # Alignment Checks:
    # 1. Aligned to atomic chunks - required for grid point generation
    aligned_bbox = bbox.expand_to_chunk_size(
      self.meta.chunk_size(mip), offset=self.meta.voxel_offset(mip)
    )
    if bbox != aligned_bbox:
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
    all_same_shard = bool(reduce(lambda a,b: operator.eq(a,b) and a,
      map(reader.get_filename, morton_codes)
    ))

    if not all_same_shard:
      raise exceptions.AlignmentError(
        "The gridpoints for this image did not all correspond to the same shard. Got: {}".format(bbox)
      )

    labels = {}
    pt_anchor = gpts[0] * chunk_size
    for pt_abs, morton_code in zip(gpts, morton_codes):
      cutout_bbx = Bbox(pt_abs * chunk_size, (pt_abs + 1) * chunk_size)

      # Neuroglancer expects border chunks not to extend beyond dataset bounds
      cutout_bbx.maxpt = cutout_bbx.maxpt.clip(None, self.meta.volume_size(mip))
      cutout_bbx -= pt_anchor

      chunk = img[ cutout_bbx.to_slices() ]
      labels[morton_code] = chunks.encode(chunk, self.meta.encoding(mip))

    shard_filename = reader.get_filename(first(labels.keys()))

    return (shard_filename, spec.synthesize_shard(labels, progress=progress))

  def is_sharded(self, mip):
    scale = self.meta.scale(mip)
    return 'sharding' in scale and scale['sharding'] is not None

