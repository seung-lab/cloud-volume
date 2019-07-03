"""
The Precomputed format is a neuroscience imaging format 
designed for cloud storage. The specification is located
here:

https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed

This datasource contains the code for manipulating images.
"""
import itertools
import uuid

import numpy as np
from tqdm import tqdm

from cloudvolume import lib, exceptions
from ...lib import Bbox, Vec
from ... import sharedmemory
from ...storage import Storage

from .. import autocropfn
from .common import chunknames
from . import tx, rx

class PrecomputedImageSource(object):
  def __init__(
    self, config, meta, cache,
    autocrop=False, bounded=True,
    non_aligned_writes=False,
    fill_missing=False, 
    delete_black_uploads=False
  ):
    self.config = config
    self.meta = meta 
    self.cache = cache 

    self.autocrop = bool(autocrop)
    self.bounded = bool(bounded)
    self.fill_missing = bool(fill_missing)
    self.non_aligned_writes = bool(non_aligned_writes)
    self.delete_black_uploads = bool(delete_black_uploads)

    self.shared_memory_id = self.generate_shared_memory_location()

  def generate_shared_memory_location(self):
    return 'precomputed-shm-' + str(uuid.uuid4())

  def unlink_shared_memory(self):
    """Unlink the current shared memory location from the filesystem."""
    return sharedmemory.unlink(self.shared_memory_id)

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

  def download(
      self, bbox, mip, parallel=1, 
      location=None, retain=False,
      use_shared_memory=False, use_file=False,
      order='F'
    ):

    self.check_bounded(bbox, mip)

    if self.autocrop:
      bbox = Bbox.intersection(bbox, self.meta.bounds(mip))

    if location is None:
      location = self.shared_memory_id

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
    )

  def upload(
      self, 
      image, offset, mip, 
      parallel=1,
      location=None, location_bbox=None, order='F',
      use_shared_memory=False, use_file=False      
    ):

    offset = Vec(*offset)
    bbox = Bbox( offset, offset + Vec(*image.shape[:3]) )

    self.check_bounded(bbox, mip)

    if self.autocrop:
      image, bbox = autocropfn(self.meta, image, bbox, mip)
      offset = bbox.minpt

    if location is None:
      location = self.shared_memory_id

    return tx.upload(
      self.meta, self.cache,
      image, offset, mip,
      compress=self.config.compress,
      cdn_cache=self.config.cdn_cache,
      parallel=parallel, 
      progress=self.config.progress,
      location=location, 
      location_bbox=location_bbox,
      location_order=order,
      use_shared_memory=use_shared_memory,
      use_file=use_file,
      delete_black_uploads=self.delete_black_uploads,
      non_aligned_writes=self.non_aligned_writes,
      green=self.config.green,
    )

  def exists(self, bbox, mip=None):
    if mip is None:
      mip = self.config.mip

    bbox = Bbox.create(bbox, self.meta.bounds(mip), bounded=True)
    realized_bbox = bbox.expand_to_chunk_size(
      self.meta.chunk_size(mip), offset=self.meta.voxel_offset(mip)
    )
    realized_bbox = Bbox.clamp(realized_bbox, self.meta.bounds(mip))

    cloudpaths = list(chunknames(
      realized_bbox, self.meta.bounds(mip), 
      self.meta.key(mip), self.meta.chunk_size(mip),
      protocol=self.meta.path.protocol
    ))

    with Storage(self.meta.cloudpath, progress=self.config.progress) as storage:
      existence_report = storage.files_exist(cloudpaths)
    return existence_report    

  def delete(self, bbox, mip=None):
    if mip is None:
      mip = self.config.mip

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

    cloudpaths = list(chunknames(
      realized_bbox, self.meta.bounds(mip), 
      self.meta.key(mip), self.meta.chunk_size(mip),
      protocol=self.meta.path.protocol
    ))

    with Storage(self.meta.cloudpath, progress=self.config.progress) as storage:
      storage.delete_files(cloudpaths)

    if self.cache.enabled:
      with Storage('file://' + self.cache.path, progress=self.config.progress) as storage:
        storage.delete_files(cloudpaths)


  def transfer_to(self, cloudpath, bbox, mip, block_size=None, compress=True):
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

    with pbar:
      with Storage(self.meta.cloudpath) as src_stor:
        with Storage(cloudpath) as dest_stor:
          for _ in range(num_blocks, 0, -1):
            srcpaths = list(itertools.islice(cloudpaths, step))
            files = src_stor.get_files(srcpaths)
            files = [ (f['filename'], f['content']) for f in files ]
            dest_stor.put_files(
              files=files, 
              compress=compress, 
              content_type=tx.content_type(destvol),
            )
            pbar.update()

