"""
The Precomputed format is a neuroscience imaging format 
designed for cloud storage. The specification is located
here:

https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed

This datasource contains the code for manipulating images.
"""
from cloudvolume import lib, exceptions
from ...lib import Bbox, Vec

from . import tx, rx

import uuid

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
          bbox, self.meta.bounds, 
          mip, self.meta.resolution(mip)
        )
      )

  def download(
      self, bbox, mip, parallel=1, 
      location=None, retain=False,
      use_shared_memory=False, use_file=False
    ):

    self.check_bounded(bbox, mip)

    if self.autocrop:
      bbox = Bbox.intersection(bbox, self.meta.bounds)

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
    )

  def upload(
      self, 
      image, offset, mip, 
      parallel=1,
      location=None, use_shared_memory=False, use_file=False,
    ):

    offset = Vec(*offset)
    bbox = Bbox( offset, offset + Vec(*image.shape[:3]) )

    self.check_bounded(bbox, mip)

    if self.autocrop:
      img_bbox = Bbox.intersection(bbox, self.meta.bounds)
      img_bbox -= (img_bbox.minpt - bbox.minpt)
      image = image[ img_bbox.to_slices() ]
      bbox = Bbox.intersection(bbox, self.meta.bounds)
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
      use_shared_memory=use_shared_memory,
      use_file=use_file,
      delete_black_uploads=self.delete_black_uploads,
      non_aligned_writes=self.non_aligned_writes,
    )

