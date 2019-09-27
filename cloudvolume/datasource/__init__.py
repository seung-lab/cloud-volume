from .. import exceptions
from ..lib import yellow, Bbox, Vec

import numpy as np

NON_ALIGNED_WRITE = yellow(
  """
  Non-Aligned writes are disabled by default. There are several good reasons 
  not to use them. 

  1) Memory and Network Inefficiency
    Producing non-aligned writes requires downloading the chunks that overlap
    with the write area but are not wholly contained. They are then painted
    in the overlap region and added to the upload queue. This requires
    the outer shell of the aligned image to be downloaded painted, and uploaded. 
  2) Race Conditions
    If you are using multiple instances of CloudVolume, the partially overlapped 
    chunks will be downloaded, partially painted, and uploaded. If this procedure
    occurs close in time between two processes, the final chunk may be partially
    painted. If you are sure only one CloudVolume instance will be accessing an
    area at any one time, this is not a problem.

  If after reading this you are still sure you want non-aligned writes, you can
  set non_aligned_writes=True.

  Alignment Check: 
    Mip:             {mip}
    Chunk Size:      {chunk_size}
    Volume Offset:   {offset}
    Received:        {got} 
    Nearest Aligned: {check}
""")

class ImageSourceInterface(object):
  def download(self, bbox, mip):
    raise NotImplementedError()
  def upload(self, image, offset, mip):
    raise NotImplementedError()
  def exists(self, bbox, mip):
    raise NotImplementedError()
  def delete(self, bbox, mip):
    raise NotImplementedError()
  def transfer_to(self, cloudpath, bbox, mip):
    raise NotImplementedError()

def readonlyguard(fn):
  def guardfn(self, *args, **kwargs):
    if self.readonly:
      raise exceptions.ReadOnlyException(self.meta.cloudpath)
    return fn(self, *args, **kwargs)
  return guardfn

def check_grid_aligned(meta, img, bounds, mip, throw_error=False):
  """Raise a cloudvolume.exceptions.AlignmentError if the provided image is not grid aligned."""
  shape = Vec(*img.shape)[:3]
  alignment_check = bounds.expand_to_chunk_size(meta.chunk_size(mip), meta.voxel_offset(mip))
  alignment_check = Bbox.clamp(alignment_check, meta.bounds(mip))
  is_aligned = np.all(alignment_check.minpt == bounds.minpt) and np.all(alignment_check.maxpt == bounds.maxpt)
  
  if throw_error and is_aligned == False:
    msg = NON_ALIGNED_WRITE.format(
      mip=mip, chunk_size=meta.chunk_size(mip), 
      offset=meta.voxel_offset(mip), 
      got=bounds, check=alignment_check
    )
    raise exceptions.AlignmentError(msg)

  return is_aligned

def autocropfn(meta, image, bbox, mip):
  cropped_bbox = Bbox.intersection(bbox, meta.bounds(mip))
  dmin = np.abs(bbox.minpt - cropped_bbox.minpt)
  img_bbox = Bbox(dmin, dmin + cropped_bbox.size())
  image = image[ img_bbox.to_slices() ]

  return image, cropped_bbox


