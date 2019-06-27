from __future__ import print_function

from intern.remote.boss import BossRemote
from intern.resource.boss.resource import (
  ChannelResource, ExperimentResource, CoordinateFrameResource
)

from .. import autocropfn

from ... import exceptions 
from ...lib import ( 
  colorize, red, mkdir, Vec, Bbox,  
  jsonify, generate_random_string
)
from ...secrets import boss_credentials
from ...volumecutout import VolumeCutout

class BossImageSource(object):
  def __init__(
    self, config, meta, cache,
    autocrop=False, bounded=True,
    non_aligned_writes=False,
    delete_black_uploads=False
  ):
    self.config = config
    self.meta = meta 
    self.cache = cache 

    self.autocrop = bool(autocrop)
    self.bounded = bool(bounded)
    self.non_aligned_writes = bool(non_aligned_writes)

  def download(self, bbox, mip):
    bounds = Bbox.clamp(bbox, self.meta.bounds(mip))

    if self.autocrop:
      image, bounds = autocropfn(self.meta, image, bounds, mip)
    
    if bounds.subvoxel():
      raise exceptions.EmptyRequestException('Requested less than one pixel of volume. {}'.format(bounds))

    x_rng = [ bounds.minpt.x, bounds.maxpt.x ]
    y_rng = [ bounds.minpt.y, bounds.maxpt.y ]
    z_rng = [ bounds.minpt.z, bounds.maxpt.z ]

    layer_type = 'image' if self.meta.layer_type == 'unknown' else self.meta.layer_type

    chan = ChannelResource(
      collection_name=self.meta.path.bucket, 
      experiment_name=self.meta.path.dataset, 
      name=self.meta.path.layer, # Channel
      type=layer_type, 
      datatype=self.meta.data_type,
    )

    rmt = BossRemote(boss_credentials)
    cutout = rmt.get_cutout(chan, mip, x_rng, y_rng, z_rng, no_cache=True)
    cutout = cutout.T
    cutout = cutout.astype(self.meta.dtype)
    cutout = cutout[::steps.x, ::steps.y, ::steps.z]

    if len(cutout.shape) == 3:
      cutout = cutout.reshape(tuple(list(cutout.shape) + [ 1 ]))

    if self.bounded or self.autocrop or bounds == bbox:
      return VolumeCutout.from_volume(self.meta, mip, cutout, bounds)

    # This section below covers the case where the requested volume is bigger
    # than the dataset volume and the bounds guards have been switched 
    # off. This is useful for Marching Cubes where a 1px excess boundary
    # is needed.
    shape = list(bbox.size3()) + [ cutout.shape[3] ]
    renderbuffer = np.zeros(shape=shape, dtype=self.meta.dtype, order='F')
    txrx.shade(renderbuffer, bbox, cutout, bounds)
    return VolumeCutout.from_volume(self.meta, mip, renderbuffer, bbox)

  def upload(self, image, offset, mip):
    shape = Vec(*image.shape[:3])
    offset = Vec(*offset)

    bounds = Bbox(offset, shape + offset)

    if bounds.subvoxel():
      raise exceptions.EmptyRequestException('Requested less than one pixel of volume. {}'.format(bounds))

    if self.autocrop:
      image, bounds = autocropfn(self.meta, image, bounds, mip)
      offset = bounds.minpt

    check_grid_aligned(
      self.meta, image, bounds, mip, 
      throw_error=(self.non_aligned_writes == False)
    )

    x_rng = [ bounds.minpt.x, bounds.maxpt.x ]
    y_rng = [ bounds.minpt.y, bounds.maxpt.y ]
    z_rng = [ bounds.minpt.z, bounds.maxpt.z ]

    layer_type = 'image' if self.layer_type == 'unknown' else self.meta.layer_type

    chan = ChannelResource(
      collection_name=self.meta.path.bucket, 
      experiment_name=self.meta.path.dataset, 
      name=self.meta.path.layer, # Channel
      type=layer_type, 
      datatype=self.meta.data_type,
    )

    if image.shape[3] == 1:
      image = image.reshape( image.shape[:3] )

    rmt = BossRemote(boss_credentials)
    image = image.T
    image = np.asfortranarray(image.astype(self.meta.dtype))

    rmt.create_cutout(chan, mip, x_rng, y_rng, z_rng, image)

  def exists(self, bbox, mip=None):
    raise NotImplementedError()

  def delete(self, bbox, mip=None):
    raise NotImplementedError()

  def transfer_to(self, cloudpath, bbox, mip, block_size=None, compress=True):
    raise NotImplementedError()