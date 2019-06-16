from __future__ import print_function

from intern.remote.boss import BossRemote
from intern.resource.boss.resource import (
  ChannelResource, ExperimentResource, CoordinateFrameResource
)

from cloudvolume import lib
from cloudvolume.cacheservice import CacheService
from cloudvolume import exceptions 
from cloudvolume.lib import ( 
  colorize, red, mkdir, Vec, Bbox,  
  jsonify, generate_slices,
  generate_random_string
)
from cloudvolume.secrets import boss_credentials
from cloudvolume.volumecutout import VolumeCutout

def download(self, requested_bbox, steps, channel_slice=slice(None)):
  bounds = Bbox.clamp(requested_bbox, self.bounds)
  
  if bounds.subvoxel():
    raise exceptions.EmptyRequestException('Requested less than one pixel of volume. {}'.format(bounds))

  x_rng = [ bounds.minpt.x, bounds.maxpt.x ]
  y_rng = [ bounds.minpt.y, bounds.maxpt.y ]
  z_rng = [ bounds.minpt.z, bounds.maxpt.z ]

  layer_type = 'image' if self.layer_type == 'unknown' else self.layer_type

  chan = ChannelResource(
    collection_name=self.path.bucket, 
    experiment_name=self.path.dataset, 
    name=self.path.layer, # Channel
    type=layer_type, 
    datatype=self.dtype,
  )

  rmt = BossRemote(boss_credentials)
  cutout = rmt.get_cutout(chan, self.mip, x_rng, y_rng, z_rng, no_cache=True)
  cutout = cutout.T
  cutout = cutout.astype(self.dtype)
  cutout = cutout[::steps.x, ::steps.y, ::steps.z]

  if len(cutout.shape) == 3:
    cutout = cutout.reshape(tuple(list(cutout.shape) + [ 1 ]))

  if self.bounded or self.autocrop or bounds == requested_bbox:
    return VolumeCutout.from_volume(self, cutout, bounds)

  # This section below covers the case where the requested volume is bigger
  # than the dataset volume and the bounds guards have been switched 
  # off. This is useful for Marching Cubes where a 1px excess boundary
  # is needed.
  shape = list(requested_bbox.size3()) + [ cutout.shape[3] ]
  renderbuffer = np.zeros(shape=shape, dtype=self.dtype, order='F')
  txrx.shade(renderbuffer, requested_bbox, cutout, bounds)
  return VolumeCutout.from_volume(self, renderbuffer, requested_bbox)

def upload_boss_image(self, img, offset):
  shape = Vec(*img.shape[:3])
  offset = Vec(*offset)

  bounds = Bbox(offset, shape + offset)

  if bounds.subvoxel():
    raise exceptions.EmptyRequestException('Requested less than one pixel of volume. {}'.format(bounds))

  x_rng = [ bounds.minpt.x, bounds.maxpt.x ]
  y_rng = [ bounds.minpt.y, bounds.maxpt.y ]
  z_rng = [ bounds.minpt.z, bounds.maxpt.z ]

  layer_type = 'image' if self.layer_type == 'unknown' else self.layer_type

  chan = ChannelResource(
    collection_name=self.path.bucket, 
    experiment_name=self.path.dataset, 
    name=self.path.layer, # Channel
    type=layer_type, 
    datatype=self.dtype,
  )

  if img.shape[3] == 1:
    img = img.reshape( img.shape[:3] )

  rmt = BossRemote(boss_credentials)
  img = img.T
  img = np.asfortranarray(img.astype(self.dtype))

  rmt.create_cutout(chan, self.mip, x_rng, y_rng, z_rng, img)

