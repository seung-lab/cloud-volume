from __future__ import annotations

from typing import Any, Optional

from intern.remote.boss import BossRemote
from intern.resource.boss.resource import (
  ChannelResource, ExperimentResource, CoordinateFrameResource
)

from .. import autocropfn, readonlyguard, ImageSourceInterface

from ... import exceptions
from ...lib import (
  colorize, red, mkdir, Vec, Bbox,
  jsonify, generate_random_string
)
from ...secrets import boss_credentials
from ...volumecutout import VolumeCutout
from ..precomputed.image.common import shade

class BossImageSource(ImageSourceInterface):
  def __init__(
    self, config: Any, meta: Any, cache: Any,
    autocrop: bool = False, bounded: bool = True,
    non_aligned_writes: bool = False,
    delete_black_uploads: bool = False,
    readonly: bool = False,
  ) -> None:
    self.config = config
    self.meta = meta
    self.cache = cache

    self.autocrop = bool(autocrop)
    self.bounded = bool(bounded)
    self.non_aligned_writes = bool(non_aligned_writes)
    self.readonly = bool(readonly)

  def download(self, bbox: Any, mip: int, parallel: int = 1, renumber: bool = False) -> VolumeCutout:
    if parallel != 1:
      raise ValueError("Only parallel=1 is supported for boss.")
    elif renumber != False:
      raise ValueError("Only renumber=False is supported for boss.")

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
    shade(renderbuffer, bbox, cutout, bounds)
    return VolumeCutout.from_volume(self.meta, mip, renderbuffer, bbox)

  @readonlyguard
  def upload(self, image: Any, offset: Any, mip: int) -> None:
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

  def exists(self, bbox: Any, mip: Optional[int] = None) -> Any:
    raise NotImplementedError()

  @readonlyguard
  def delete(self, bbox: Any, mip: Optional[int] = None) -> None:
    raise NotImplementedError()

  def transfer_to(self, cloudpath: str, bbox: Any, mip: int, block_size: Optional[int] = None, compress: bool = True) -> Any:
    raise NotImplementedError()
