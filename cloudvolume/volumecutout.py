from __future__ import annotations

import os
from typing import Any, Optional

import json

import numpy as np
from tqdm import tqdm

from .lib import mkdir, save_images, Bbox
from .paths import ExtractedPath

class VolumeCutout(np.ndarray):

  def __new__(
    cls,
    buf: np.ndarray,
    path: ExtractedPath,
    cloudpath: str,
    resolution: np.ndarray | list[float],
    mip: int,
    layer_type: str,
    bounds: Bbox,
    handle: Any,
    *args: Any,
    **kwargs: Any,
  ) -> VolumeCutout:
    return super(VolumeCutout, cls).__new__(cls, shape=buf.shape, buffer=np.asfortranarray(buf), dtype=buf.dtype, order='F')

  def __init__(
    self,
    buf: np.ndarray,
    path: ExtractedPath,
    cloudpath: str,
    resolution: np.ndarray | list[float],
    mip: int,
    layer_type: str,
    bounds: Bbox,
    handle: Any,
    *args: Any,
    **kwargs: Any,
  ) -> None:
    super(VolumeCutout, self).__init__()

    self.dataset_name = path.dataset
    self.layer = path.layer
    self.path = path
    self.resolution = resolution
    self.cloudpath = cloudpath
    self.mip = mip
    self.layer_type = layer_type
    self.bounds = bounds
    self.handle = handle

  def close(self) -> None:
    # This bizzare construction is because of this error:
    # Traceback (most recent call last):
    #   File "cloud-volume/cloudvolume/volumecutout.py", line 30, in __del__
    #     self.close()
    #   File "cloud-volume/cloudvolume/volumecutout.py", line 26, in close
    #     if self.handle and not self.handle.closed:
    # ValueError: mmap closed or invalid

    # However testing if it is closed does not throw an error. So we test
    # for closure and capture the exception if self.handle is None.

    try:
      if not self.handle.closed:
        self.handle.close()
    except AttributeError:
      pass

  # How to add a new attribute to an ndarray:
  # https://docs.scipy.org/doc/numpy-1.13.0/uer/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray

  # Overriding __getitem__ and __setitem__ are
  # not easy to do.
  def __array_finalize__(self, obj: Any) -> None:
    if obj is None:
      return

    self.dataset_name = getattr(obj, 'dataset', None)
    self.layer = getattr(obj, 'layer', None)
    self.path = getattr(obj, 'path', None)
    self.resolution = getattr(obj, 'resolution', None)
    self.cloudpath = getattr(obj, 'cloudpath', None)
    self.mip = getattr(obj, 'mip', None)
    self.layer_type = getattr(obj, 'layer_type', None)
    self.bounds = getattr(obj, 'bounds', None)
    self.handle = None #getattr(obj, 'handle', None)

  def __del__(self) -> None:
    sup = super(VolumeCutout, self)
    if hasattr(sup, '__del__'):
      sup.__del__()
    self.close()

  @classmethod
  def from_volume(
    cls,
    meta: Any,
    mip: int,
    buf: np.ndarray,
    bounds: Bbox,
    handle: Any = None,
  ) -> VolumeCutout:
    return VolumeCutout(
      buf=buf,
      path=meta.path,
      cloudpath=meta.cloudpath,
      resolution=meta.resolution(mip),
      mip=mip,
      layer_type=meta.layer_type,
      bounds=bounds,
      handle=handle,
    )

  @property
  def num_channels(self) -> int:
    return self.shape[3]

  def save_images(
    self,
    directory: Optional[str] = None,
    axis: str = 'z',
    channel: Optional[int] = None,
    global_norm: bool = True,
    image_format: str = 'PNG',
    progress: bool = True,
  ) -> None:
    """See cloudvolume.lib.save_images for more information."""
    if directory is None:
      directory = os.path.join('./saved_images', self.path.dataset, self.path.layer, str(self.mip), self.bounds.to_filename())

    return save_images(
      self, directory, axis,
      channel, global_norm, image_format,
      progress
    )

  def viewer(self, port: int = 8080) -> None:
    """Start a local web app on the given port that lets you explore this cutout."""
    import microviewer
    microviewer.view(
      self,
      seg=(self.layer_type == "segmentation"),
      resolution=self.resolution,
      cloudpath=self.cloudpath,
      port=port,
    )
