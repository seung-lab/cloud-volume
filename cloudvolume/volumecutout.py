import os

import json
from six.moves import range

import numpy as np
from tqdm import tqdm

from .lib import mkdir, save_images

from . import viewer

class VolumeCutout(np.ndarray):

  def __new__(cls, buf, path, cloudpath, resolution, mip, layer_type, bounds, handle, *args, **kwargs):
    return super(VolumeCutout, cls).__new__(cls, shape=buf.shape, buffer=np.asfortranarray(buf), dtype=buf.dtype, order='F')

  def __init__(self, buf, path, cloudpath, resolution, mip, layer_type, bounds, handle, *args, **kwargs):
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

  def close(self):
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

  def __del__(self):
    sup = super(VolumeCutout, self)
    if hasattr(sup, '__del__'):
      sup.__del__()
    self.close()

  @classmethod
  def from_volume(cls, volume, buf, bounds, handle=None):
    return VolumeCutout(
      buf=buf,
      path=volume.path,
      cloudpath=volume.cloudpath,
      resolution=volume.resolution,
      mip=volume.mip,
      layer_type=volume.layer_type,
      bounds=bounds,
      handle=handle,
    )

  @property
  def num_channels(self):
    return self.shape[3]

  def save_images(self, directory=None, axis='z', channel=None, global_norm=True, image_format='PNG'):
    """See cloudvolume.lib.save_images for more information."""
    if directory is None:
      directory = os.path.join('./saved_images', self.dataset_name, self.layer, str(self.mip), self.bounds.to_filename())

    return save_images(self, directory, axis, channel, global_norm, image_format)

  def view(self, port=8080):
    """Start a local web app on the given port that lets you explore this cutout."""
    viewer.run([ self ], port=port)
