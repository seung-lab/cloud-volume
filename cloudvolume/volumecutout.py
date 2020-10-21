import os

import json
from six.moves import range

import numpy as np
from tqdm import tqdm

from .lib import mkdir, save_images

from . import microviewer

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

  # How to add a new attribute to an ndarray:
  # https://docs.scipy.org/doc/numpy-1.13.0/uer/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray

  # Overriding __getitem__ and __setitem__ are 
  # not easy to do. 
  def __array_finalize__(self, obj):
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

  def __del__(self):
    sup = super(VolumeCutout, self)
    if hasattr(sup, '__del__'):
      sup.__del__()
    self.close()

  @classmethod
  def from_volume(cls, meta, mip, buf, bounds, handle=None):
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
  def num_channels(self):
    return self.shape[3]

  def save_images(
    self, directory=None, axis='z', 
    channel=None, global_norm=True, 
    image_format='PNG', progress=True
  ):
    """See cloudvolume.lib.save_images for more information."""
    if directory is None:
      directory = os.path.join('./saved_images', self.path.dataset, self.path.layer, str(self.mip), self.bounds.to_filename())

    return save_images(
      self, directory, axis, 
      channel, global_norm, image_format, 
      progress
    )

  def viewer(self, port=8080):
    """Start a local web app on the given port that lets you explore this cutout."""
    microviewer.run([ self ], port=port)
