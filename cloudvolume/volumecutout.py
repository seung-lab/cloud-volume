import os

from six.moves import range
import numpy as np
from tqdm import tqdm

from .lib import mkdir, save_images

class VolumeCutout(np.ndarray):

  def __new__(cls, buf, dataset_name, layer, mip, layer_type, bounds, handle, *args, **kwargs):
    return super(VolumeCutout, cls).__new__(cls, shape=buf.shape, buffer=np.asfortranarray(buf), dtype=buf.dtype, order='F')

  def __init__(self, buf, dataset_name, layer, mip, layer_type, bounds, handle, *args, **kwargs):
    super(VolumeCutout, self).__init__()
    
    self.dataset_name = dataset_name
    self.layer = layer
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
      dataset_name=volume.dataset_name,
      layer=volume.layer,
      mip=volume.mip,
      layer_type=volume.layer_type,
      bounds=bounds,
      handle=handle,
    )

  @property
  def num_channels(self):
    return self.shape[3]

  def save_images(self, axis='z', channel=None, directory=None, global_norm=True, image_format='PNG'):
    """See cloudvolume.lib.save_images for more information."""
    if directory is None:
      directory = os.path.join('./saved_images', self.dataset_name, self.layer, str(self.mip), self.bounds.to_filename())

    return save_images(self, axis, channel, directory, global_norm, image_format)
