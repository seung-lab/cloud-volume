import os

from six.moves import range
import numpy as np
from PIL import Image
from tqdm import tqdm

from .lib import mkdir

class VolumeCutout(np.ndarray):

  def __new__(cls, buf, dataset_name, layer, mip, layer_type, bounds, handle, *args, **kwargs):
    return super(VolumeCutout, cls).__new__(cls, shape=buf.shape, buffer=np.ascontiguousarray(buf), dtype=buf.dtype)

  def __init__(self, buf, dataset_name, layer, mip, layer_type, bounds, handle, *args, **kwargs):
    super(VolumeCutout, self).__init__()
    
    self.dataset_name = dataset_name
    self.layer = layer
    self.mip = mip
    self.layer_type = layer_type
    self.bounds = bounds
    self.handle = handle

  def __del__(self):
    super(VolumeCutout, self).__del__()
    if self.handle:
      self.handle.close()

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

  def save_images(self, axis='z', channel=None, directory=None, image_format='PNG'):

    if directory is None:
      directory = os.path.join('./saved_images', self.dataset_name, self.layer, str(self.mip), self.bounds.to_filename())
    
    mkdir(directory)

    print("Saving to {}".format(directory))

    indexmap = {
      'x': 0,
      'y': 1,
      'z': 2,
    }

    index = indexmap[axis]

    channel = slice(None) if channel is None else channel

    for level in tqdm(range(self.shape[index]), desc="Saving Images"):
      if index == 0:
        img = self[level, :, :, channel ]
      elif index == 1:
        img = self[:, level, :, channel ]
      elif index == 2:
        img = self[:, :, level, channel ]
      else:
        raise NotImplemented

      num_channels = img.shape[2]

      for channel_index in range(num_channels):
        img2d = img[:, :, channel_index]

        # discovered that downloaded cube is in a weird rotated state.
        # it requires a 90deg counterclockwise rotation on xy plane (leaving z alone)
        # followed by a flip on Y
        if axis == 'z':
          img2d = np.flipud(np.rot90(img2d, 1)) 

        if img2d.dtype == 'uint8':
          img2d = Image.fromarray(img2d, 'L')
        else:
          img2d = img2d.astype('uint32')
          img2d[:,:] |= 0xff000000 # for little endian abgr
          img2d = Image.fromarray(img2d, 'RGBA')

        file_index = str(level).zfill(2)
        filename = '{}.{}'.format(file_index, image_format.lower())
        if num_channels > 1:
          filename = '{}-{}'.format(channel_index, filename)

        path = os.path.join(directory, filename)
        img2d.save(path, image_format)