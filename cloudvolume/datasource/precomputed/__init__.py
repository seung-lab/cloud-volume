"""
The Precomputed format is a neuroscience imaging format 
designed for cloud storage. The specification is located
here:

https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed

This datasource contains the code for manipulating images.
"""

from cloudvolume import lib

from .rx import download
from .tx import upload

class PrecomputedImageSource(object):
  def __init__(
    self, cloudpath, mip=0, bounded=True, autocrop=False, 
    fill_missing=False, cache=False, compress_cache=None, 
    cdn_cache=True, progress=INTERACTIVE, 
    compress=None, non_aligned_writes=False, parallel=1,
    delete_black_uploads=False
  ):

    self.autocrop = bool(autocrop)
    self.bounded = bool(bounded)
    self.cdn_cache = cdn_cache
    self.compress = compress
    self.delete_black_uploads = bool(delete_black_uploads)
    self.fill_missing = bool(fill_missing)
    self.non_aligned_writes = bool(non_aligned_writes)
    self.progress = bool(progress)
    self.path = lib.extract_path(cloudpath)
    self.shared_memory_id = self.generate_shared_memory_location()
    
    self.init_submodules(cache)
    self.cache.compress = compress_cache

    if self.path.layer == 'info':
      warn("WARNING: Your layer is named 'info', is that what you meant? {}".format(
          self.path
      ))

    # needs to be set after info is defined since
    # its setter is based off of scales
    self.mip = mip

    self.pid = os.getpid()

  def download(self, bbox):
    pass 

  def upload(self, bbox):
    pass

