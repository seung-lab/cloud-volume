"""
The Precomputed format is a neuroscience imaging format 
designed for cloud storage. The specification is located
here:

https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed

This datasource contains the code for manipulating images.
"""

from .rx import download
from .tx import upload

class PrecomputedImageSource(object):
  def __init__(self, cloudpath, shm, parallel, green, caching):

  def download(self, bbox):
    pass 

  def upload(self, bbox):
    pass

class ShardedPrecomputedImageSource(object):
  def __init__(self, cloudpath, shm, parallel, green, caching):

  def download(self, bbox):
    pass 

  def upload(self, bbox):
    pass



class BossImageSource(object):
  def __init__(self, cloudpath, shm, parallel, green, caching):

  def download(self, bbox):
    pass 

  def upload(self, bbox):
    pass
