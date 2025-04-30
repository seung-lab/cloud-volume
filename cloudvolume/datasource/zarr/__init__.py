from cloudfiles import CloudFiles

from ..zarr2 import create_zarr2
from ..zarr3 import create_zarr3

from ...cloudvolume import register_plugin

def create_zarr(
  cloudpath:str, *args, **kwargs 
):
  cf = CloudFiles(cloudpath)
  is_zarr3 = cf.exists("zarr.json")

  if is_zarr3:
    return create_zarr3(cloudpath, *args, **kwargs)
  else:
    return create_zarr2(cloudpath, *args, **kwargs)

def register():
  register_plugin('zarr', create_zarr)