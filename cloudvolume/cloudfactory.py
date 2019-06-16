



def CloudVolumeFactory(
    image_cloudpath, # format protocol:// storage provider:// path
    mesh_cloudpath=None,
    skeleton_cloudpath=None,

  ):

  path = lib.extract_path(cloudpath)



  if path.format == 'precomputed': # classic and sharded based on info

  elif path.format == 'graphene':

  elif path.format == 'boss': # implied s3










