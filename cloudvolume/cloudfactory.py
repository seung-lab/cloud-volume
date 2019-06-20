



def CloudVolumeFactory(
    image_cloudpath, # format protocol:// storage provider:// path
    mesh_cloudpath=None,
    skeleton_cloudpath=None,
    cloudpath, mip=0, bounded=True, autocrop=False, 
    fill_missing=False, cache=False, compress_cache=None, 
    cdn_cache=True, progress=INTERACTIVE, info=None, provenance=None, 
    compress=None, # only one option for BOSS
    non_aligned_writes=False, # not applicable to BOSS? actually.... it might be good to enforce it.
    parallel=1,
    delete_black_uploads=False
  ):

  path = lib.extract_path(cloudpath)



  if path.format == 'precomputed': # classic and sharded based on info

  elif path.format == 'graphene':

  elif path.format == 'boss': # implied s3










