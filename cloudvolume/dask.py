"""Utilties for translating to/from dask arrays.

NOTE: Using the thread-based dask scheduler is very inefficient with
      CloudVolume because CV is pure python and will cause substantial
      GIL contention between the dask worker threads.  It is HIGHLY
      ADVISABLE to use a distributed process scheduler with one thread
      per process.

"""
import numpy as np
from .cloudvolume import CloudVolume


def to_cloudvolume(arr,
                   cloudpath,
                   resolution=(1, 1, 1),
                   voxel_offset=(0, 0, 0),
                   layer_type=None,
                   encoding='raw',
                   max_mip=0,
                   compute=True,
                   return_stored=False,
                   **kwargs):
  """Save 3d or 4d dask array to the precomputed CloudVolume storage format.

  NOTE: DO NOT USE thread-based dask scheduler. See comment at top of module.

  See https://docs.dask.org/en/latest/array.html for details about the format.

  Parameters
  ----------
  arr: dask.array
    Data to store
  cloudpath: str
    Path to the dataset layer. This should match storage's supported
    providers.
    e.g. Google: gs://$BUCKET/$DATASET/$LAYER/
         S3    : s3://$BUCKET/$DATASET/$LAYER/
         Lcl FS: file:///tmp/$DATASET/$LAYER/
         Boss  : boss://$COLLECTION/$EXPERIMENT/$CHANNEL
         HTTP/S: http(s)://.../$CHANNEL
         matrix: matrix://$BUCKET/$DATASET/$LAYER/
  resolution: Iterable of ints of length 3
    The x, y, z voxel dimensions in nanometers
  voxel_offset: Iterable of ints of length 3
    The x, y, z beginning of dataset in positive cartesian space.
  layer_type: str
    "image" or "segmentation"
  max_mip: int
    Maximum mip level id.
  compute: boolean, optional
    If true compute immediately, return ``dask.delayed.Delayed`` otherwise.
  return_stored: boolean, optional
    Optionally return stored results.
  kwargs: passed to the ``cloudvolume.CloudVolume()`` function, e.g., compression options

  Raises
  ------
  ValueError
    If ``arr`` has ndim different that 3 or 4, or ``layer_type`` is unsupported.

  Returns
  -------
  See notes on `compute` and `return_stored` parameters.
  """
  import dask
  import dask.array as da
  if not da.core._check_regular_chunks(arr.chunks):
    raise ValueError('Attempt to save array to cloudvolume with irregular '
                     'chunking, please call `arr.rechunk(...)` first.')

  if not layer_type:
    if arr.dtype in (bool, np.uint32, np.uint64, np.uint16):
      layer_type = 'segmentation'
    elif np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating):
      layer_type = 'image'
    else:
      raise ValueError('Unsupported layer_type for CloudVolume: %s' % layer_type)

  if arr.ndim == 3:
    num_channels = 1
    chunk_size = arr.chunksize
  elif arr.ndim == 4:
    num_channels = arr.shape[-1]
    chunk_size = arr.chunksize[:3]
  else:
    raise ValueError('CloudVolume only supports 3 or 4 dimensions.  Array has %d.' % arr.ndim)

  info = CloudVolume.create_new_info(num_channels,
                                     layer_type,
                                     arr.dtype.name,
                                     encoding,
                                     resolution,
                                     voxel_offset,
                                     arr.shape[:3],
                                     chunk_size=chunk_size,
                                     max_mip=max_mip)

  # Delay writing any metadata until computation time.
  #   - the caller may never do the full computation
  #   - the filesystem may be slow, and there is a desire to open files
  #     in parallel on worker machines.
  vol = dask.delayed(_create_cloudvolume)(cloudpath, info, **kwargs)
  return arr.store(vol, lock=False, compute=compute, return_stored=return_stored)


def _create_cloudvolume(cloudpath, info, **kwargs):
  """Create cloudvolume and commit metadata."""
  vol = CloudVolume(cloudpath, info=info, progress=False, **kwargs)
  vol.commit_info()
  vol.provenance.processing = [{'method': 'cloudvolume.dask.to_cloudvolume'}]
  vol.commit_provenance()
  return vol


def from_cloudvolume(cloudpath, chunks=None, name=None, **kwargs):
  """Load dask array from a cloudvolume compatible dataset.

  NOTE: DO NOT USE thread-based dask scheduler. See comment at top of module.

  Volumes with a single channel will be returned as 4d arrays with a
  length-1 channel dimension, even if they were stored from 3d data.

  The channel dimension is returned as a single-chunk by default, as that
  is how CloudVolumes are stored.

  See https://docs.dask.org/en/latest/array.html for details about the format.

  Parameters
  ----------
  cloudpath: str
    Path to the dataset layer. This should match storage's supported
    providers.
    e.g. Google: gs://$BUCKET/$DATASET/$LAYER/
         S3    : s3://$BUCKET/$DATASET/$LAYER/
         Lcl FS: file:///tmp/$DATASET/$LAYER/
         Boss  : boss://$COLLECTION/$EXPERIMENT/$CHANNEL
         HTTP/S: http(s)://.../$CHANNEL
         matrix: matrix://$BUCKET/$DATASET/$LAYER/
  chunks: tuple of ints or tuples of ints
    Passed to ``da.from_array``, allows setting the chunks on
    initialisation, if the chunking scheme in the stored dataset is not
    optimal for the calculations to follow.  Note that the chunking should
    be compatible with an underlying 4d array.
  name: str, optional
     An optional keyname for the array.  Defaults to hashing the input
  kwargs: passed to the ``cloudvolume.CloudVolume()`` function, e.g., compression options

  Returns
  -------
  Dask array
  """
  import dask.array as da
  from dask.base import tokenize

  vol = CloudVolume(cloudpath, progress=False, **kwargs)
  if chunks is None:
      chunks = tuple(vol.chunk_size) + (vol.num_channels, )
  if name is None:
      name = 'from-cloudvolume-' + tokenize(vol, chunks, **kwargs)
  return da.from_array(vol, chunks, name=name)
