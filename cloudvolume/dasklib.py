"""Utilties for translating to/from dask arrays."""
import numpy as np
from .cloudvolume import CloudVolume


def dask_to_cloudvolume(arr,
                        vol_path,
                        resolution=(1, 1, 1),
                        voxel_offset=(0, 0, 0),
                        layer_type=None,
                        encoding='raw',
                        max_mip=0,
                        compute=True,
                        return_stored=False,
                        **kwargs):
    """Save 3d or 4d dask array to the precomputed CloudVolume storage format.

    See https://docs.dask.org/en/latest/array.html for details about the format.

    Parameters
    ----------
    arr: dask.array
        Data to store
    vol_path: str
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
    compute, return_stored: see ``store()``
    kwargs: passed to the ``cloudvolume.CloudVolume()`` function, e.g., compression options

    Raises
    ------
    ValueError
        If ``arr`` has ndim different that 3 or 4, or ``layer_type`` is unsupported.
    """
    import dask.array as da
    if not da.core._check_regular_chunks(arr.chunks):
        raise ValueError('Attempt to save array to cloudvolume with irregular '
                         'chunking, please call `arr.rechunk(...)` first.')

    if not layer_type:
        if arr.dtype in (np.bool, np.uint32, np.uint64, np.uint16):
            layer_type = 'segmentation'
        elif np.issubdtype(arr.dtype, np.integer) or np.issubdtype(
                arr.dtype, np.floating):
            layer_type = 'image'
        else:
            raise ValueError('Unsupported layer_type for CloudVolume: %s' %
                             layer_type)

    if arr.ndim == 3:
        num_channels = 1
        chunk_size = arr.chunksize
    elif arr.ndim == 4:
        num_channels = arr.shape[-1]
        chunk_size = arr.chunksize[:3]
    else:
        raise ValueError('Unsupported ndim for CloudVolume: %d' % arr.ndim)

    info = CloudVolume.create_new_info(num_channels,
                                       layer_type,
                                       arr.dtype.name,
                                       encoding,
                                       resolution,
                                       voxel_offset,
                                       arr.shape[:3],
                                       chunk_size=chunk_size,
                                       max_mip=max_mip)
    vol = CloudVolume(vol_path, info=info, progress=False, **kwargs)
    # pylint: disable=no-member
    vol.commit_info()
    vol.provenance.processing = [{'method': 'dask_to_cloudvolume'}]
    vol.commit_provenance()
    return arr.store(vol,
                     lock=False,
                     compute=compute,
                     return_stored=return_stored)


def dask_from_cloudvolume(vol_path, chunks=None, name=None, **kwargs):
    """Load dask array from the cloudvolume storage format.

    Volumes with a single channel will be returned as 4d arrays with a
    length-1 channel dimension, even if they were stored from 3d data.

    The channel dimension is returned as a single-chunk by default, as that
    is how CloudVolumes are stored.

    See https://docs.dask.org/en/latest/array.html for details about the format.

    Parameters
    ----------
    vol_path: str
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
    """
    import dask.array as da
    from dask.base import tokenize

    vol = CloudVolume(vol_path, progress=False, **kwargs)
    # pylint: disable=no-member
    if chunks is None:
        chunks = tuple(vol.chunk_size) + (vol.num_channels, )
    if name is None:
        name = 'from-cloudvolume-' + tokenize(vol, chunks, **kwargs)
    return da.from_array(vol, chunks, name=name)
