import sys
import time
from typing import Optional, Union, Tuple

import multiprocessing as mp
import numpy as np
from tqdm import tqdm

from cloudfiles.paths import normalize

from .exceptions import UnsupportedFormatError, DimensionError, InfoUnavailableError
from .lib import generate_random_string
from .paths import strict_extract, to_https_protocol
from .types import CompressType, ParallelType, CacheType, SecretsType

# NOTE: Plugins are registered in __init__.py

# Set the interpreter bool
try:
  INTERACTIVE = bool(sys.ps1)
except AttributeError:
  INTERACTIVE = bool(sys.flags.interactive)

REGISTERED_PLUGINS = {}
def register_plugin(key, creation_function):
  REGISTERED_PLUGINS[key.lower()] = creation_function

def compute_num_threads(num_threads:ParallelType) -> int:
  if isinstance(num_threads, bool):
    return mp.cpu_count() if num_threads == True else 1
  elif num_threads == 0:
    return mp.cpu_count()
  elif num_threads <= 0:
    raise ValueError(f'Number of processes or threads must be >= 0. Got: {num_threads}')
  else:
    return max(int(num_threads), 1)

class SharedConfiguration(object):
  """
  Hack around python's inability to
  pass primatives by reference.
  We would like updates to e.g. mip or parallel
  to be passively absorbed by all listening
  data sources rather than work hard to actively
  synchronize them.
  """
  def __init__(
    self, cdn_cache:bool, compress:CompressType, 
    compress_level:Optional[int], green:bool,
    mip:int, parallel:ParallelType, progress:bool, secrets:SecretsType,
    spatial_index_db:Optional[str], cache_locking:bool,
    codec_threads:ParallelType,
    *args, **kwargs
  ):
    self.cdn_cache = cdn_cache
    self.cache_locking = bool(cache_locking)
    self.codec_threads = compute_num_threads(codec_threads)
    self.compress = compress
    self.compress_level = compress_level
    self.green = bool(green)
    self.mip = mip
    self.parallel = compute_num_threads(parallel) 
    self.progress = bool(progress)
    self.secrets = secrets
    self.spatial_index_db = spatial_index_db
    self.args = args
    self.kwargs = kwargs

class CloudVolume:
  def __new__(cls,
    cloudpath:str, 
    mip:int = 0, 
    bounded:bool = True, 
    autocrop:bool = False, 
    fill_missing:bool = False,
    cache:CacheType = False, 
    compress_cache:CompressType = None,
    cdn_cache:bool = True, 
    progress:bool = INTERACTIVE,
    info:dict = None,
    provenance:dict = None,
    compress:CompressType = None,
    compress_level:Optional[int] = None, 
    non_aligned_writes:bool = False,
    parallel:ParallelType = 1,
    delete_black_uploads:bool = False, 
    background_color:int = 0,
    green_threads:bool = False,
    use_https:bool = False,
    max_redirects:int = 10,
    mesh_dir:Optional[str] = None,
    skel_dir:Optional[str] = None, 
    agglomerate:bool = False,
    secrets:SecretsType = None, 
    spatial_index_db:Optional[str] = None,
    lru_bytes:int = 0,
    cache_locking:bool = True,
    lru_encoding:str = "same",
    timestamp:Optional[int] = None,
    codec_threads:int = 1,
  ):
    """
    A "serverless" Python client for reading and writing arbitrarily large 
    Neuroglancer Precomputed volumes both locally and on cloud services using.  
    A CloudVolume instance represents a dataset interface at a given mip level 
    (it doesn't load the entire dataset into memory).  

    Neuroglancer datasets specify metadata in an `info` file located at the root 
    of a data layer. It contains, among other things, the bounds of the 
    volume described as a 3D "voxel_offset" and 3D "size" in voxels, and the
    resolution of the dataset.

    Example:

      from cloudvolume import CloudVolume

      vol = CloudVolume('gs://mylab/mouse/image', progress=True)
      image = vol[:,:,:] # Download an image stack as a numpy array
      vol[:,:,:] = image # Upload an image stack from a numpy array
      
      label = 1
      mesh = vol.mesh.get(label) 
      skel = vol.skeleton.get(label)

    Required:
      cloudpath: Path to the dataset layer. This should match storage's supported
        providers.

        e.g. Google: gs://$BUCKET/$DATASET/$LAYER/
             S3    : s3://$BUCKET/$DATASET/$LAYER/
             Lcl FS: file:///tmp/$DATASET/$LAYER/
             Boss  : boss://$COLLECTION/$EXPERIMENT/$CHANNEL
             HTTP/S: http(s)://.../$CHANNEL
             matrix: matrix://$BUCKET/$DATASET/$LAYER/
    Optional:
      agglomerate: (bool, graphene only) sets the default mode for downloading
        images to agglomerated (True) vs watershed (False).
      autocrop: (bool) If the specified retrieval bounding box exceeds the
          volume bounds, process only the area contained inside the volume. 
          This can be useful way to ensure that you are staying inside the 
          bounds when `bounded=False`.
      background_color: (number) Specifies what the "background value" of the
        volume is (traditionally 0). This is mainly for changing the behavior
        of delete_black_uploads.
      bounded: (bool) If a region outside of volume bounds is accessed:
          True: Throw an error
          False: Allow accessing the region. If no files are present, an error 
              will still be thrown. Consider combining this option with 
              `fill_missing=True`. However, this can be dangrous as it allows
              missing files and potentially network errors to be intepreted as 
              zeros.
      cache: (bool or str) Store downs and uploads in a cache on disk
            and preferentially read from it before redownloading.
          - falsey value: no caching will occur.
          - True: cache will be located in a standard location.
          - non-empty string: cache is located at this file path

          After initialization, you can adjust this setting via:
          `cv.cache.enabled = ...` which accepts the same values.

          Note: This cache is totally separate from the LRU controlled by 
          lru_bytes.
      cache_locking: (bool) Use CloudFiles inter-process file locking to make cache 
        access safe for multiple simultaneous processes.
      cdn_cache: (int, bool, or str) Sets Cache-Control HTTP header on uploaded 
        image files. Most cloud providers perform some kind of caching. As of 
        this writing, Google defaults to 3600 seconds. Most of the time you'll 
        want to go with the default. 
        - int: number of seconds for cache to be considered fresh (max-age)
        - bool: True: max-age=3600, False: no-cache
        - str: set the header manually
      codec_threads: (int) for data codecs which support
        multithreading (e.g. JPEG-XL), this sets the number of threads to use. 
        0 means use all cores.
      compress: (bool, str, None) pick which compression method to use.
          None: (default) gzip for raw arrays and no additional compression
            for compressed_segmentation and fpzip.
          bool: 
            True=gzip, 
            False=no compression, Overrides defaults
          str: 
            'gzip': Extension so that we can add additional methods in the future 
                    like lz4 or zstd. 
            'br': Brotli compression, better compression rate than gzip
            '': no compression (same as False).
      compress_level: (int, None) level for compression. Higher number results
          in better compression but takes longer.
        Defaults to 9 for gzip (ranges from 0 to 9).
        Defaults to 5 for brotli (ranges from 0 to 11).
      compress_cache: (None or bool) If not None, override default compression 
          behavior for the cache.
      delete_black_uploads: (bool) If True, on uploading an entirely black chunk,
          issue a DELETE request instead of a PUT. This can be useful for avoiding storing
          tiny files in the region around an ROI. Some storage systems using erasure coding 
          don't do well with tiny file sizes.
      fill_missing: (bool) If a chunk file is unable to be fetched:
          True: Use a block of zeros
          False: Throw an error
      green_threads: (bool) Use green threads instead of preemptive threads. This
        can result in higher download performance for some compression types. Preemptive
        threads seem to reduce performance on multi-core machines that aren't densely
        loaded as the CPython threads are assigned to multiple cores and the thrashing
        + GIL reduces performance. You'll need to add the following code to the top
        of your program to use green threads:

            import gevent.monkey
            gevent.monkey.patch_all(threads=False)
      lru_bytes: (int) number of bytes used to cache recently used image 
        tiles in memory. This is an in-memory cache and is completely separate from
        the `cache` parameter that handles disk IO. Tiles are stripped over only their
        second stage compression.
      lru_encoding: (str) You can set the LRU encoding to be different from the source
        encoding. This can help reduce the memory usage of the LRU or take advantage of
        special codec properties. "same" means use the same encoding as the source.
        Using "raw" will speed up the LRU at the expense of memory if the source encoding is 
        not raw. Using a different encoding than the source encoding, other than "raw", will
        initially be slower as chunks will need to be reencoded.
      info: (dict) In lieu of fetching a neuroglancer info file, use this one.
          This is useful when creating new datasets and for repeatedly initializing
          a new cloudvolume instance.
      max_redirects: (int) if > 0, allow up to this many redirects via info file 'redirect'
          data fields. If <= 0, allow no redirections and access the current info file directly
          without raising an error.
      mesh_dir: (str) if not None, override the info['mesh'] key before pulling the
        mesh info file.
      mip: (int or iterable) Which level of downsampling to read and write from.
          0 is the highest resolution. You can also specify the voxel resolution
          like mip=[6,6,30] which will search for the appropriate mip level.
      non_aligned_writes: (bool) Enable non-aligned writes. Not multiprocessing 
          safe without careful design. When not enabled, a 
          cloudvolume.exceptions.AlignmentError is thrown for non-aligned writes. 
          
          https://github.com/seung-lab/cloud-volume/wiki/Advanced-Topic:-Non-Aligned-Writes

      parallel (int: 1, bool): Number of extra processes to launch, 1 means only 
          use the main process. If parallel is True use the number of CPUs 
          returned by multiprocessing.cpu_count(). When parallel > 1, shared
          memory (Linux) or emulated shared memory via files (other platforms) 
          is used by the underlying download.
      progress: (bool) Show progress bars. 
          Defaults to True in interactive python, False in script execution mode.
      provenance: (string, dict) In lieu of fetching a provenance 
          file, use this one. 
      secrets: (dict) provide per-instance authorization tokens. If not provided,
        defaults to looking in .cloudvolume/secrets for necessary tokens.
      skel_dir: (str) if not None, override the info['skeletons'] key before 
        pulling the skeleton info file.
      spatial_index_db: (str) A path to an sqlite3 or mysql database that follows 
        the following uri schema. sqlite is assumed if no scheme is present in 
        the uri.
          [sqlite://]filename.db
          mysql://<username>:<password>@<host>:<port>/<db_name>

        Igneous generated datasets include a JSON based spatial
        database that tiles the dataset. This can be fast enough up to about 100 TVx
        datasets. Above that, a proper database is required for efficient queries.
        We provide multiple SQL database types that the index can be hosted on.
      timestamp: (int, graphene only) Sets the timestamp for graphene proofreading 
        volumes so you can get the state at a given proofreading status.
      use_https: (bool) maps gs:// and s3:// to their respective https paths. The 
        https paths hit a cached, read-only version of the data and may be faster.
    """
    if use_https:
      cloudpath = to_https_protocol(cloudpath)

    kwargs = dict(locals())
    del kwargs['cls']

    def init(cloudpath):
      path = strict_extract(cloudpath)
      if path.format in REGISTERED_PLUGINS:
        kwargs["cloudpath"] = normalize(cloudpath)
        return REGISTERED_PLUGINS[path.format](**kwargs)
      else:
        raise UnsupportedFormatError(
          "Unknown format {}".format(path.format)
        )

    try:
      return init(cloudpath)
    except InfoUnavailableError as err:
      if 'precomputed://' not in cloudpath and cloudpath[:4] != 'zarr':
        try:
          return init('zarr://' + cloudpath)
        except:
          raise err
      else:
        raise err


  @classmethod
  def create_new_info(cls, *args, **kwargs):
    from .frontends import CloudVolumePrecomputed
    # For backwards compatibility, but this only 
    # makes sense for Precomputed anyway
    return CloudVolumePrecomputed.create_new_info(*args, **kwargs)

  @classmethod
  def from_crackle(cls,
    src:Union[str,bytes],
    cloudpath:str,
    resolution:Tuple[int,int,int] = (1,1,1),
    voxel_offset:Tuple[int,int,int] = (0,0,0),
    chunk_size:Tuple[int,int,int] = (256,256,16),
    encoding:str = "compressed_segmentation",
    compress:str = "gzip",
    progress:bool = False,
    allow_mmap:bool = True,
  ):
    """
    Created a precomputed segmentation dataset from a 
    single crackle file representing a whole dataset.
    """
    import crackle

    if isinstance(src, crackle.CrackleArray):
      arr = src
    elif isinstance(src, str):
      arr = crackle.aload(src, allow_mmap=allow_mmap)
    else:
      arr = crackle.CrackleArray(src)

    info = cls.create_new_info(
      num_channels=1, 
      layer_type="segmentation",
      data_type=np.dtype(arr.dtype).name,
      encoding=encoding, 
      resolution=resolution,
      voxel_offset=voxel_offset, 
      volume_size=arr.shape[:3],
      chunk_size=chunk_size,
      max_mip=0,
    )
    
    vol = CloudVolume(
      cloudpath, info=info, bounded=True, 
      compress=compress, progress=False,
    )
    # save the info file
    vol.commit_info()
    if isinstance(src, str):
      vol.provenance.sources = [src]
    vol.provenance.processing.append({
      'method': 'from_crackle',
      'date': time.strftime('%Y-%m-%d %H:%M %Z')
    })
    vol.commit_provenance()

    sz = arr.shape[2]
    cz = chunk_size[2]

    n_z_chunks = int(np.ceil(sz / cz))

    for z_i in tqdm(range(n_z_chunks), desc="Converting", disable=(not progress)):
      slc = np.s_[:,:, (z_i * cz) : min( (z_i+1) * cz, sz) ]
      labels = arr[slc]
      vol[slc] = labels

    return vol

  @classmethod
  def from_numpy(cls, 
    arr, 
    vol_path='file:///tmp/image/' + generate_random_string(),
    resolution=(4,4,40), voxel_offset=(0,0,0), 
    chunk_size=(128,128,64), layer_type=None, max_mip=0,
    encoding='raw', compress=None, progress=False,
    encoding_level=None, encoding_effort=None,
  ):
    """
    Create a new dataset from a numpy array.

    max_mip: (int) the maximum mip level id in the info file. 
    Note that currently the numpy array can only sit in mip 0,
    the max_mip was only created in info file.
    the numpy array itself was not downsampled. 
    """
    if not layer_type:
      if arr.dtype in (bool, np.uint32, np.uint64, np.uint16):
        layer_type = 'segmentation'
      elif np.issubdtype(arr.dtype, np.integer) \
                        or np.issubdtype(arr.dtype, np.floating):
        layer_type = 'image'
      else:
        raise ValueError(f"{arr.dtype} is not supported.")

    if arr.ndim == 3:
      num_channels = 1
    elif arr.ndim == 4:
      num_channels = arr.shape[-1]
    else:
      raise DimensionError(f"CloudVolume only accepts 3 or 4 dimensional images. Got: {arr.ndim}")
      
    info = cls.create_new_info(
      num_channels, layer_type, np.dtype(arr.dtype).name,
      encoding, resolution,
      voxel_offset, arr.shape[:3],
      chunk_size=chunk_size, max_mip=max_mip,
      encoding_level=encoding_level, encoding_effort=encoding_effort, 
    )
    
    vol = CloudVolume(
      vol_path, info=info, bounded=True, 
      compress=compress, progress=progress
    )
    # save the info file
    vol.commit_info()
    vol.provenance.processing.append({
      'method': 'from_numpy',
      'date': time.strftime('%Y-%m-%d %H:%M %Z')
    })
    vol.commit_provenance()
    # save the numpy array
    vol[:,:,:] = arr
    return vol
