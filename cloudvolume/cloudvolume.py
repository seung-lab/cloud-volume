from __future__ import print_function

from functools import partial
import json
import json5
import os
import re
import sys
import shutil
import weakref

from six.moves import range
import numpy as np
from PIL import Image
from tqdm import tqdm
from six import string_types

from intern.remote.boss import BossRemote
from intern.resource.boss.resource import ChannelResource, ExperimentResource, CoordinateFrameResource
from .secrets import boss_credentials, CLOUD_VOLUME_DIR

from . import lib, chunks
from .lib import ( 
  toabs, colorize, red, yellow, 
  mkdir, clamp, xyzrange, Vec, 
  Bbox, min2, max2, check_bounds, 
  jsonify 
)
from .meshservice import PrecomputedMeshService
from .provenance import DataLayerProvenance
from .storage import SimpleStorage, Storage
from .threaded_queue import ThreadedQueue

# Set the interpreter bool
try:
    INTERACTIVE = bool(sys.ps1)
except AttributeError:
    INTERACTIVE = bool(sys.flags.interactive)

if sys.version_info < (3,):
    integer_types = (int, long,)
else:
    integer_types = (int,)

__all__ = [ 'CloudVolume', 'EmptyVolumeException', 'EmptyRequestException' ]

def warn(text):
  print(colorize('yellow', text))

class EmptyVolumeException(Exception):
  """Raised upon finding a missing chunk."""
  pass

class EmptyRequestException(Exception):
  """
  Requesting uploading or downloading 
  a bounding box of less than one cubic voxel
  is impossible.
  """
  pass

DEFAULT_CHUNK_SIZE = (64,64,64)


class CloudVolume(object):
  """
  CloudVolume represents an interface to a dataset layer at a given
  mip level. You can use it to send and receive data from neuroglancer
  datasets on supported hosts like Google Storage, S3, and local Filesystems.

  Uploading to and downloading from a neuroglancer dataset requires specifying
  an `info` file located at the root of a data layer. Amongst other things, 
  the bounds of the volume are described in the info file via a 3D "offset" 
  and 3D "shape" in voxels.

  Required:
    cloudpath: Path to the dataset layer. This should match storage's supported
      providers.

      e.g. Google: gs://neuroglancer/$DATASET/$LAYER/
           S3    : s3://neuroglancer/$DATASET/$LAYER/
           Lcl FS: file:///tmp/$DATASET/$LAYER/
           Boss  : boss://$COLLECTION/$EXPERIMENT/$CHANNEL
  Optional:
    mip: (int) Which level of downsampling to read to/write from. 0 is the highest resolution.
    bounded: (bool) If a region outside of volume bounds is accessed:
        True: Throw an error
        False: Fill the region with black (useful for e.g. marching cubes's 1px boundary)
    autocrop: (bool) If the uploaded or downloaded region exceeds bounds, process only the
      area contained in bounds. Only has effect when bounded=True.
    fill_missing: (bool) If a file inside volume bounds is unable to be fetched:
        True: Use a block of zeros
        False: Throw an error
    cache: (bool or str) Store downloaded and uploaded files in a cache on disk 
      and preferentially read from it before redownloading. 
        - falsey value: no caching will occur.
        - True: cache will be located in a standard location.
        - non-empty string: cache is located at this file path
    cdn_cache: (int, bool, or str) Sets the Cache-Control HTTP header on uploaded image files.
      Most cloud providers perform some kind of caching. As of this writing, Google defaults to
      3600 seconds. Most of the time you'll want to go with the default. 
      - int: number of seconds for cache to be considered fresh (max-age)
      - bool: True: max-age=3600, False: no-cache
      - str: set the header manually
    info: (dict) in lieu of fetching a neuroglancer info file, use this provided one.
            This is useful when creating new datasets.
    provenance: (string, dict, or object) in lieu of fetching a neuroglancer provenance file, use this provided one.
            This is useful when doing multiprocessing.
    progress: (bool) Show tqdm progress bars. 
        Defaults True in interactive python, False in script execution mode.
    compress: (bool, str, None) pick which compression method to use. 
      None: (default) Use the pre-programmed recommendation (e.g. gzip raw arrays and compressed_segmentation)
      bool: True=gzip, False=no compression, Overrides defaults
      str: 'gzip', extension so that we can add additional methods in the future like lz4 or zstd. 
        '' means no compression (same as False).
  """
  def __init__(self, cloudpath, mip=0, bounded=True, autocrop=False, fill_missing=False, 
      cache=False, cdn_cache=True, progress=INTERACTIVE, info=None, provenance=None, 
      compress=None):

    self.autocrop = bool(autocrop)
    self.bounded = bounded
    self.cache = cache
    self.cdn_cache = cdn_cache
    self.compress = compress
    self.fill_missing = fill_missing
    self.mip = mip
    self.mesh = PrecomputedMeshService(weakref.proxy(self)) 
    self.progress = progress
    self.path = lib.extract_path(cloudpath)
    
    if self.cache:
      if not os.path.exists(self.cache_path):
        mkdir(self.cache_path)

      if not os.access(self.cache_path, os.R_OK|os.W_OK):
        raise IOError('Cache directory needs read/write permission: ' + self.cache_path)

    if info is None:
      self.refresh_info()
      if self.cache:
        self._check_cached_info_validity()
    else:
      self.info = info

    if provenance is None:
      self.provenance = None
      self.refresh_provenance()
      self._check_cached_provenance_validity()
    else:
      self.provenance = self._cast_provenance(provenance)

    try:
      self.mip = self.available_mips[self.mip]
    except:
      raise Exception("MIP {} has not been generated.".format(self.mip))

  @property
  def _storage(self):
    if self.path.protocol == 'boss':
      return None
    
    try:
      return Storage(self.layer_cloudpath, n_threads=0)
    except:
      if self.path.layer == 'info':
        warn("WARNING: Your layer is named 'info', is that what you meant? {}".format(
            self.path
        ))
      raise
      
  @classmethod
  def create_new_info(cls, num_channels, layer_type, data_type, encoding, resolution, voxel_offset, volume_size, mesh=None, chunk_size=DEFAULT_CHUNK_SIZE):
    """
    Used for creating new neuroglancer info files.

    Required:
      num_channels: (int) 1 for grayscale, 3 for RGB 
      layer_type: (str) typically "image" or "segmentation"
      data_type: (str) e.g. "uint8", "uint16", "uint32", "float32"
      encoding: (str) "raw" for binaries like numpy arrays, "jpeg"
      resolution: int (x,y,z), x,y,z voxel dimensions in nanometers
      voxel_offset: int (x,y,z), beginning of dataset in positive cartesian space
      volume_size: int (x,y,z), extent of dataset in cartesian space from voxel_offset
    
    Optional:
      mesh: (str) name of mesh directory, typically "mesh"
      chunk_size: int (x,y,z), dimensions of each downloadable 3D image chunk in voxels

    Returns: dict representing a single mip level that's JSON encodable
    """
    info = {
      "num_channels": int(num_channels),
      "type": layer_type,
      "data_type": data_type,
      "scales": [{
        "encoding": encoding,
        "chunk_sizes": [chunk_size],
        "key": "_".join(map(str, resolution)),
        "resolution": list(map(int, resolution)),
        "voxel_offset": list(map(int, voxel_offset)),
        "size": list(map(int, volume_size)),
      }],
    }

    if mesh:
      info['mesh'] = 'mesh' if not isinstance(mesh, string_types) else mesh

    return info

  def refresh_info(self):
    if self.cache:
      info = self._read_cached_json('info')
      if info:
        self.info = info
        return self.info

    self.info = self._fetch_info()
    self._maybe_cache_info()
    return self.info

  def _check_cached_info_validity(self):
    """
    ValueError if cache differs at all from source data layer with
    an excepton for volume_size which prints a warning.
    """
    cache_info = self._read_cached_json('info')
    if not cache_info:
      return

    fresh_info = self._fetch_info()

    mismatch_error = ValueError("""
      Data layer info file differs from cache. Please check whether this
      change invalidates your cache. 

      If VALID do one of:
        1) Manually delete the cache (see location below)
        2) Refresh your on-disk cache as follows:
          vol = CloudVolume(..., cache=False) # refreshes from source
          vol.cache = True
          vol.commit_info() # writes to disk
      If INVALID do one of: 
        1) Delete the cache manually (see cache location below) 
        2) Instantiate as follows: 
          vol = CloudVolume(..., cache=False) # refreshes info from source
          vol.flush_cache() # deletes cache
          vol.cache = True
          vol.commit_info() # writes info to disk

      CACHED: {cache}
      SOURCE: {source}
      CACHE LOCATION: {path}
      """.format(
        cache=cache_info, 
        source=fresh_info, 
        path=self.cache_path
    ))

    try:
      fresh_sizes = [ scale['size'] for scale in fresh_info['scales'] ]
      cache_sizes = [ scale['size'] for scale in cache_info['scales'] ]
    except KeyError:
      raise mismatch_error

    for scale in fresh_info['scales']:
      del scale['size']

    for scale in cache_info['scales']:
      del scale['size']

    if fresh_info != cache_info:
      raise mismatch_error

    if fresh_sizes != cache_sizes:
      warn("WARNING: Data layer bounding box differs in cache.\nCACHED: {}\nSOURCE: {}\nCACHE LOCATION:{}".format(
        cache_sizes, fresh_sizes, self.cache_path
      ))

  def _fetch_info(self):
    if self.path.protocol != "boss":
      infojson = self._storage.get_file('info')

      if infojson is None:
        raise ValueError(red('No info file was found: {}'.format(self.info_cloudpath)))

      infojson = infojson.decode('utf-8')
      return json.loads(infojson)
    else:
      return self.fetch_boss_info()

  def refreshInfo(self):
    warn("WARNING: refreshInfo is deprecated. Use refresh_info instead.")
    return self.refresh_info()

  def fetch_boss_info(self):
    experiment = ExperimentResource(
      name=self.path.dataset, 
      collection_name=self.path.bucket
    )
    rmt = BossRemote(boss_credentials)
    experiment = rmt.get_project(experiment)

    coord_frame = CoordinateFrameResource(name=experiment.coord_frame)
    coord_frame = rmt.get_project(coord_frame)

    channel = ChannelResource(self.path.layer, self.path.bucket, self.path.dataset)
    channel = rmt.get_project(channel)    

    unit_factors = {
      'nanometers': 1,
      'micrometers': 1e3,
      'millimeters': 1e6,
      'centimeters': 1e7,
    }

    unit_factor = unit_factors[coord_frame.voxel_unit]

    cf = coord_frame
    resolution = [ cf.x_voxel_size, cf.y_voxel_size, cf.z_voxel_size ]
    resolution = [ int(round(_)) * unit_factor for _ in resolution ]

    bbox = Bbox(
      (cf.x_start, cf.y_start, cf.z_start),
      (cf.x_stop, cf.y_stop, cf.z_stop)
    )
    bbox.maxpt = bbox.maxpt 

    layer_type = 'unknown'
    if 'type' in channel.raw:
      layer_type = channel.raw['type']

    info = CloudVolume.create_new_info(
      num_channels=1,
      layer_type=layer_type,
      data_type=channel.datatype,
      encoding='raw',
      resolution=resolution,
      voxel_offset=bbox.minpt,
      volume_size=bbox.size3(),
    )

    each_factor = Vec(2,2,1)
    if experiment.hierarchy_method == 'isotropic':
      each_factor = Vec(2,2,2)
    
    factor = each_factor.clone()
    for mip in range(1, experiment.num_hierarchy_levels):
      self.add_scale(factor, info=info)
      factor *= each_factor

    return info

  def commitInfo(self):
    warn("WARNING: commitInfo is deprecated use commit_info instead.")
    return self.commit_info()

  def commit_info(self):
    if self.path.protocol == 'boss':
      return self 

    infojson = jsonify(self.info, 
      sort_keys=True,
      indent=2, 
      separators=(',', ': ')
    )

    self._storage.put_file('info', infojson, 
      content_type='application/json', 
      cache_control='no-cache'
    ).wait()
    self._maybe_cache_info()
    return self

  def _read_cached_json(self, filename):
      with Storage('file://' + self.cache_path, n_threads=0) as storage:
        jsonfile = storage.get_file(filename)

      if jsonfile:
        jsonfile = jsonfile.decode('utf-8')
        return json.loads(jsonfile)
      else:
        return None

  def _maybe_cache_info(self):
    if self.cache:
      with Storage('file://' + self.cache_path, n_threads=0) as storage:
        storage.put_file('info', jsonify(self.info), 'application/json')

  def refresh_provenance(self):
    if self.cache:
      prov = self._read_cached_json('provenance')
      if prov:
        self.provenance = DataLayerProvenance(**prov)
        return self.provenance

    self.provenance = self._fetch_provenance()
    self._maybe_cache_provenance()
    return self.provenance

  def _cast_provenance(self, prov):
    if isinstance(prov, DataLayerProvenance):
      return prov
    elif isinstance(prov, string_types):
      prov = json.loads(prov)

    provobj = DataLayerProvenance(**prov)
    provobj.sources = provobj.sources or []  
    provobj.owners = provobj.owners or []
    provobj.processing = provobj.processing or []
    provobj.description = provobj.description or ""
    provobj.validate()
    return provobj

  def _fetch_provenance(self):
    if self.path.protocol == 'boss':
      return self.provenance

    if self._storage.exists('provenance'):
      provfile = self._storage.get_file('provenance')
      provfile = provfile.decode('utf-8')

      try:
        provfile = json5.loads(provfile)
      except ValueError:
        raise ValueError(red("""The provenance file could not be JSON decoded. 
          Please reformat the provenance file before continuing. 
          Contents: {}""".format(provfile)))
    else:
      provfile = {
        "sources": [],
        "owners": [],
        "processing": [],
        "description": "",
      }

    return self._cast_provenance(provfile)

  def commit_provenance(self):
    if self.path.protocol == 'boss':
      return self.provenance

    prov = self.provenance.serialize()

    # hack to pretty print provenance files
    prov = json.loads(prov)
    prov = jsonify(prov, 
      sort_keys=True,
      indent=2, 
      separators=(',', ': ')
    )

    self._storage.put_file('provenance', prov, 
      content_type='application/json',
      cache_control='no-cache',
    )
    self._maybe_cache_provenance()
    return self.provenance

  def _maybe_cache_provenance(self):
    if self.cache and self.provenance:
      with Storage('file://' + self.cache_path, n_threads=0) as storage:
        storage.put_file('provenance', self.provenance.serialize(), 'application/json')
    return self

  def _check_cached_provenance_validity(self):
    cached_prov = self._read_cached_json('provenance')
    if not cached_prov:
      return

    cached_prov = self._cast_provenance(cached_prov)
    fresh_prov = self._fetch_provenance()
    if cached_prov != fresh_prov:
      warn("""
      WARNING: Cached provenance file does not match source.

      CACHED: {}
      SOURCE: {}
      """.format(cached_prov.serialize(), fresh_prov.serialize()))

  @property
  def dataset_name(self):
    return self.path.dataset
  
  @property
  def layer(self):
    return self.path.layer

  @property
  def scales(self):
    return self.info['scales']

  @property
  def scale(self):
    return self.mip_scale(self.mip)

  def mip_scale(self, mip):
    return self.info['scales'][mip]

  @property
  def base_cloudpath(self):
    return self.path.protocol + "://" + os.path.join(self.path.bucket, self.path.intermediate_path, self.dataset_name)

  @property
  def layer_cloudpath(self):
    return os.path.join(self.base_cloudpath, self.layer)

  @property
  def info_cloudpath(self):
    return os.path.join(self.layer_cloudpath, 'info')

  @property
  def cache_path(self):
    if type(self.cache) is not str:
      return toabs(os.path.join(CLOUD_VOLUME_DIR, 'cache', 
        self.path.protocol, self.path.bucket.replace('/', ''), self.path.intermediate_path,
        self.path.dataset, self.path.layer
      ))
    else:
      return toabs(self.cache)

  @property
  def shape(self):
    """Returns Vec(x,y,z,channels) shape of the volume similar to numpy.""" 
    return self.mip_shape(self.mip)

  def mip_shape(self, mip):
    size = self.mip_volume_size(mip)
    return Vec(size.x, size.y, size.z, self.num_channels)

  @property
  def volume_size(self):
    """Returns Vec(x,y,z) shape of the volume (i.e. shape - channels) similar to numpy.""" 
    return self.mip_volume_size(self.mip)

  def mip_volume_size(self, mip):
    return Vec(*self.info['scales'][mip]['size'])

  @property
  def available_mips(self):
    """Returns a list of mip levels that are defined."""
    return range(len(self.info['scales']))

  @property
  def layer_type(self):
    """e.g. 'image' or 'segmentation'"""
    return self.info['type']

  @property
  def dtype(self):
    """e.g. 'uint8'"""
    return self.data_type

  @property
  def data_type(self):
    return self.info['data_type']

  @property
  def encoding(self):
    """e.g. 'raw' or 'jpeg'"""
    return self.mip_encoding(self.mip)

  def mip_encoding(self, mip):
    return self.info['scales'][mip]['encoding']

  @property
  def num_channels(self):
    return int(self.info['num_channels'])

  @property
  def voxel_offset(self):
    """Vec(x,y,z) start of the dataset in voxels"""
    return self.mip_voxel_offset(self.mip)

  def mip_voxel_offset(self, mip):
    return Vec(*self.info['scales'][mip]['voxel_offset'])

  @property 
  def resolution(self):
    """Vec(x,y,z) dimensions of each voxel in nanometers"""
    return self.mip_resolution(self.mip)

  def mip_resolution(self, mip):
    return Vec(*self.info['scales'][mip]['resolution'])

  @property
  def downsample_ratio(self):
    """Describes how downsampled the current mip level is as an (x,y,z) factor triple."""
    return self.resolution / self.mip_resolution(0)

  @property
  def underlying(self):
    """Underlying chunk size dimensions in voxels"""
    return self.mip_underlying(self.mip)

  def mip_underlying(self, mip):
    return Vec(*self.info['scales'][mip]['chunk_sizes'][0])

  @property
  def key(self):
    """The subdirectory within the data layer containing the chunks for this mip level"""
    return self.mip_key(self.mip)

  def mip_key(self, mip):
    return self.info['scales'][mip]['key']

  @property
  def bounds(self):
    """Returns a bounding box for the dataset with dimensions in voxels"""
    return self.mip_bounds(self.mip)

  def mip_bounds(self, mip):
    offset = self.mip_voxel_offset(mip)
    shape = self.mip_volume_size(mip)
    return Bbox( offset, offset + shape )

  def slices_to_global_coords(self, slices):
    """
    Used to convert from a higher mip level into mip 0 resolution.
    """
    if type(slices) is Bbox:
      slices = slices.to_slices()

    maxsize = list(self.mip_volume_size(0) + self.mip_voxel_offset(0)) + [ self.num_channels ]
    minsize = list(self.mip_voxel_offset(0)) + [ 0 ]

    slices = generate_slices(slices, minsize, maxsize)[:3]
    lower = Vec(*map(lambda x: x.start, slices), dtype=np.int64)
    upper = Vec(*map(lambda x: x.stop, slices), dtype=np.int64)
    step = Vec(*map(lambda x: x.step, slices), dtype=np.int64)

    dsr = self.downsample_ratio.astype(np.int64)

    lower *= dsr
    upper *= dsr

    signs = step / np.absolute(step)
    step = signs * max2(np.absolute(step * dsr), Vec(1,1,1))
    step = Vec(*np.round(step))

    return [
      slice(lower.x, upper.x, step.x),
      slice(lower.y, upper.y, step.y),
      slice(lower.z, upper.z, step.z)
    ]


  def slices_from_global_coords(self, slices):
    """
    Used for converting from mip 0 coordinates to upper mip level
    coordinates. This is mainly useful for debugging since the neuroglancer
    client displays the mip 0 coordinates for your cursor.
    """

    if type(slices) is Bbox:
      slices = slices.to_slices()

    maxsize = list(self.mip_volume_size(0) + self.mip_voxel_offset(0)) + [ self.num_channels ]
    minsize = list(self.mip_voxel_offset(0)) + [ 0 ]

    slices = generate_slices(slices, minsize, maxsize)[:3]
    lower = Vec(*map(lambda x: x.start, slices), dtype=np.int64)
    upper = Vec(*map(lambda x: x.stop, slices), dtype=np.int64)
    step = Vec(*map(lambda x: x.step, slices), dtype=np.int64)

    dsr = self.downsample_ratio.astype(np.int64)

    lower //= dsr
    upper //= dsr

    signs = step / np.absolute(step)
    step = signs * max2(np.absolute(step / dsr), Vec(1,1,1))
    step = Vec(*np.round(step))

    return [
      slice(lower.x, upper.x, step.x),
      slice(lower.y, upper.y, step.y),
      slice(lower.z, upper.z, step.z)
    ]

  def reset_scales(self):
    """Used for manually resetting downsamples if something messed up."""
    self.info['scales'] = self.info['scales'][0:1]
    return self.commit_info()

  def add_scale(self, factor, info=None):
    """
    Generate a new downsample scale to for the info file and return an updated dictionary.
    You'll still need to call self.commit_info() to make it permenant.

    Required:
      factor: int (x,y,z), e.g. (2,2,1) would represent a reduction of 2x in x and y

    Returns: info dict
    """
    if not info:
      info = self.info

    # e.g. {"encoding": "raw", "chunk_sizes": [[64, 64, 64]], "key": "4_4_40", 
    # "resolution": [4, 4, 40], "voxel_offset": [0, 0, 0], 
    # "size": [2048, 2048, 256]}
    fullres = info['scales'][0]

    # If the voxel_offset is not divisible by the ratio,
    # zooming out will slightly shift the data.
    # Imagine the offset is 10
    #    the mip 1 will have an offset of 5
    #    the mip 2 will have an offset of 2 instead of 2.5 
    #        meaning that it will be half a pixel to the left
    
    chunk_size = lib.find_closest_divisor(fullres['chunk_sizes'][0], closest_to=[64,64,64])

    def downscale(size, roundingfn):
      smaller = Vec(*size, dtype=np.float32) / Vec(*factor)
      return list(map(int, roundingfn(smaller)))

    newscale = {
      u"encoding": fullres['encoding'],
      u"chunk_sizes": [ list(map(int, chunk_size)) ],
      u"resolution": list(map(int, Vec(*fullres['resolution']) * factor )),
      u"voxel_offset": downscale(fullres['voxel_offset'], np.floor),
      u"size": downscale(fullres['size'], np.ceil),
    }

    newscale[u'key'] = str("_".join([ str(res) for res in newscale['resolution']]))

    new_res = np.array(newscale['resolution'], dtype=int)

    preexisting = False
    for index, scale in enumerate(info['scales']):
      res = np.array(scale['resolution'], dtype=int)
      if np.array_equal(new_res, res):
        preexisting = True
        info['scales'][index] = newscale
        break

    if not preexisting:    
      info['scales'].append(newscale)

    return newscale

  def __interpret_slices(self, slices):
    """
    Convert python slice objects into a more useful and computable form:

    - requested_bbox: A bounding box representing the volume requested
    - steps: the requested stride over x,y,z
    - channel_slice: A python slice object over the channel dimension

    Returned as a tuple: (requested_bbox, steps, channel_slice)
    """
    maxsize = list(self.bounds.maxpt) + [ self.num_channels ]
    minsize = list(self.bounds.minpt) + [ 0 ]

    slices = generate_slices(slices, minsize, maxsize, bounded=self.bounded)
    channel_slice = slices.pop()

    minpt = Vec(*[ slc.start for slc in slices ])
    maxpt = Vec(*[ slc.stop for slc in slices ]) 
    steps = Vec(*[ slc.step for slc in slices ])

    return Bbox(minpt, maxpt), steps, channel_slice

  def __realized_bbox(self, requested_bbox):
    """
    The requested bbox might not be aligned to the underlying chunk grid 
    or even outside the bounds of the dataset. Convert the request into
    a bbox representing something that can be actually downloaded.

    Returns: Bbox
    """
    realized_bbox = requested_bbox.expand_to_chunk_size(self.underlying, offset=self.voxel_offset)
    return Bbox.clamp(realized_bbox, self.bounds)

  def exists(self, bbox_or_slices):
    """
    Produce a summary of whether all the requested chunks exist.

    bbox_or_slices: accepts either a Bbox or a tuple of slices representing
      the requested volume. 
    Returns: { chunk_file_name: boolean, ... }
    """
    if type(bbox_or_slices) is Bbox:
      requested_bbox = bbox_or_slices
    else:
      (requested_bbox, steps, channel_slice) = self.__interpret_slices(bbox_or_slices)
    realized_bbox = self.__realized_bbox(requested_bbox)
    cloudpaths = self.__chunknames(realized_bbox, self.bounds, self.key, self.underlying)

    with Storage(self.layer_cloudpath, progress=self.progress) as storage:
      existence_report = storage.files_exist(cloudpaths)
    return existence_report

  def delete(self, bbox_or_slices):
    """
    Delete the files within the bounding box.

    bbox_or_slices: accepts either a Bbox or a tuple of slices representing
      the requested volume. 
    """
    if type(bbox_or_slices) is Bbox:
      requested_bbox = bbox_or_slices
    else:
      (requested_bbox, steps, channel_slice) = self.__interpret_slices(bbox_or_slices)
    realized_bbox = self.__realized_bbox(requested_bbox)

    if requested_bbox != realized_bbox:
      raise ValueError("Unable to delete non-chunk aligned bounding boxes. Requested: {}, Realized: {}".format(
        requested_bbox, realized_bbox
      ))

    cloudpaths = self.__chunknames(realized_bbox, self.bounds, self.key, self.underlying)

    with Storage(self.layer_cloudpath, progress=self.progress) as storage:
      storage.delete_files(cloudpaths)

    if self.cache:
      with Storage('file://' + self.cache_path, progress=self.progress) as storage:
        storage.delete_files(cloudpaths)

  def __getitem__(self, slices):
    (requested_bbox, steps, channel_slice) = self.__interpret_slices(slices)

    if self.autocrop:
      requested_bbox = Bbox.intersection(requested_bbox, self.bounds)
    
    if self.path.protocol != 'boss':
      return self._cutout(requested_bbox, steps, channel_slice)
    
    cutout = self._boss_cutout(requested_bbox, steps, channel_slice)      

    if self.bounded or self.autocrop:
      return cutout
    elif cutout.bounds == requested_bbox:
      return cutout

    # This section below covers the case where the requested volume is bigger
    # than the dataset volume and the bounds guards have been switched 
    # off. This is useful for Marching Cubes where a 1px excess boundary
    # is needed.
    shape = list(requested_bbox.size3()) + [ cutout.shape[3] ]
    renderbuffer = np.zeros(shape=shape, dtype=self.dtype)
    lp = cutout.bounds.minpt - requested_bbox.minpt
    hp = lp + cutout.bounds.size3()
    renderbuffer[ lp.x:hp.x, lp.y:hp.y, lp.z:hp.z, : ] = cutout 
    return VolumeCutout.from_volume(self, renderbuffer, requested_bbox)

  def flush_cache(self):
    if os.path.exists(self.cache_path):
        shutil.rmtree(self.cache_path) 

  def _content_type(self):
    if self.encoding == 'jpeg':
      return 'image/jpeg'
    return 'application/octet-stream'

  def _should_compress(self):
    if self.compress is None:
      return 'gzip' if self.encoding in ('raw', 'compressed_segmentation') else None
    elif self.compress == True:
      return 'gzip'
    elif self.compress == False:
      return None
    else:
      return self.compress

  def _compute_data_locations(self, cloudpaths):
    if not self.cache:
      return { 'local': [], 'remote': cloudpaths }

    def noextensions(fnames):
      return [ os.path.splitext(fname)[0] for fname in fnames ]

    list_dir = mkdir(os.path.join(self.cache_path, self.key))
    filenames = noextensions(os.listdir(list_dir))

    basepathmap = { os.path.basename(path): os.path.dirname(path) for path in cloudpaths }

    # check which files are already cached, we only want to download ones not in cache
    requested = set([ os.path.basename(path) for path in cloudpaths ])
    already_have = requested.intersection(set(filenames))
    to_download = requested.difference(already_have)

    download_paths = [ os.path.join(basepathmap[fname], fname) for fname in to_download ]    

    return { 'local': already_have, 'remote': download_paths }

  def _cutout(self, requested_bbox, steps, channel_slice=slice(None)):
    cloudpath_bbox = requested_bbox.expand_to_chunk_size(self.underlying, offset=self.voxel_offset)
    cloudpath_bbox = Bbox.clamp(cloudpath_bbox, self.bounds)
    cloudpaths = self.__chunknames(cloudpath_bbox, self.bounds, self.key, self.underlying)

    def multichannel_shape(bbox):
      shape = bbox.size3()
      return (shape[0], shape[1], shape[2], self.num_channels)

    renderbuffer = np.zeros(shape=multichannel_shape(requested_bbox), dtype=self.dtype)

    def decode(filename, content):
      bbox = Bbox.from_filename(filename)
      content_len = len(content) if content is not None else 0

      if not content:
        if self.fill_missing:
          content = ''
        else:
          raise EmptyVolumeException(filename)

      try:
        return chunks.decode(
          content, self.encoding, multichannel_shape(bbox), self.dtype
        )
      except Exception as error:
        print(red('File Read Error: {} bytes, {}, {}, errors: {}'.format(
            content_len, bbox, filename, error)))
        raise

    ZERO3 = Vec(0,0,0)

    def paint(filename, content):
        bbox = Bbox.from_filename(filename) # possible off by one error w/ exclusive bounds
        img3d = decode(filename, content)
        
        if not Bbox.intersects(requested_bbox, bbox):
          return

        spt = max2(bbox.minpt, requested_bbox.minpt)
        ept = min2(bbox.maxpt, requested_bbox.maxpt)

        istart = max2(spt - bbox.minpt, ZERO3)
        iend = min2(ept - bbox.maxpt, ZERO3) + img3d.shape[:3]

        rbox = Bbox(spt, ept) - requested_bbox.minpt
        renderbuffer[ rbox.to_slices() ] = img3d[ istart.x:iend.x, istart.y:iend.y, istart.z:iend.z, : ]

    def download(cloudpath, filename, cache, iface):
      content = SimpleStorage(cloudpath).get_file(filename)
      paint(filename, content)
      if cache:
        content = content or b''
        SimpleStorage('file://' + self.cache_path).put_file(
          file_path=filename, 
          content=content, 
          content_type=self._content_type(), 
          compress=self._should_compress()
        )

    locations = self._compute_data_locations(cloudpaths)
    cachedir = 'file://' + os.path.join(self.cache_path, self.key)
    progress = 'Downloading' if self.progress else None

    with ThreadedQueue(n_threads=20, progress=progress) as tq:
      for filename in locations['local']:
        dl = partial(download, cachedir, filename, False)
        tq.put(dl)
      for filename in locations['remote']:
        dl = partial(download, self.layer_cloudpath, filename, self.cache)
        tq.put(dl)

    renderbuffer = renderbuffer[ ::steps.x, ::steps.y, ::steps.z, channel_slice ]
    return VolumeCutout.from_volume(self, renderbuffer, requested_bbox)
  
  def _boss_cutout(self, requested_bbox, steps, channel_slice=slice(None)):
    bounds = Bbox.clamp(requested_bbox, self.bounds)
    
    if bounds.volume() < 1:
      raise ValueError('Requested less than one pixel of volume. {}'.format(bounds))

    x_rng = [ bounds.minpt.x, bounds.maxpt.x ]
    y_rng = [ bounds.minpt.y, bounds.maxpt.y ]
    z_rng = [ bounds.minpt.z, bounds.maxpt.z ]

    layer_type = 'image' if self.layer_type == 'unknown' else self.layer_type

    chan = ChannelResource(
      collection_name=self.path.bucket, 
      experiment_name=self.path.dataset, 
      name=self.path.layer, # Channel
      type=layer_type, 
      datatype=self.dtype,
    )

    rmt = BossRemote(boss_credentials)
    cutout = rmt.get_cutout(chan, self.mip, x_rng, y_rng, z_rng).T
    cutout = cutout.astype(self.dtype)
    cutout = cutout[::steps.x, ::steps.y, ::steps.z]

    if len(cutout.shape) == 3:
      cutout = cutout.reshape(tuple(list(cutout.shape) + [ 1 ]))

    return VolumeCutout.from_volume(self, cutout, bounds)

  def __setitem__(self, slices, img):
    imgshape = list(img.shape)
    if len(imgshape) == 3:
      imgshape = imgshape + [ self.num_channels ]

    maxsize = list(self.bounds.maxpt) + [ self.num_channels ]
    minsize = list(self.bounds.minpt) + [ 0 ]
    slices = generate_slices(slices, minsize, maxsize, bounded=self.bounded)
    bbox = Bbox.from_slices(slices)

    slice_shape = list(bbox.size3()) + [ slices[3].stop - slices[3].start ]

    if not np.array_equal(imgshape, slice_shape):
      raise ValueError("Illegal slicing, Image shape: {} != {} Slice Shape".format(imgshape, slice_shape))

    if self.autocrop:
      if not self.bounds.contains_bbox(bbox):
        cropped_bbox = Bbox.intersection(bbox, self.bounds)
        dmin = np.absolute(bbox.minpt - cropped_bbox.minpt)
        dmax = dmin + cropped_bbox.size3()
        img = img[ dmin.x:dmax.x, dmin.y:dmax.y, dmin.z:dmax.z ] 
        bbox = cropped_bbox

    if bbox.volume() < 1:
      return

    if self.path.protocol == 'boss':
      self.upload_boss_image(img, bbox.minpt)
    else:
      self.upload_image(img, bbox.minpt)

  def upload_boss_image(self, img, offset):
    shape = Vec(*img.shape[:3])
    offset = Vec(*offset)

    bounds = Bbox(offset, shape + offset)

    if bounds.volume() < 1:
      raise EmptyRequestException('Requested less than one pixel of volume. {}'.format(bounds))

    x_rng = [ bounds.minpt.x, bounds.maxpt.x ]
    y_rng = [ bounds.minpt.y, bounds.maxpt.y ]
    z_rng = [ bounds.minpt.z, bounds.maxpt.z ]

    layer_type = 'image' if self.layer_type == 'unknown' else self.layer_type

    chan = ChannelResource(
      collection_name=self.path.bucket, 
      experiment_name=self.path.dataset, 
      name=self.path.layer, # Channel
      type=layer_type, 
      datatype=self.dtype,
    )

    if img.shape[3] == 1:
      img = img.reshape( img.shape[:3] )

    rmt = BossRemote(boss_credentials)
    img = img.T
    img = np.ascontiguousarray(img.astype(self.dtype))

    rmt.create_cutout(chan, self.mip, x_rng, y_rng, z_rng, img)

  def upload_image(self, img, offset):
    if self.path.protocol == 'boss':
      raise NotImplementedError

    if str(self.dtype) != str(img.dtype):
      raise ValueError('The uploaded image data type must match the volume data type. volume: {}, image: {}'.format(self.dtype, img.dtype))

    iterator = tqdm(self._generate_chunks(img, offset), desc='Rechunking image', disable=(not self.progress))

    if self.cache:
        mkdir(self.cache_path)
        if self.progress:
          print("Caching upload...")
        cachestorage = Storage('file://' + self.cache_path, progress=self.progress)

    cloudstorage = Storage(self.layer_cloudpath, progress=self.progress)
    for imgchunk, spt, ept in iterator:
      if np.array_equal(spt, ept):
          continue

      # handle the edge of the dataset
      clamp_ept = min2(ept, self.bounds.maxpt)
      newept = clamp_ept - spt
      imgchunk = imgchunk[ :newept.x, :newept.y, :newept.z, : ]

      filename = "{}-{}_{}-{}_{}-{}".format(
        spt.x, clamp_ept.x,
        spt.y, clamp_ept.y, 
        spt.z, clamp_ept.z
      )

      cloudpath = os.path.join(self.key, filename)
      encoded = chunks.encode(imgchunk, self.encoding)

      cloudstorage.put_file(
        file_path=cloudpath, 
        content=encoded,
        content_type=self._content_type(), 
        compress=self._should_compress(),
        cache_control=self._cdn_cache_control(self.cdn_cache),
      )

      if self.cache:
        cachestorage.put_file(
          file_path=cloudpath,
          content=encoded, 
          content_type=self._content_type(), 
          compress=self._should_compress()
        )

    desc = 'Uploading' if self.progress else None
    cloudstorage.wait(desc)
    cloudstorage.kill_threads()
    
    if self.cache:
      desc = 'Caching' if self.progress else None
      cachestorage.wait(desc)
      cachestorage.kill_threads()

  def _cdn_cache_control(self, val=None):
    """Translate self.cdn_cache into a Cache-Control HTTP header."""
    if val is None:
      return 'max-age=3600, s-max-age=3600'
    elif type(val) is str:
      return val
    elif type(val) is bool:
      if val:
        return 'max-age=3600, s-max-age=3600'
      else:
        return 'no-cache'
    elif type(val) is int:
      if val < 0:
        raise ValueError('cdn_cache must be a positive integer, boolean, or string. Got: ' + str(val))

      if val == 0:
        return 'no-cache'
      else:
        return 'max-age={}, s-max-age={}'.format(val, val)
    else:
      raise NotImplementedError(type(val) + ' is not a supported cache_control setting.')

  def _generate_chunks(self, img, offset):
    shape = Vec(*img.shape)[:3]
    offset = Vec(*offset)[:3]

    bounds = Bbox( offset, shape + offset)

    alignment_check = bounds.round_to_chunk_size(self.underlying, self.voxel_offset)
    alignment_check = Bbox.clamp(alignment_check, self.bounds)

    if not np.all(alignment_check.minpt == bounds.minpt) or not np.all(alignment_check.maxpt == bounds.maxpt):
      raise ValueError('Only chunk aligned writes are currently supported. Got: {}, Volume Offset: {}, Alignment Check: {}'.format(
        bounds, self.voxel_offset, alignment_check)
      )

    bounds = Bbox.clamp(bounds, self.bounds)

    img_offset = bounds.minpt - offset
    img_end = Vec.clamp(bounds.size3() + img_offset, Vec(0,0,0), shape)

    if len(img.shape) == 3:
      img = img[:, :, :, np.newaxis ]
  
    for startpt in xyzrange( img_offset, img_end, self.underlying ):
      endpt = min2(startpt + self.underlying, shape)
      chunkimg = img[ startpt.x:endpt.x, startpt.y:endpt.y, startpt.z:endpt.z, : ]

      spt = (startpt + bounds.minpt).astype(int)
      ept = (endpt + bounds.minpt).astype(int)
    
      yield chunkimg, spt, ept 

  def __chunknames(self, bbox, volume_bbox, key, chunk_size):
    paths = []

    for x,y,z in xyzrange( bbox.minpt, bbox.maxpt, chunk_size ):
      highpt = min2(Vec(x,y,z) + chunk_size, volume_bbox.maxpt)
      filename = "{}-{}_{}-{}_{}-{}".format(
        x, highpt.x,
        y, highpt.y, 
        z, highpt.z
      )
      paths.append( os.path.join(key, filename) )

    return paths

  def save_mesh(self, *args, **kwargs):
    warn("WARNING: vol.save_mesh is deprecated. Please use vol.mesh.save(...) instead.")
    self.mesh.save(*args, **kwargs)
    

def generate_slices(slices, minsize, maxsize, bounded=True):
  """Assisting function for __getitem__. e.g. vol[:,:,:,:]"""

  if isinstance(slices, integer_types) or isinstance(slices, float):
    slices = [ slice(int(slices), int(slices)+1, 1) ]
  if type(slices) == slice:
    slices = [ slices ]

  slices = list(slices)

  while len(slices) < len(maxsize):
    slices.append( slice(None, None, None) )

  # First three slices are x,y,z, last is channel. 
  # Handle only x,y,z here, channel seperately
  for index, slc in enumerate(slices):
    if isinstance(slc, integer_types) or isinstance(slc, float):
      slices[index] = slice(int(slc), int(slc)+1, 1)
    else:
      start = minsize[index] if slc.start is None else slc.start
      end = maxsize[index] if slc.stop is None else slc.stop 
      step = 1 if slc.step is None else slc.step

      if step < 0:
        raise ValueError('Negative step sizes are not supported. Got: {}'.format(step))

      # note: when unbounded, negative indicies do not refer to
      # the end of the volume as they can describe, e.g. a 1px
      # border on the edge of the beginning of the dataset as in
      # marching cubes.
      if bounded:
        # if start < 0: # this is support for negative indicies
          # start = maxsize[index] + start         
        check_bounds(start, minsize[index], maxsize[index])
        # if end < 0: # this is support for negative indicies
        #   end = maxsize[index] + end
        check_bounds(end, minsize[index], maxsize[index])

      slices[index] = slice(start, end, step)

  return slices

class VolumeCutout(np.ndarray):

  def __new__(cls, buf, dataset_name, layer, mip, layer_type, bounds, *args, **kwargs):
    return super(VolumeCutout, cls).__new__(cls, shape=buf.shape, buffer=np.ascontiguousarray(buf), dtype=buf.dtype)

  def __init__(self, buf, dataset_name, layer, mip, layer_type, bounds, *args, **kwargs):
    super(VolumeCutout, self).__init__()
    
    self.dataset_name = dataset_name
    self.layer = layer
    self.mip = mip
    self.layer_type = layer_type
    self.bounds = bounds

  @classmethod
  def from_volume(cls, volume, buf, bounds):
    return VolumeCutout(
      buf=buf,
      dataset_name=volume.dataset_name,
      layer=volume.layer,
      mip=volume.mip,
      layer_type=volume.layer_type,
      bounds=bounds,
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

