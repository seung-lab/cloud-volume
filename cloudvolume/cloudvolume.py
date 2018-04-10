from __future__ import print_function

from functools import partial
import json
import json5
import os
import re
import sys
import uuid
import weakref

from six.moves import range
import numpy as np
from tqdm import tqdm
from six import string_types
import multiprocessing as mp

from intern.remote.boss import BossRemote
from intern.resource.boss.resource import ChannelResource, ExperimentResource, CoordinateFrameResource
from .secrets import boss_credentials, CLOUD_VOLUME_DIR

from . import lib, chunks
from .cacheservice import CacheService
from .lib import ( 
  toabs, colorize, red, yellow, 
  mkdir, clamp, xyzrange, Vec, 
  Bbox, min2, max2, check_bounds, 
  jsonify, generate_slices
)
from .meshservice import PrecomputedMeshService
from .provenance import DataLayerProvenance
from .storage import SimpleStorage, Storage
from . import txrx
from .volumecutout import VolumeCutout
from . import sharedmemory

# Set the interpreter bool
try:
    INTERACTIVE = bool(sys.ps1)
except AttributeError:
    INTERACTIVE = bool(sys.flags.interactive)

def warn(text):
  print(colorize('yellow', text))

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
    parallel (int: 1, bool): number of extra processes to launch, 1 means only use the main process. If parallel is True
      use the number of CPUs returned by multiprocessing.cpu_count()
    output_to_shared_memory (bool: False, str): Write results to shared memory. Don't make copies from this buffer
      and don't automatically unlink it. If a string is provided, use that shared memory location rather than
      the default.
    provenance: (string, dict, or object) in lieu of fetching a neuroglancer provenance file, use this provided one.
            This is useful when doing multiprocessing.
    progress: (bool) Show tqdm progress bars. 
        Defaults True in interactive python, False in script execution mode.
    compress: (bool, str, None) pick which compression method to use. 
      None: (default) Use the pre-programmed recommendation (e.g. gzip raw arrays and compressed_segmentation)
      bool: True=gzip, False=no compression, Overrides defaults
      str: 'gzip', extension so that we can add additional methods in the future like lz4 or zstd. 
        '' means no compression (same as False).
    non_aligned_writes: (bool) Enable non-aligned writes. Not multiprocessing safe without careful design.
      When not enabled, a ValueError is thrown for non-aligned writes.
  """
  def __init__(self, cloudpath, mip=0, bounded=True, autocrop=False, fill_missing=False, 
      cache=False, cdn_cache=True, progress=INTERACTIVE, info=None, provenance=None, 
      compress=None, non_aligned_writes=False, parallel=1, output_to_shared_memory=False):

    self.autocrop = bool(autocrop)
    self.bounded = bool(bounded)
    self.cdn_cache = cdn_cache
    self.compress = compress
    self.fill_missing = bool(fill_missing)
    self.mip = int(mip)
    self.non_aligned_writes = bool(non_aligned_writes)
    self.progress = bool(progress)
    self.path = lib.extract_path(cloudpath)
    self.shared_memory_id = self.generate_shared_memory_location()
    if type(output_to_shared_memory) == str:
      self.shared_memory_id = str(output_to_shared_memory)
    self.output_to_shared_memory = bool(output_to_shared_memory)

    if type(parallel) == bool:
      self.parallel = mp.cpu_count() if parallel == True else 1
    else:
      self.parallel = int(parallel)
    
    if self.parallel <= 0:
      raise ValueError('Number of processes must be >= 1. Got: ' + str(self.parallel))

    self.init_submodules(cache)

    if info is None:
      self.refresh_info()
      if self.cache.enabled:
        self.cache.check_info_validity()
    else:
      self.info = info

    if provenance is None:
      self.provenance = None
      self.refresh_provenance()
      self.cache.check_provenance_validity()
    else:
      self.provenance = self._cast_provenance(provenance)

    try:
      self.mip = self.available_mips[self.mip]
    except:
      raise Exception("MIP {} has not been generated.".format(self.mip))

  def init_submodules(self, cache):
    self.cache = CacheService(cache, weakref.proxy(self)) 
    self.mesh = PrecomputedMeshService(weakref.proxy(self)) 

  def generate_shared_memory_location(self):
    return 'cloudvolume-shm-' + str(uuid.uuid4())

  def unlink_shared_memory(self):
    return sharedmemory.unlink(self.shared_memory_id)

  @property
  def _storage(self):
    if self.path.protocol == 'boss':
      return None
    
    try:
      return SimpleStorage(self.layer_cloudpath)
    except:
      if self.path.layer == 'info':
        warn("WARNING: Your layer is named 'info', is that what you meant? {}".format(
            self.path
        ))
      raise
      
  @classmethod
  def create_new_info(cls, num_channels, layer_type, data_type, encoding, resolution, voxel_offset, volume_size, mesh=None, chunk_size=(64,64,64)):
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
    if self.cache.enabled:
      info = self.cache.get_json('info')
      if info:
        self.info = info
        return self.info

    self.info = self._fetch_info()
    self.cache.maybe_cache_info()
    return self.info

  def _fetch_info(self):
    if self.path.protocol == "boss":
      return self.fetch_boss_info()
    
    with self._storage as stor:
      info = stor.get_json('info')

    if info is None:
      raise ValueError(red('No info file was found: {}'.format(self.info_cloudpath)))
    return info

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

    with self._storage as stor:
      stor.put_file('info', infojson, 
        content_type='application/json', 
        cache_control='no-cache'
      )
    self.cache.maybe_cache_info()
    return self

  def refresh_provenance(self):
    if self.cache.enabled:
      prov = self.cache.get_json('provenance')
      if prov:
        self.provenance = DataLayerProvenance(**prov)
        return self.provenance

    self.provenance = self._fetch_provenance()
    self.cache.maybe_cache_provenance()
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

    with self._storage as stor:
      if stor.exists('provenance'):
        provfile = stor.get_file('provenance')
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

    with self._storage as stor:
      stor.put_file('provenance', prov, 
        content_type='application/json',
        cache_control='no-cache',
      )
    self.cache.maybe_cache_provenance()
    return self.provenance

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
    return self.cache.path

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
    cloudpaths = txrx.chunknames(realized_bbox, self.bounds, self.key, self.underlying)

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

    cloudpaths = txrx.chunknames(realized_bbox, self.bounds, self.key, self.underlying)

    with Storage(self.layer_cloudpath, progress=self.progress) as storage:
      storage.delete_files(cloudpaths)

    if self.cache.enabled:
      with Storage('file://' + self.cache.path, progress=self.progress) as storage:
        storage.delete_files(cloudpaths)

  def __getitem__(self, slices):
    (requested_bbox, steps, channel_slice) = self.__interpret_slices(slices)

    if self.autocrop:
      requested_bbox = Bbox.intersection(requested_bbox, self.bounds)
    
    if self.path.protocol != 'boss':
      return txrx.cutout(self, requested_bbox, steps, channel_slice, parallel=self.parallel)
    
    return self._boss_cutout(requested_bbox, steps, channel_slice)

  def flush_cache(self, preserve=None):
    """See vol.cache.flush"""
    warn("CloudVolume.flush_cache(...) is deprecated. Please use CloudVolume.cache.flush(...) instead.")
    return self.cache.flush(preserve=preserve)

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

    if self.bounded or self.autocrop or bounds == requested_bbox:
      return VolumeCutout.from_volume(self, cutout, bounds)

    # This section below covers the case where the requested volume is bigger
    # than the dataset volume and the bounds guards have been switched 
    # off. This is useful for Marching Cubes where a 1px excess boundary
    # is needed.
    shape = list(requested_bbox.size3()) + [ cutout.shape[3] ]
    renderbuffer = np.zeros(shape=shape, dtype=self.dtype)
    txrx.shade(renderbuffer, requested_bbox, cutout, bounds)
    return VolumeCutout.from_volume(self, renderbuffer, requested_bbox)

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
      txrx.upload_image(self, img, bbox.minpt)

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

  def save_mesh(self, *args, **kwargs):
    warn("WARNING: vol.save_mesh is deprecated. Please use vol.mesh.save(...) instead.")
    self.mesh.save(*args, **kwargs)
    


