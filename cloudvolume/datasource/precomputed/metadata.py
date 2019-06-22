import json
import os

import json5
import multiprocessing as mp
import numpy as np
from six import string_types
from six.moves import range
from tqdm import tqdm

from ...provenance import DatasetProvenance, DataLayerProvenance
from ...storage import SimpleStorage

from ...lib import ( 
  extract_path, colorize, red, mkdir, Vec, Bbox,  
  jsonify, generate_slices,
)

class PrecomputedMetadata(object):
  """
  The PrecomputedMetadataService provides methods for fetching,
  saving, and accessing information about the data type & compression, 
  bounding box, resolution, and provenance of a given dataset 
  stored in Precomputed format.  

  This class is a building block for building a class that can
  read and write Precomputed data types.
  """
  def __init__(self, cloudpath, cache=None, info=None, provenance=None):
    self.path = extract_path(cloudpath)
    self.cache = cache
    if self.cache:
      self.cache.meta = self
    self.info = None

    if info is None:
      self.refresh_info()
      if self.cache and self.cache.enabled:
        self.cache.check_info_validity()
    else:
      self.info = info

    if provenance is None:
      self.provenance = None
      self.refresh_provenance()
      if self.cache:
        self.cache.check_provenance_validity()
    else:
      self.provenance = self._cast_provenance(provenance)

  def refresh_info(self):
    """
    Refresh the current info file from the cache (if enabled) 
    or primary storage (e.g. the cloud) if not cached.

    Raises cloudvolume.exceptions.InfoUnavailableError when the info file 
    is unable to be retrieved.

    See also: fetch_info

    Returns: dict
    """
    if self.cache and self.cache.enabled:
      info = self.cache.get_json('info')
      if info:
        self.info = info
        return self.info

    self.info = self.fetch_info()

    if self.cache:
      self.cache.maybe_cache_info()
    return self.info

  def fetch_info(self):
    """
    Refresh the current info file from primary storage (e.g. the cloud) without
    refrence to the cache. The cache will not be updated.
  
    Raises cloudvolume.exceptions.InfoUnavailableError when the info file 
    is unable to be retrieved.

    See also: refresh_info

    Returns: dict
    """
    with SimpleStorage(self.cloudpath) as stor:
      info = stor.get_json('info')

    if info is None:
      raise exceptions.InfoUnavailableError(
        red('No info file was found: {}'.format(self.info_cloudpath))
      )
    return info

  def commit_info(self):
    """
    Save the current info dict as JSON into cache and primary storage.

    Raises KeyError if an encoding of 'compressed_segmentation' is specified
    without 'compressed_segmentation_block_size'.

    Raises ValueError if 'compressed_segmentation' is specified and the 
    data type is not uint32 or uint64.
    """
    for scale in self.scales:
      if scale['encoding'] == 'compressed_segmentation':
        if 'compressed_segmentation_block_size' not in scale.keys():
          raise KeyError("""
            'compressed_segmentation_block_size' must be set if 
            compressed_segmentation is set as the encoding.

            A typical value for compressed_segmentation_block_size is (8,8,8)

            Info file specification:
            https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/README.md#info-json-file-specification
          """)
        elif self.data_type not in ('uint32', 'uint64'):
          raise ValueError("compressed_segmentation can only be used with uint32 and uint64 data types.")

    infojson = jsonify(self.info, 
      sort_keys=True,
      indent=2, 
      separators=(',', ': ')
    )

    with SimpleStorage(self.cloudpath) as stor:
      stor.put_file('info', infojson, 
        content_type='application/json', 
        cache_control='no-cache'
      )

    if self.cache:
      self.cache.maybe_cache_info()

  def refresh_provenance(self):
    """
    Refresh the current irovenance file from the cache (if enabled) 
    or primary storage (e.g. the cloud) if not cached. If the provenance
    file does not exist, no error is raised and None is returned.

    Raises ValueError if the provenance file is present but can not
    be json5 decoded.

    See also: fetch_provenance

    Returns: dict or None
    """
    if self.cache and self.cache.enabled:
      prov = self.cache.get_json('provenance')
      if prov:
        self.provenance = DataLayerProvenance(**prov)
        return self.provenance

    self.provenance = self.fetch_provenance()
    if self.cache:
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

  def fetch_provenance(self):
    """
    Refresh the current provenance file from primary storage (e.g. the cloud)
    without reference to cache. The cache will not be updated.
  
    Raises cloudvolume.exceptions.provenanceUnavailableError when the info file 
    is unable to be retrieved.

    See also: refresh_provenance

    Returns: dict
    """
    with SimpleStorage(self.cloudpath) as stor:
      provfile = stor.get_file('provenance')
      if provfile:
        provfile = provfile.decode('utf-8')

        # The json5 decoder is *very* slow
        # so use the stricter but much faster json 
        # decoder first, and try it only if it fails.
        try:
          provfile = json.loads(provfile)
        except json.decoder.JSONDecodeError:
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
    """
    Save the current provenance object as JSON into cache (if enabled) 
    and primary storage.
    """
    prov = self.provenance.serialize()

    # hack to pretty print provenance files
    prov = json.loads(prov)
    prov = jsonify(prov, 
      sort_keys=True,
      indent=2, 
      separators=(',', ': ')
    )

    with SimpleStorage(self.cloudpath) as stor:
      stor.put_file('provenance', prov, 
        content_type='application/json',
        cache_control='no-cache',
      )
    if self.cache:
      self.cache.maybe_cache_provenance()

  @property
  def dataset(self):
    return self.path.dataset
  
  @property
  def layer(self):
    return self.path.layer

  @property
  def scales(self):
    return self.info['scales']

  @scales.setter
  def scales(self, val):
    self.info['scales'] = val

  def scale(self, mip):
    return self.info['scales'][mip]

  @property
  def basepath(self):
    return os.path.join(self.path.bucket, self.path.intermediate_path, self.dataset)

  @property 
  def layerpath(self):
    return os.path.join(self.basepath, self.layer)

  @property
  def base_cloudpath(self):
    return self.path.protocol + "://" + self.basepath

  @property 
  def cloudpath(self):
    return os.path.join(self.base_cloudpath, self.layer)
  
  @property
  def infopath(self):
    return os.path.join(self.layer_cloudpath, 'info')

  def shape(self, mip):
    """Returns Vec(x,y,z,channels) shape of the volume similar to numpy.""" 
    size = self.volume_size(mip)
    return Vec(size.x, size.y, size.z, self.num_channels)

  def volume_size(self, mip):
    """Returns Vec(x,y,z) shape of the volume (i.e. shape - channels).""" 
    return Vec(*self.info['scales'][mip]['size'])

  @property
  def available_mips(self):
    """Returns a list of mip levels that are defined."""
    return range(len(self.info['scales']))

  @property
  def available_resolutions(self):
    """Returns a list of defined resolutions."""
    return (s["resolution"] for s in self.scales)

  @property
  def layer_type(self):
    """e.g. 'image' or 'segmentation'"""
    return self.info['type']

  @property
  def dtype(self):
    """e.g. np.uint8"""
    return np.dtype(self.data_type)

  @property
  def data_type(self):
    """e.g. 'uint8'"""
    return self.info['data_type']

  def encoding(self, mip):
    """e.g. 'raw' or 'jpeg'"""
    return self.info['scales'][mip]['encoding']

  def compressed_segmentation_block_size(self, mip):
    if 'compressed_segmentation_block_size' in self.info['scales'][mip]:
      return self.info['scales'][mip]['compressed_segmentation_block_size']
    return None

  @property
  def num_channels(self):
    return int(self.info['num_channels'])

  def voxel_offset(self, mip):
    """Vec(x,y,z) start of the dataset in voxels"""
    return Vec(*self.info['scales'][mip]['voxel_offset'])

  def resolution(self, mip):
    """Vec(x,y,z) dimensions of each voxel in nanometers"""
    return Vec(*self.info['scales'][mip]['resolution'])

  def downsample_ratio(self, mip):
    """Describes how downsampled the current mip level is as an (x,y,z) factor triple."""
    return self.resolution(mip) / self.resolution(0)

  def chunk_size(self, mip):
    """Underlying chunk size dimensions in voxels. Synonym for underlying."""
    return Vec(*self.info['scales'][mip]['chunk_sizes'][0])

  def key(self, mip):
    """The subdirectory within the data layer containing the chunks for this mip level"""
    return self.info['scales'][mip]['key']

  def bounds(self, mip):
    """Returns a 3D spatial bounding box for the dataset with dimensions in voxels."""
    offset = self.voxel_offset(mip)
    shape = self.volume_size(mip)
    return Bbox( offset, offset + shape )

  def bbox(self, mip):
    bounds = self.bounds(mip)
    minpt = list(bounds.minpt) + [ 0 ]
    maxpt = list(bounds.maxpt) + [ self.num_channels ]
    return Bbox(minpt, maxpt)

  def point_to_mip(self, pt, mip, to_mip):
    pt = Vec(*pt)
    downsample_ratio = self.resolution(mip).astype(np.float32) / self.resolution(to_mip).astype(np.float32)
    return np.floor(pt * downsample_ratio)

  def bbox_to_mip(self, bbox, mip, to_mip):
    """Convert bbox or slices from one mip level to another."""
    if not type(bbox) is Bbox:
      bbox = lib.generate_slices(
        bbox, 
        self.bounds(mip).minpt, 
        self.bounds(mip).maxpt, 
        bounded=False
      )
      bbox = Bbox.from_slices(bbox)

    def one_level(bbox, mip, to_mip):
      original_dtype = bbox.dtype
      # setting type required for Python2
      downsample_ratio = self.resolution(mip).astype(np.float32) / self.resolution(to_mip).astype(np.float32)
      bbox = bbox.astype(np.float64)
      bbox *= downsample_ratio
      bbox.minpt = np.floor(bbox.minpt)
      bbox.maxpt = np.ceil(bbox.maxpt)
      bbox = bbox.astype(original_dtype)
      return bbox

    delta = 1 if to_mip >= mip else -1
    while (mip != to_mip):
      bbox = one_level(bbox, mip, mip + delta)
      mip += delta

    return bbox

  def reset_scales(self):
    """Used for manually resetting downsamples if something messed up."""
    self.info['scales'] = self.info['scales'][0:1]

  def add_scale(self, factor, encoding=None, chunk_size=None, info=None):
    """
    Generate a new downsample scale to for the info file and return an updated dictionary.
    You'll still need to call self.commit_info() to make it permenant.

    Required:
      factor: int (x,y,z), e.g. (2,2,1) would represent a reduction of 2x in x and y

    Optional:
      encoding: force new layer to e.g. jpeg or compressed_segmentation
      chunk_size: force new layer to new chunk size

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
    if not chunk_size:
      chunk_size = lib.find_closest_divisor(fullres['chunk_sizes'][0], closest_to=[64,64,64])

    if encoding is None:
      encoding = fullres['encoding']

    newscale = {
      u"encoding": encoding,
      u"chunk_sizes": [ list(map(int, chunk_size)) ],
      u"resolution": list(map(int, Vec(*fullres['resolution']) * factor )),
      u"voxel_offset": downscale(fullres['voxel_offset'], factor, np.floor),
      u"size": downscalef(ullres['size'], factor, np.ceil),
    }

    if newscale['encoding'] == 'compressed_segmentation':
      if 'compressed_segmentation_block_size' in fullres:
        newscale['compressed_segmentation_block_size'] = fullres['compressed_segmentation_block_size']  
      else: 
        newscale['compressed_segmentation_block_size'] = (8,8,8)

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