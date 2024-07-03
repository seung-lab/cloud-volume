from typing import List

from collections.abc import Iterable
from functools import reduce
import json
import operator
import os
import posixpath

import json5
import multiprocessing as mp
import numpy as np
from six import string_types
from six.moves import range
from tqdm import tqdm

from cloudfiles import CloudFiles 

from ... import exceptions
from ...provenance import DatasetProvenance, DataLayerProvenance

from ... import lib
from ...lib import ( 
  colorize, red, mkdir, 
  Vec, Bbox, jsonify, 
)
from ...paths import strict_extract, ascloudpath, to_https_protocol

def downscale(size, factor_in_mip, roundingfn):
  smaller = Vec(*size, dtype=np.float32) / Vec(*factor_in_mip)
  return list(map(int, roundingfn(smaller)))

class PrecomputedMetadata(object):
  """
  The PrecomputedMetadataService provides methods for fetching,
  saving, and accessing information about the data type & compression, 
  bounding box, resolution, and provenance of a given dataset 
  stored in Precomputed format.  

  This class is a building block for building a class that can
  read and write Precomputed data types.
  """
  def __init__(
    self, cloudpath, config, cache=None, 
    info=None, provenance=None, 
    max_redirects=10, use_https=False
  ):
    self.path = strict_extract(cloudpath)
    self.cache = cache
    if self.cache:
      self.cache.meta = self
    self.config = config
    self.info = None
    self.rois = None

    self.redirected_from = []
    self.use_https = use_https

    if info is None:
      self.refresh_info(max_redirects=max_redirects)
      if self.cache and self.cache.enabled:
        self.cache.check_info_validity()
    else:
      self.info = info

    if provenance is None:
      self.provenance = None
      self.refresh_provenance()
      if self.cache.enabled:
        self.cache.check_provenance_validity()
    else:
      self.provenance = self._cast_provenance(provenance)

  def check_for_placeholder_scale(self, mip:int):
    key = self.key(mip).lower()
    return 'placeholder' in key

  @classmethod
  def create_info(cls, 
    num_channels, layer_type, data_type, encoding, 
    resolution, voxel_offset, volume_size, 
    mesh=None, skeletons=None, chunk_size=(128,128,64),
    compressed_segmentation_block_size=(8,8,8),
    max_mip=0, factor=Vec(2,2,1), redirect=None
  ):
    """
    Create a new neuroglancer Precomputed info file.

    Required:
      num_channels: (int) 1 for grayscale, 3 for RGB 
      layer_type: (str) typically "image" or "segmentation"
      data_type: (str) e.g. "uint8", "uint16", "uint32", "float32"
      encoding: (str) "raw" for binaries like numpy arrays, "jpeg"
      resolution: float (x,y,z), x,y,z voxel dimensions in nanometers
      voxel_offset: int (x,y,z), beginning of dataset in positive cartesian space
      volume_size: int (x,y,z), extent of dataset in cartesian space from voxel_offset
    
    Optional:
      mesh: (str) name of mesh directory, typically "mesh"
      skeletons: (str) name of skeletons directory, typically "skeletons"
      chunk_size: int (x,y,z), dimensions of each downloadable 3D image chunk in voxels
      compressed_segmentation_block_size: (x,y,z) dimensions of each compressed sub-block
        (only used when encoding is 'compressed_segmentation')
      max_mip: (int), the maximum mip level id.
      factor: (Vec), the downsampling factor for each mip level
      redirect: (str), cloudpath to redirect to

    Returns: dict representing a single mip level that's JSON encodable
    """
    if not isinstance(factor, Vec):
      factor = Vec(*factor)

    if not isinstance(data_type, str):
      data_type = np.dtype(data_type).name

    precision = max(map(lib.getprecision, resolution))
    res_dtype = float
    if precision == 0:
      res_dtype = int

    resolution = np.asarray(resolution, dtype=res_dtype)

    info = {
      "num_channels": int(num_channels),
      "type": layer_type,
      "data_type": data_type,
      "scales": [{
        "encoding": encoding,
        "chunk_sizes": [ list(map(int, chunk_size)) ],
        "key": "_".join(map(str, resolution)),
        "resolution": list(map(res_dtype, resolution)),
        "voxel_offset": list(map(int, voxel_offset)),
        "size": list(map(int, volume_size)),
      }],
    }

    if redirect is not None:
      info['redirect'] = str(redirect)
 
    # add mip levels
    # the max_mip should be inclusive
    for mip in range(1, max_mip + 1):
      cls.add_scale(None, factor ** mip, encoding=encoding, chunk_size=chunk_size, info=info)

    if encoding == 'compressed_segmentation':
      info['scales'][0]['compressed_segmentation_block_size'] = list(map(int, compressed_segmentation_block_size))

    if mesh:
      info['mesh'] = 'mesh' if not isinstance(mesh, string_types) else mesh

    if skeletons:
      info['skeletons'] = 'skeletons' if not isinstance(skeletons, string_types) else skeletons      
    
    return info

  def refresh_info(self, max_redirects=10, force_fetch=False):
    """
    Refresh the current info file from the cache (if enabled) 
    or primary storage (e.g. the cloud) if not cached.

    max_redirects: number of times to allow redirection. set to 0 to
      force getting the origin info file loaded.
    force_fetch: bypass the cache for reading, but allow writing

    Raises:
      cloudvolume.exceptions.InfoUnavailableError when the info file 
        is unable to be retrieved.
      cloudvolume.exceptions.TooManyRedirects if more than max_redirects
        are followed.
      cloudvolume.exceptions.CyclicRedirect if a previously visited 
        location is revisited.

    See also: fetch_info

    Returns: dict
    """
    if self.cache and self.cache.enabled and not force_fetch:
      info = self.cache.get_json('info')
      if info:
        self.info = info
        return self.info

    self.info = self.redirectable_fetch_info(max_redirects)
    self.rois = self.parse_rois(self.info)

    if self.cache:
      self.cache.maybe_cache_info()
    return self.info

  def parse_rois(self, info) -> List[Bbox]:
    """Parse ROIs from the info file at mip 0."""
    scale = info['scales'][0]
    if 'rois' not in scale:
      return None

    bboxes = [ 
      Bbox.from_list(roi) for roi in scale["rois"] 
    ]
    bboxes.sort(key=lambda bx: bx.minpt.z)
    return bboxes

  def fetch_info(self):
    """
    Refresh the current info file from primary storage (e.g. the cloud) without
    refrence to the cache. The cache will not be updated.
  
    Raises cloudvolume.exceptions.InfoUnavailableError when the info file 
    is unable to be retrieved.

    See also: refresh_info

    Returns: dict
    """
    info = CloudFiles(self.cloudpath, secrets=self.config.secrets).get_json('info')

    if info is None:
      raise exceptions.InfoUnavailableError(
        red('No info file was found: {}'.format(self.infopath))
      )

    return info

  def redirectable_fetch_info(self, max_redirects=10):
    """
    Refresh the current info file from primary storage (e.g. the cloud) without
    refrence to the cache. The cache will not be updated. 'redirect' links
    in the info file will be followed up to `max_redirects` times after which
    an exception will be raised.

    Raises:
      cloudvolume.exceptions.InfoUnavailableError when the info file 
        is unable to be retrieved.
      cloudvolume.exceptions.TooManyRedirects if more than max_redirects
        are followed.
      cloudvolume.exceptions.CyclicRedirect if a previously visited 
        location is revisited.

    See also: refresh_info, fetch_info

    Optional:
      max_redirects: if 'redirect' is specified in an info file, 
        follow the link up to this many times to the pointed locations.

    Returns: dict
    """
    visited = []

    if max_redirects <= 0:
      return self.fetch_info()

    if self.path.format == 'graphene':
      start = self.server_url
    else:
      start = self.cloudpath

    for _ in range(max_redirects):
      info = self.fetch_info()

      if 'redirect' not in info or not info['redirect']:
        break

      path = strict_extract(info['redirect'])
      if self.use_https:
        path = to_https_protocol(path)

      if path == self.path:
        break 
      elif path in visited:
        raise exceptions.CyclicRedirect(
          """
Tried to redirect through a cycle.

Start: {}
Hops: 
\t{}
\n""".format(
          start, 
          "\n\t".join([ 
            str(i+1) + ". " + ascloudpath(v) for i, v in enumerate(visited) 
          ]))
        )

      visited.append(path)
      self.path = path
    else:
      raise exceptions.TooManyRedirects(
        "Tried to redirect more than {} hops.".format(max_redirects)
      )

    self.redirected_from = visited[:-1]

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
    # use put instead of put_json to preserve formatting
    cf = CloudFiles(self.cloudpath, secrets=self.config.secrets)
    cf.put(
      'info', infojson,
      cache_control='no-cache',
      content_type='application/json'
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
    cf = CloudFiles(self.cloudpath, secrets=self.config.secrets)
    provfile = cf.get('provenance')
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

    # need to use put vs put_json to preserve formatting
    cf = CloudFiles(self.cloudpath, secrets=self.config.secrets)
    cf.put(
      'provenance', prov, 
      cache_control='no-cache', 
      content_type='application/json'
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

  def join(self, *paths):
    if self.path.protocol == 'file':
      return os.path.join(*paths)
    else:
      return posixpath.join(*paths)

  @property
  def basepath(self):
    return self.path.basepath
    
  @property 
  def layerpath(self):
    return self.join(self.basepath, self.layer)

  @property
  def base_cloudpath(self):
    return self.path.protocol + "://" + self.basepath

  @property 
  def cloudpath(self):
    return self.join(self.base_cloudpath, self.layer)
  
  @property
  def infopath(self):
    return self.join(self.cloudpath, 'info')

  @property
  def skeletons(self):
    return self.info['skeletons'] if 'skeletons' in self.info else None

  @property
  def mesh(self):
    return self.info['mesh'] if 'mesh' in self.info else None

  def shape(self, mip):
    """Returns Vec(x,y,z,channels) shape of the volume similar to numpy.""" 
    size = self.volume_size(mip)
    return Vec(size.x, size.y, size.z, self.num_channels)

  def volume_size(self, mip):
    """Returns Vec(x,y,z) shape of the volume (i.e. shape - channels).""" 
    return Vec(*self.info['scales'][mip]['size'])

  def voxels(self, mip):
    """Returns the number of voxels in this mip level."""
    return reduce(operator.mul, self.volume_size(mip))

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
  def data_type(self) -> str:
    """e.g. 'uint8'"""
    return self.info['data_type']

  def encoding(self, mip, encoding=None):
    """
    If encoding is provided, set the encoding for this
    mip level. If the encoding is not provided, this is
    a getter.

    Typical values: 'raw', 'jpeg', 'compressed_segmentation'

    Returns encoding for the mip level either way.
    """
    if encoding is None:
      return self.info['scales'][mip]['encoding']

    encoding = encoding.lower()
    scale = self.scale(mip)
    scale['encoding'] = encoding
    if (encoding == 'compressed_segmentation' \
      and 'compressed_segmentation_block_size' not in scale):

      scale['compressed_segmentation_block_size'] = (8,8,8)

    return encoding

  def compression_params(self, mip):
    encoding = self.encoding(mip)
    scale = self.scale(mip)
    if encoding == 'zfpc':
      return self.zfpc_encoding_params(mip)
    elif encoding == 'compressed_segmentation':
      return { "block_size": self.compressed_segmentation_block_size(mip) }
    elif encoding == 'png':
      return { "level": scale.get("png_level", None) }
    elif encoding == 'jpeg':
      return { "level": scale.get("jpeg_quality", None) }
    elif encoding == 'fpzip':
      return { "level": scale.get("fpzip_precision", None) }
    else:
      return {}

  def zfpc_encoding_params(self, mip):
    """
    Returns tuning arguments for zfpc.compress.

    Reads parameters from scale:
    zfpc_rate, zfpc_precision, zfpc_tolerance, 
    and zfpc_correlated_dims ([bool x 4])
    """
    scale = self.scale(mip)
    return {
      'rate': scale.get('zfpc_rate', -1),
      'precision': scale.get('zfpc_precision', -1),
      'tolerance': scale.get('zfpc_tolerance', -1),
      'correlated_dims': scale.get('zfpc_correlated_dims', [True]*4),
    }

  def compressed_segmentation_block_size(self, mip):
    if 'compressed_segmentation_block_size' in self.info['scales'][mip]:
      return self.info['scales'][mip]['compressed_segmentation_block_size']
    return None

  @property
  def num_channels(self):
    return int(self.info['num_channels'])

  def voxel_offset(self, mip):
    """Vec(x,y,z) start of the dataset in voxels"""
    scale = self.scale(mip)
    if 'voxel_offset' in scale:
      return Vec(*scale['voxel_offset'], dtype=int)
    else:
      return Vec(0,0,0)

  def resolution(self, mip):
    """Vec(x,y,z) dimensions of each voxel in nanometers"""
    res = self.info['scales'][mip]['resolution']
    dtype = float if lib.floating(res) else int
    return Vec(*res, dtype=dtype)

  def to_mip(self, mip):
    mip = list(mip) if isinstance(mip, Iterable) else int(mip)
    try:
      if isinstance(mip, list):  # mip specified by voxel resolution
        return next(
          (i for (i,s) in enumerate(self.scales) if s["resolution"] == mip)
        )
      else:  # mip specified by index into downsampling hierarchy
        return self.available_mips[mip]
    except Exception:
      if isinstance(mip, list):
        opening_text = "Scale <{}>".format(", ".join(map(str, mip)))
      else:
        opening_text = "MIP {}".format(str(mip))
  
      scales = [ ",".join(map(str, scale)) for scale in self.available_resolutions ]
      scales = [ "<{}>".format(scale) for scale in scales ]
      scales = ", ".join(scales)
      msg = "{} not found. {} available: {}".format(
        opening_text, len(self.available_mips), scales
      )
      raise exceptions.ScaleUnavailableError(msg)

  def downsample_ratio(self, mip):
    """Describes how downsampled the current mip level is as an (x,y,z) factor triple."""
    return self.resolution(mip) / self.resolution(0)

  def chunk_size(self, mip):
    """Underlying chunk size dimensions in voxels. Synonym for underlying."""
    return Vec(*self.scale(mip)['chunk_sizes'][0])

  def key(self, mip):
    """The subdirectory within the data layer containing the chunks for this mip level"""
    return self.scale(mip)['key']

  @property
  def keys(self):
    return [ self.key(mip) for mip in self.available_mips ]

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
    return np.floor(pt * downsample_ratio).astype(np.int64)

  def bbox_to_mip(self, bbox, mip, to_mip):
    """Convert bbox or slices from one mip level to another."""
    bbox = Bbox.create(bbox, self.bounds(mip))

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

  def overlaps_roi(self, pt_or_bbox, mip = 0) -> bool:
    """Returns True if the point or bbox overlaps the ROI including the boundary."""
    if self.rois is None:
      return True      

    if isinstance(pt_or_bbox, Bbox):
      if mip > 0:
        pt_or_bbox = self.bbox_to_mip(pt_or_bbox, mip, 0)

      for bbox in self.rois:
        if bbox.overlaps_bbox(pt_or_bbox):
          return True
    else:
      if mip > 0:
        pt_or_bbox = self.point_to_mip(pt_or_bbox, mip, 0)

      for bbox in self.rois:
        if bbox.contains(pt_or_bbox):
          return True

    return False

  def reset_scales(self):
    """Used for manually resetting downsamples if something messed up."""
    self.info['scales'] = self.info['scales'][0:1]

  def add_resolution(
    self, res, encoding=None, 
    chunk_size=None, info=None,
    encoding_level=None,
  ):
    if lib.floating(res):
      factor = Vec(*res, dtype=float) / self.resolution(0)
    else:
      factor = Vec(*res) // self.resolution(0)

    return self.add_scale(
      factor, encoding, chunk_size, info, 
      encoding_level=encoding_level
    )

  def add_scale(
    self, factor, 
    encoding=None, chunk_size=None, info=None,
    encoding_level=None,
  ):
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

    factor = np.round(factor).astype(int)

    for dim in factor:
      pot = np.log2(dim)
      if pot != int(pot):
        print(
          f"WARNING: adding scale with a non-power-of-two scale factor: {factor}"
          f"This is very uncommon and will likely cause problems at some point."
        )
        break

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
    if chunk_size is None:
      chunk_size = lib.find_closest_divisor(fullres['chunk_sizes'][0], closest_to=[64,64,64])

    if encoding is None:
      encoding = fullres['encoding']

    precision = max(map(lib.getprecision, fullres['resolution']))
    precision = max(precision, max(map(lib.getprecision, factor)))

    dtype = float
    if precision == 0:
      dtype = int

    newscale = {
      u"encoding": encoding,
      u"chunk_sizes": [ list(map(int, chunk_size)) ],
      u"resolution": list(map(dtype, Vec(*fullres['resolution'], dtype=dtype) * factor )),
      u"voxel_offset": downscale(fullres.get('voxel_offset', (0,0,0)), factor, np.floor),
      u"size": downscale(fullres['size'], factor, np.ceil),
    }

    if encoding_level is not None:
      if encoding == "jpeg":
        newscale["jpeg_quality"] = int(encoding_level)
      elif encoding == "png":
        newscale["png_level"] = int(encoding_level)
      elif encoding == "fpzip":
        newscale["fpzip_precision"] = int(encoding_level)

    if newscale['encoding'] == 'compressed_segmentation':
      if 'compressed_segmentation_block_size' in fullres:
        newscale['compressed_segmentation_block_size'] = fullres['compressed_segmentation_block_size']  
      else: 
        newscale['compressed_segmentation_block_size'] = (8,8,8)

    newscale[u'key'] = str("_".join([ str(res) for res in newscale['resolution']]))

    new_res = np.array(newscale['resolution'], dtype=dtype)

    preexisting = False
    for index, scale in enumerate(info['scales']):
      res = np.array(scale['resolution'], dtype=dtype)
      if np.array_equal(new_res, res):
        preexisting = True
        info['scales'][index] = newscale
        break

    if not preexisting:    
      info['scales'].append(newscale)

    return newscale

  def lock_mips(self, mips):
    """
    Establishes a write lock on the specified mip levels.
    The lock is written to the cloud info file.
    """
    mips = lib.toiter(mips)
    if max(mips) > max(self.available_mips):
      raise ValueError("Cannot lock a mip level that doesn't exist. Highest mip: {} Got: {}".format(
        max(self.available_mips), mips
      ))

    try:
      self.refresh_info(force_fetch=True)

      for mip in mips:
        self.info['scales'][mip]['locked'] = True

      self.commit_info()
    except Exception as err:
      msg = lib.red("Unable to acquire write lock on mips {}!".format(list(mips)))
      raise exceptions.WriteLockAcquisitionError(msg)

  def unlock_mips(self, mips):
    """
    Releases a write lock on the specified mip levels.
    The lock is written to the cloud info file.
    """
    mips = lib.toiter(mips)
    if max(mips) > max(self.available_mips):
      raise ValueError("Cannot unlock a mip level that doesn't exist. Highest mip: {} Got: {}".format(
        max(self.available_mips), mips
      ))

    try:
      self.refresh_info(force_fetch=True)

      for mip in mips:
        self.info['scales'][mip]['locked'] = False

      self.commit_info()
    except Exception as err:
      msg = lib.yellow("Unable to release lock on mips {}".format(list(mips)))
      raise exceptions.WriteLockReleaseError(msg)

  def locked_mips(self):
    return set([ i for i, scale in enumerate(self.info['scales']) if scale.get('locked', False) ])