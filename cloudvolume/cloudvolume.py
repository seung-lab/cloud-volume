from __future__ import print_function

from collections import namedtuple
import json
import os
import re

import numpy as np
from PIL import Image
from tqdm import tqdm

from intern.remote.boss import BossRemote
from intern.resource.boss.resource import ChannelResource, ExperimentResource, CoordinateFrameResource
from secrets import boss_credentials

import chunks
import lib
from lib import mkdir, clamp, xyzrange, Vec, Bbox, min2, max2, check_bounds
from storage import Storage
import mesh2obj

__all__ = [ 'CloudVolume', 'EmptyVolumeException' ]

ExtractedPath = namedtuple('ExtractedPath', 
  ('protocol','bucket_name', 'dataset_name','layer_name')
)

class EmptyVolumeException(Exception):
  """Raised upon finding a missing chunk."""
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
    fill_missing: (bool) If a file inside volume bounds is unable to be fetched:
        True: Use a block of zeros
        False: Throw an error
    info: (dict) in lieu of fetching a neuroglancer info file, use this provided one.
            This is useful when creating new datasets.
  """
  def __init__(self, cloudpath, mip=0, bounded=True, fill_missing=False, info=None):
    super(self.__class__, self).__init__()

    extract = CloudVolume.extract_path(cloudpath)

    self._protocol = extract.protocol
    self._bucket = extract.bucket_name

    # You can access these two with properties
    self._dataset_name = extract.dataset_name
    self._layer = extract.layer_name

    self.mip = mip
    self.bounded = bounded
    self.fill_missing = fill_missing

    self._storage = None
    if self._protocol != 'boss':
      self._storage = Storage(self.layer_cloudpath, n_threads=0)

    if info is None:
      self.refreshInfo()
    else:
      self.info = info

    try:
      self.mip = self.available_mips[self.mip]
    except:
      raise Exception("MIP {} has not been generated.".format(self.mip))

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
        "resolution": list(resolution),
        "voxel_offset": list(voxel_offset),
        "size": list(volume_size),
      }],
    }

    if mesh:
      info['mesh'] = 'mesh' if type(mesh) not in (str, unicode) else mesh

    return info

  @classmethod
  def extract_path(cls, cloudpath):
    """cloudpath: e.g. gs://neuroglancer/DATASET/LAYER/info or s3://..."""
    match = re.match(r'^(gs|file|s3|boss)://(/?[\d\w_\.\-]+)/([\d\w_\.\-]+)/([\d\w_\.\-]+)/?', cloudpath)
    return ExtractedPath(*match.groups())

  def refreshInfo(self):
    if self._protocol != "boss":
      infojson = self._storage.get_file('info')
      self.info = json.loads(infojson)
    else:
      self.info = self.fetchBossInfo()
    return self.info

  def fetchBossInfo(self):
    experiment = ExperimentResource(
      name=self._dataset_name, 
      collection_name=self._bucket
    )
    rmt = BossRemote(boss_credentials)
    experiment = rmt.get_project(experiment)

    coord_frame = CoordinateFrameResource(name=experiment.coord_frame)
    coord_frame = rmt.get_project(coord_frame)

    channel = ChannelResource(self._layer, self._bucket, self._dataset_name)
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
    bbox.maxpt = bbox.maxpt - 1 # boss uses exclusive outer bound

    layer_type = 'unknown'
    if 'type' in channel.raw:
      layer_type = channel.raw['type']

    return CloudVolume.create_new_info(
      num_channels=1,
      layer_type=layer_type,
      data_type=channel.datatype,
      encoding='raw',
      resolution=resolution,
      voxel_offset=bbox.minpt,
      volume_size=bbox.size3(),
    )

  def commitInfo(self):
    if self._protocol == 'boss':
      return self 

    infojson = json.dumps(self.info)
    self._storage.put_file('info', infojson, 'application/json').wait()
    return self

  @property
  def dataset_name(self):
    return self._dataset_name

  @dataset_name.setter
  def dataset_name(self, name):
    if name != self._dataset_name:
      self._dataset_name = name
      self.refreshInfo()
  
  @property
  def layer(self):
    return self._layer

  @layer.setter
  def layer(self, name):
    if name != self._layer:
      self._layer = name
      self.refreshInfo()

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
    return "{}://{}/{}/".format(self._protocol, self._bucket, self.dataset_name)

  @property
  def layer_cloudpath(self):
    return os.path.join(self.base_cloudpath, self.layer)

  @property
  def info_cloudpath(self):
    return os.path.join(self.layer_cloudpath, 'info')

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

  def slices_from_global_coords(self, slices):
    """
    Used for converting from mip 0 coordinates to upper mip level
    coordinates. This is mainly useful for debugging since the neuroglancer
    client displays the mip 0 coordinates for your cursor.
    """

    maxsize = list(self.mip_volume_size(0)) + [ self.num_channels ]
    minsize = list(self.mip_voxel_offset(0)) + [ 0 ]

    slices = generate_slices(slices, minsize, maxsize)[:3]
    lower = Vec(*map(lambda x: x.start, slices))
    upper = Vec(*map(lambda x: x.stop, slices))
    step = Vec(*map(lambda x: x.step, slices))

    lower /= self.downsample_ratio
    upper /= self.downsample_ratio

    signs = step / np.absolute(step)
    step = signs * max2(np.absolute(step / self.downsample_ratio), Vec(1,1,1))
    step = Vec(*np.round(step))

    return [
      slice(lower.x, upper.x, step.x),
      slice(lower.y, upper.y, step.y),
      slice(lower.z, upper.z, step.z)
    ]

  def reset_scales(self):
    """Used for manually resetting downsamples if something messed up."""
    self.info['scales'] = self.info['scales'][0:1]
    return self.commitInfo()

  def add_scale(self, factor):
    """
    Generate a new downsample scale to for the info file and return an updated dictionary.
    You'll still need to call self.commitInfo() to make it permenant.

    Required:
      factor: int (x,y,z), e.g. (2,2,1) would represent a reduction of 2x in x and y

    Returns: info dict
    """
    # e.g. {"encoding": "raw", "chunk_sizes": [[64, 64, 64]], "key": "4_4_40", 
    # "resolution": [4, 4, 40], "voxel_offset": [0, 0, 0], 
    # "size": [2048, 2048, 256]}
    fullres = self.info['scales'][0]

    # If the voxel_offset is not divisible by the ratio,
    # zooming out will slightly shift the data.
    # Imagine the offset is 10
    #    the mip 1 will have an offset of 5
    #    the mip 2 will have an offset of 2 instead of 2.5 
    #        meaning that it will be half a pixel to the left
    
    chunk_size = lib.find_closest_divisor(fullres['chunk_sizes'][0], closest_to=[64,64,64])

    def downscale(size, roundingfn):
      smaller = Vec(*size, dtype=np.float32) / Vec(*factor)
      return list(roundingfn(smaller).astype(int))

    newscale = {
      u"encoding": fullres['encoding'],
      u"chunk_sizes": [ chunk_size ],
      u"resolution": list( Vec(*fullres['resolution']) * factor ),
      u"voxel_offset": downscale(fullres['voxel_offset'], np.floor),
      u"size": downscale(fullres['size'], np.ceil),
    }

    newscale[u'key'] = unicode("_".join([ str(res) for res in newscale['resolution']]))

    new_res = np.array(newscale['resolution'], dtype=int)

    preexisting = False
    for index, scale in enumerate(self.info['scales']):
      res = np.array(scale['resolution'], dtype=int)
      if np.array_equal(new_res, res):
        preexisting = True
        self.info['scales'][index] = newscale
        break

    if not preexisting:    
      self.info['scales'].append(newscale)

    return newscale

  def __getitem__(self, slices):
    maxsize = list(self.bounds.maxpt) + [ self.num_channels ]
    minsize = list(self.bounds.minpt) + [ 0 ]

    slices = generate_slices(slices, minsize, maxsize, bounded=self.bounded)
    channel_slice = slices.pop()

    minpt = Vec(*[ slc.start for slc in slices ])
    maxpt = Vec(*[ slc.stop for slc in slices ]) 
    steps = Vec(*[ slc.step for slc in slices ])

    requested_bbox = Bbox(minpt, maxpt)
    
    if self._protocol == 'boss':
      cutout = self._boss_cutout(requested_bbox, steps, channel_slice)
    else:
      cutout = self._cutout(requested_bbox, steps, channel_slice)

    if self.bounded:
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

  def _cutout(self, requested_bbox, steps, channel_slice=slice(None)):
    realized_bbox = requested_bbox.expand_to_chunk_size(self.underlying, offset=self.voxel_offset)
    realized_bbox = Bbox.clamp(realized_bbox, self.bounds)

    def multichannel_shape(bbox):
      shape = bbox.size3()
      return (shape[0], shape[1], shape[2], self.num_channels)

    cloudpaths = self.__chunknames(realized_bbox, self.bounds, self.key, self.underlying)
    renderbuffer = np.zeros(shape=multichannel_shape(realized_bbox), dtype=self.dtype)

    with Storage(self.layer_cloudpath) as storage:
      files = storage.get_files(cloudpaths)

    for fileinfo in tqdm(files, total=len(cloudpaths), desc="Rendering Image"):
      if fileinfo['error'] is not None:
        raise fileinfo['error']

      bbox = Bbox.from_filename(fileinfo['filename'])
      content_len = len(fileinfo['content']) if fileinfo['content'] is not None else 0

      if not fileinfo['content']:
        if self.fill_missing:
          fileinfo['content'] = ''
        else:
          raise EmptyVolumeException(fileinfo['filename'])

      try:
        img3d = chunks.decode(
          fileinfo['content'], self.encoding, multichannel_shape(bbox), self.dtype
        )
      except Exception:
        print('File Read Error: {} bytes, {}, {}, errors: {}'.format(
            content_len, bbox, fileinfo['filename'], fileinfo['error']))
        raise
      
      start = bbox.minpt - realized_bbox.minpt
      end = min2(start + self.underlying, renderbuffer.shape[:3] )
      delta = min2(end - start, img3d.shape[:3])
      end = start + delta

      renderbuffer[ start.x:end.x, start.y:end.y, start.z:end.z, : ] = img3d[ :delta.x, :delta.y, :delta.z, : ]

    bounded_request = Bbox.clamp(requested_bbox, self.bounds)
    lp = bounded_request.minpt - realized_bbox.minpt # low realized point
    hp = lp + bounded_request.size3()

    renderbuffer = renderbuffer[ lp.x:hp.x:steps.x, lp.y:hp.y:steps.y, lp.z:hp.z:steps.z, channel_slice ] 
    return VolumeCutout.from_volume(self, renderbuffer, bounded_request)
  
  def _boss_cutout(self, requested_bbox, steps, channel_slice=slice(None)):
    bounds = Bbox.clamp(requested_bbox, self.bounds)

    if bounds.volume() < 1:
      raise ValueError('Requested less than one pixel of volume. {}'.format(bounds))

    x_rng = [ bounds.minpt.x, bounds.maxpt.x ]
    y_rng = [ bounds.minpt.y, bounds.maxpt.y ]
    z_rng = [ bounds.minpt.z, bounds.maxpt.z ]

    layer_type = 'image' if self.layer_type == 'unknown' else self.layer_type

    chan = ChannelResource(
      collection_name=self._bucket, 
      experiment_name=self._dataset_name, 
      name=self._layer, # Channel
      type=layer_type, 
      datatype=self.dtype,
    )

    rmt = BossRemote(boss_credentials)
    cutout = rmt.get_cutout(chan, self.mip, x_rng, y_rng, z_rng).T
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
    slices = generate_slices(slices, minsize, maxsize)
    bbox = Bbox.from_slices(slices)

    slice_shape = list(bbox.size3()) + [ slices[3].stop - slices[3].start ]

    if not np.array_equal(imgshape, slice_shape):
      raise ValueError("Illegal slicing, Image shape: {} != {} Slice Shape".format(imgshape, slice_shape))

    if self._protocol == 'boss':
      self.upload_boss_image(img, bbox.minpt)
    else:
      self.upload_image(img, bbox.minpt)

  def upload_boss_image(self, img, offset):
    shape = Vec(*img.shape[:3])
    offset = Vec(*offset)

    bounds = Bbox(offset, shape + offset)

    if bounds.volume() < 1:
      raise ValueError('Requested less than one pixel of volume. {}'.format(bounds))

    x_rng = [ bounds.minpt.x, bounds.maxpt.x ]
    y_rng = [ bounds.minpt.y, bounds.maxpt.y ]
    z_rng = [ bounds.minpt.z, bounds.maxpt.z ]

    layer_type = 'image' if self.layer_type == 'unknown' else self.layer_type

    chan = ChannelResource(
      collection_name=self._bucket, 
      experiment_name=self._dataset_name, 
      name=self._layer, # Channel
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
    if self._protocol == 'boss':
      raise NotImplementedError

    if str(self.dtype) != str(img.dtype):
      raise ValueError('The uploaded image data type must match the volume data type. volume: {}, image: {}'.format(self.dtype, img.dtype))

    uploads = []
    for imgchunk, spt, ept in tqdm(self._generate_chunks(img, offset), desc='uploading image'):
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
      uploads.append( (cloudpath, encoded) )

    content_type = 'application/octet-stream'
    if self.encoding == 'jpeg':
      content_type == 'image/jpeg'

    compress = (self.encoding in ('raw', 'compressed_segmentation'))

    with Storage(self.layer_cloudpath) as storage:
      storage.put_files(uploads, content_type=content_type, compress=compress)

  def _generate_chunks(self, img, offset):
    shape = Vec(*img.shape)[:3]
    offset = Vec(*offset)[:3]

    bounds = Bbox( offset, shape + offset)

    alignment_check = bounds.round_to_chunk_size(self.underlying, self.voxel_offset)

    if not np.all(alignment_check.minpt == bounds.minpt):
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

  def get_mesh(self, segid):
    """Download the raw mesh fragments for this seg ID."""
    mesh_dir = self.info['mesh']

    mesh_json_file_name = str(segid) + ':0'

    download_path = os.path.join(mesh_dir, mesh_json_file_name)

    with Storage(self.layer_cloudpath) as stor:
      fragments = json.loads(stor.get_file(download_path))['fragments']
      
      # Older mesh manifest generation tasks had a bug where they
      # accidently included the manifest file in the list of mesh
      # fragments. Exclude these accidental files, no harm done.
      fragments = [ f for f in fragments if f != mesh_json_file_name ] 

      paths = [ os.path.join(mesh_dir, fragment) for fragment in fragments ]
      frag_datas = stor.get_files(paths)  
    return frag_datas

  def save_mesh(self, segids, file_format='obj'):
    """
    Save one or more segids into a common mesh format as a single file.

    segids: int, string, or list thereof

    Supported Formats: 'obj'
    """
    if type(segids) != list:
      segids = [ segids ]

    fragments = []
    for segid in segids:
      fragments.extend( self.get_mesh(segid) )

    meshdata = mesh2obj.decode_downloaded_data(fragments)

    if file_format != 'obj':
      raise NotImplementedError('Only .obj is currently supported.')

    filename = str(segids[0])
    if len(segids) > 1:
      filename = "{}_{}".format(segids[0], segids[-1])

    num_vertices = 0
    with open('./{}.obj'.format(filename), 'wb') as f:
      for name, fragment in meshdata.items():
        mesh_data = mesh2obj.mesh_to_obj(fragment, num_vertices)
        f.write('\n'.join(mesh_data) + '\n')
        num_vertices += fragment['num_vertices']


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

  def __del__(self):
    if self._storage:
      self._storage.kill_threads()

def generate_slices(slices, minsize, maxsize, bounded=True):
  """Assisting function for __getitem__. e.g. vol[:,:,:,:]"""

  if isinstance(slices, int) or isinstance(slices, float) or isinstance(slices, long):
    slices = [ slice(int(slices), int(slices)+1, 1) ]
  if type(slices) == slice:
    slices = [ slices ]

  slices = list(slices)

  while len(slices) < len(maxsize):
    slices.append( slice(None, None, None) )

  # First three slices are x,y,z, last is channel. 
  # Handle only x,y,z here, channel seperately
  for index, slc in enumerate(slices):
    if isinstance(slc, int) or isinstance(slc, float) or isinstance(slc, long):
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
    super(VolumeCutout, self).__init__(self, shape=buf.shape, buffer=buf, dtype=buf.dtype)
    
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

  def upload(self, info):
    bounds = self.bounds.shrunk_to_chunk_size( DEFAULT_CHUNK_SIZE )

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

    for level in tqdm(xrange(self.shape[index]), desc="Saving Images"):
      if index == 0:
        img = self[level, :, :, channel ]
      elif index == 1:
        img = self[:, level, :, channel ]
      elif index == 2:
        img = self[:, :, level, channel ]
      else:
        raise NotImplemented

      num_channels = img.shape[2]

      for channel_index in xrange(num_channels):
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

