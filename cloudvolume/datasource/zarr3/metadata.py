import copy
import re
import os

import numpy as np

from cloudfiles import CloudFiles

from cloudvolume.datasource.precomputed.metadata import PrecomputedMetadata
from cloudvolume.lib import jsonify, Vec, Bbox, spatial_unit_in_meters, time_unit_in_seconds

from ... import exceptions
from ...provenance import DataLayerProvenance

ZARR3_VALID_DATATYPES = { 
  "int8", "int16", "int32", "int64",
  "uint8", "uint16", "uint32", "uint64",
  "float16", "float32", "float64",
  "complex64", "complex128",
  "r8", "r16", "r32", "r64",
}

DEFAULT_CODEC = [
  {"configuration":{"endian":"little"},"name":"bytes"},
  {
    "name": "blosc",
    "configuration": {
          "cname": "lz4",
          "clevel": 1,
          "shuffle": "shuffle",
          "typesize": 4,
          "blocksize": 0,
    },
  }
]

class Zarr3Metadata(PrecomputedMetadata):
  def __init__(self, cloudpath, config, cache,  info=None):
    
    orig_info = info

    # some default values, to be overwritten
    if not info:
      info = PrecomputedMetadata.create_info(
        num_channels=1, layer_type='image', data_type='uint8', 
        encoding='raw', resolution=[1,1,1], voxel_offset=[0,0,0], 
        volume_size=[1,1,1]
      )

    super().__init__(
      cloudpath, config, cache, info=info, provenance=None
    )

    self.zarrays = []
    self.zinfo = {}
    self.ome = self.default_attributes(3)

    if orig_info is None:
      self.info = self.fetch_info()
    else:
      self.render_zarr_metadata()

    self.provenance = DataLayerProvenance()

    if self.ndim > 5:
      raise ValueError("CloudVolume's zarr3 implementation only supports up to 5 dimensions (x,y,z,channel,time)")

  @property
  def zarr_format(self) -> int:
    return self.zinfo.get("zarr_format", None)

  def compute_resolution(self, mip:int) -> np.ndarray:
    scale = self.datasets()[mip]

    transforms = scale["coordinateTransformations"]

    res = np.ones([ self.ndim ], dtype=np.float32)

    for transform in transforms:
      if transform["type"] != "scale":
        continue

      if isinstance(transform["scale"], str):
        raise NotImplementedError(f"Binary scale data not currently supported. Located at {transform['scale'][:1000]}...")

      res *= np.array(transform["scale"], dtype=np.float32)

    return res

  def compute_voxel_offset(self, mip:int) -> np.ndarray:
    ds =  self.datasets()

    if len(ds) == 0:
      return np.zeros([self.ndim], dtype=int)

    scale = ds[mip]

    transforms = scale["coordinateTransformations"]

    # given in physical units
    voxel_offset = np.zeros([ self.ndim ], dtype=np.float32)

    for transform in transforms:
      if transform["type"] != "translation":
        continue

      if isinstance(transform["translation"], str):
        raise NotImplementedError(f"Binary translation data not currently supported. Located at {transform['scale'][:1000]}...")

      voxel_offset += np.array(transform["translation"], dtype=np.float32)

    resolution = self.compute_resolution(mip)

    return voxel_offset // resolution

  def default_attributes(self, num_axes):
    ome = {
      "version": "0.5",
      "multiscales": [
        {
          "axes": [
            {
              "name": "t",
              "type": "time",
              "unit": "millisecond"
            },
            {
              "name": "c",
              "type": "channel"
            },
            {
              "name": "z",
              "type": "space",
              "unit": "nanometer"
            },
            {
              "name": "y",
              "type": "space",
              "unit": "nanometer"
            },
            {
              "name": "x",
              "type": "space",
              "unit": "nanometer"
            }
          ],
          "datasets": [
            
          ],
        }
      ],
    }
    ome["multiscales"][0]["axes"] = ome["multiscales"][0]["axes"][-num_axes:]
    return ome

  def spatial_resolution_in_nm(self, mip):
    scale_factors = np.ones([3], dtype=np.float64)
    positions = [0,0,0]
    for i, axis in enumerate(self.axes()):
      if axis["type"] != "space":
        continue
      
      if axis["name"] == "x":
        scale_factors[0] = spatial_unit_in_meters(axis.get("unit", "nanometer"))
        positions[0] = i
      elif axis["name"] == "y":
        scale_factors[1] = spatial_unit_in_meters(axis.get("unit", "nanometer"))
        positions[1] = i
      elif axis["name"] == "z":
        scale_factors[2] = spatial_unit_in_meters(axis.get("unit", "nanometer"))
        positions[2] = i

    try:
      resolution = self.compute_resolution(mip)
      resolution = np.array([
        resolution[positions[0]],
        resolution[positions[1]],
        resolution[positions[2]]
      ], dtype=np.float64)
    except IndexError:
      resolution = np.ones([3], dtype=np.float64)

    return resolution * (scale_factors / 1e-9)

  def time_resolution_in_seconds(self, mip):
    i = 0
    unit = "second"
    for axis in self.axes():
      if axis["type"] == "time":
        unit = axis["unit"]
        break
      i += 1

    scale_factor = time_unit_in_seconds(unit)

    try:
      resolution = self.compute_resolution(mip)
      return resolution[i] * scale_factor
    except IndexError:
      return scale_factor

  def is_group(self):
    return len(self.datasets()) > 0

  def is_sharded(self, mip):
    for codec in self.codecs(mip):
      if codec.get("name", "") == "sharding_indexed":
        return True
    return False

  def has_time_axis(self):
    try:
      return self.time_index() is not None
    except ValueError:
      return False

  def datasets(self):
    return self.ome["multiscales"][0]["datasets"]

  def axes(self):
    return self.ome["multiscales"][0]["axes"]

  def time_index(self):
    for i, axis in enumerate(self.axes()):
      if axis["type"] == "time":
        return i
    raise ValueError("No time axis.")

  def shape(self, mip):
    """Returns Vec(x,y,z,channels) shape of the volume similar to numpy.""" 
    size = self.volume_size(mip)
    values = [size.x, size.y, size.z, self.num_channels]
    if self.has_time_axis():
      values.append(self.num_frames(mip))
    return Vec(*values)

  def spatial_chunk_size(self, mip):
    axes = self.axes()

    shape = [0,0,0]
    for i, axis in enumerate(axes):
      if axis["type"] == "space" and axis["name"] == "x":
        shape[0] = i
      elif axis["type"] == "space" and axis["name"] == "y":
        shape[1] = i
      elif axis["type"] == "space" and axis["name"] == "z":
        shape[2] = i

    cs = self.zarrays[0]["chunk_grid"]["configuration"]["chunk_shape"]
    return np.array([ cs[shape[0]], cs[shape[1]], cs[shape[2]] ], dtype=int)

  def time_chunk_size(self, mip):
    i = self.time_index()
    return self.zarrays[mip]["chunk_grid"]["configuration"]["chunk_shape"][i]

  def num_time_chunks(self, mip):
    nframes = self.num_frames(mip)
    t_chunk_size = self.time_chunk_size(mip)
    return int(np.ceil(nframes / t_chunk_size))

  def num_frames(self, mip):
    try:
      i = self.time_index()
      return self.zarrays[mip]["shape"][i]
    except ValueError:
      return 1

  def chunk_name(self, mip, *args, convert_order=False):
    sep = self.dimension_separator(mip)

    if convert_order:
      seq = self.zarr_axes_to_cv_axes()
      values = [ str(args[val]) for val in seq ]
    else:
      values = [ str(val) for val in args ]

    dsep = self.directory_separator()

    filename = sep.join([ *values ])

    chunk_key_encoding = self.chunk_key_encoding(mip)["name"]

    if chunk_key_encoding == "default":
      if self.is_group():
        return dsep.join([ self.key(mip), 'c', filename ])
      else:
        return dsep.join([ 'c', filename ])
    elif chunk_key_encoding == "v2":
      if self.is_group():
        return dsep.join([ self.key(mip), filename ])
      else:
        return filename
    else:
      raise ValueError(f"chunk_key_encoding '{chunk_key_encoding}' is not supported.")

  @property
  def ndim(self):
    return len(self.axes())

  def duration_in_seconds(self):
    return self.time_resolution_in_seconds(0) * self.num_frames(0)

  def codecs(self, mip):
    return self.zarrays[mip].get("codecs", [{}])

  def background_color(self, mip):
    color = self.zarrays[mip].get("fill_value", 0)
    if color == "NaN":
      return np.nan
    return color

  def set_background_color(self, mip, value):
    self.zarrays[mip]["fill_value"] = value

  def filename_regexp(self, mip):
    scale = self.ome["multiscales"][0]
    axes = scale["axes"]

    dsep = self.directory_separator()
    regexp = ""
    if len(self.ome["multiscales"][0]["datasets"]):
      regexp = rf"(?P<mip>\d+){dsep}"
    
    if self.chunk_key_encoding(mip)["name"] == "default":
      regexp += f'c{dsep}'

    groups = []
    for axis in axes:
      groups.append(fr"(?P<{axis['name']}>-?\d+)")

    regexp += self.dimension_separator(mip).join(groups)

    return re.compile(regexp)

  def directory_separator(self):
    cf = CloudFiles(self.cloudpath)
    dsep = '/'
    if cf.protocol == "file":
      dsep = os.path.sep  
    if dsep == '\\':
      dsep = '\\\\' # compensate for regexp escaping
    return dsep

  def chunk_key_encoding(self, mip):
    return self.zarrays[mip].get("chunk_key_encoding", {
      "name": "default",
      "configuration": { "separator": "/" },
    })

  def dimension_separator(self, mip):
    data = self.chunk_key_encoding(mip)
    config = data.get("configuration", { "separator": "/" })
    return config.get("separator", "/")

  def commit_info(self):
    self.render_zarr_metadata()

    cf = CloudFiles(self.cloudpath, secrets=self.config.secrets)

    to_upload = []
    if self.is_group():
      for i, zarray in enumerate(self.zarrays):
        to_upload.append(
          (cf.join(str(i), "zarr.json"), zarray)
        )

    zinfo = copy.deepcopy(self.zinfo)
    ome = copy.deepcopy(self.ome)
    ome["version"] = "0.5"

    if 'attributes' in zinfo:
      zinfo["attributes"]["ome"] = ome
    else:
      zinfo["attributes"] = { "ome": ome }

    to_upload.append(
      ( "zarr.json", zinfo )
    )

    compress = "br"
    if cf.protocol == "file":
      compress = False # otherwise zarr can't read the file

    cf.put_jsons(to_upload, compress=compress)

  def to_zarr_volume_size(self, mip):
    axes = self.axes()
      
    shape = []
    for axis in axes:
      if axis["type"] == "channel":
        shape.append(self.num_channels)
      elif axis["type"] == "time":
        shape.append(1)
      elif axis["type"] == "space" and axis["name"] == "x":
        shape.append(self.volume_size(mip)[0])
      elif axis["type"] == "space" and axis["name"] == "y":
        shape.append(self.volume_size(mip)[1])
      elif axis["type"] == "space" and axis["name"] == "z":
        shape.append(self.volume_size(mip)[2])

    return shape

  def zarr_axes_to_cv_axes(self):
    axes = self.axes()

    shape = []
    for i, axis in enumerate(axes):
      if axis["type"] == "channel":
        shape.append(3)
      elif axis["type"] == "time":
        shape.append(4)
      elif axis["type"] == "space" and axis["name"] == "x":
        shape.append(0)
      elif axis["type"] == "space" and axis["name"] == "y":
        shape.append(1)
      elif axis["type"] == "space" and axis["name"] == "z":
        shape.append(2)

    return shape

  def cv_axes_to_zarr_axes(self):
    seq = self.zarr_axes_to_cv_axes()
    shape = [0] * len(seq)
    for i, val in enumerate(seq):
      shape[val] = i
    return shape

  def render_zarr_metadata(self):
    """Convert the current info file into zarr metadata."""
    datasets = []

    while len(self.zarrays) < len(self.scales):
      self.zarrays.append({})

    for mip, scale in enumerate(self.scales):

      scale_params = []
      chunk_params = []
      translation_params = []
      for axis in self.axes():
        if axis["type"] == "channel":
          scale_params.append(1.0)
          chunk_params.append(self.num_channels)
          translation_params.append(0.0)
        elif axis["type"] == "time":
          scale_params.append(1.0)
          chunk_params.append(self.time_chunk_size(mip))
          translation_params.append(0.0)
        elif axis["type"] == "space" and axis["name"] == "x":
          scale_params.append(scale["resolution"][0])
          chunk_params.append(self.chunk_size(mip)[0])
          translation_params.append(scale_params[-1] * scale["voxel_offset"][0])
        elif axis["type"] == "space" and axis["name"] == "y":
          scale_params.append(scale["resolution"][1])
          chunk_params.append(self.chunk_size(mip)[1])
          translation_params.append(scale_params[-1] * scale["voxel_offset"][1])
        elif axis["type"] == "space" and axis["name"] == "z":
          scale_params.append(scale["resolution"][2])
          chunk_params.append(self.chunk_size(mip)[2])
          translation_params.append(scale_params[-1] * scale["voxel_offset"][2])

      dataset = {
        "coordinateTransformations": [
          {
            "scale": scale_params,
            "type": "scale"
          },
          {
            "translation": translation_params,
            "type": "translation",
          },
        ],
        "path": str(mip),
      }
      datasets.append(dataset)

      zscale = self.zarrays[mip] or {}

      if self.data_type not in ZARR3_VALID_DATATYPES:
        raise ValueError(f"{self.data_type} is not a valid zarr3 data type.")

      zscale["data_type"] = self.data_type

      zscale["chunk_grid"] = {
        "name": "regular", # core
        "configuration": { 
          "chunk_shape": [ int(x) for x in chunk_params ],
        }
      }
      zscale["shape"] = self.to_zarr_volume_size(mip)

      zscale["fill_value"] = self.background_color(mip)

      zscale["zarr_format"] = 3
      zscale["chunk_key_encoding"] = zscale.get("chunk_key_encoding", {
        "name": "default",
        "configuration": {
          "separator": "/",
        },
      })

      # TODO: In the future, figure out how to make it easier to 
      # render this from info file scales. There's a number of
      # features that don't really translate across the formats
      # like e.g. transpose, crc32, codecs.
      zscale["codecs"] = zscale.get("codecs", DEFAULT_CODEC)

      self.zarrays[mip] = zscale

    self.ome["multiscales"][0]["datasets"] = datasets

    self.zinfo = self.zarrays[0]
    if self.is_group():
      self.zinfo["node_type"] = "group"
    else:
      self.zinfo["node_type"] = "array"


  def zarr_to_info(self, zarrays):
    def extract_spatial(attr, dtype):
      spatial = np.ones([3], dtype=dtype)
      scale = self.ome["multiscales"][0]
      axes = scale["axes"]
      
      for axis, res in zip(axes, attr):
        if axis["type"] != "space":
          continue
        if axis["name"] == "x":
          spatial[0] = res
        elif axis["name"] == "y":
          spatial[1] = res
        elif axis["name"] == "z":
          spatial[2] = res

      return spatial

    def chunk_size_mip(mip:int):
      zchunk_size = zarrays[mip]["chunk_grid"]["configuration"]["chunk_shape"]
      chunk_size = zchunk_size[::-1]
      if 'dimension_names' in zarrays[mip]:
        chunk_size = [0,0,0]
        for i, axis in enumerate(zarrays[mip]['dimension_names']):
          if axis == 'x':
            chunk_size[0] = zchunk_size[i]
          elif axis == 'y':
            chunk_size[1] = zchunk_size[i]
          elif axis == 'z':
            chunk_size[2] = zchunk_size[i]
      return chunk_size

    def extract_spatial_size(mip:int):
      shape = zarrays[mip]["shape"]
      return extract_spatial(shape, int)

    try:
      num_channels = len([ 
        chan for chan in self.ome["channels"] if chan["active"] 
      ])
    except KeyError:
      num_channels = 1

    if not zarrays:
      raise exceptions.InfoUnavailableError("Missing mip level zarr.json")

    base_res = self.spatial_resolution_in_nm(0)

    encoding = zarrays[0]["codecs"][0].get("name", "raw")
    if encoding == "bytes":
      encoding = "raw"

    info = PrecomputedMetadata.create_info(
      num_channels=num_channels,
      layer_type='image',
      data_type=str(np.dtype(zarrays[0]["data_type"])),
      encoding=encoding,
      resolution=base_res,
      voxel_offset=self.compute_voxel_offset(0),
      volume_size=extract_spatial_size(0),
      chunk_size=chunk_size_mip(0),
    )

    num_mips = len(zarrays)

    for mip in range(1, num_mips):
      res = self.spatial_resolution_in_nm(mip)
      factor = np.round(res / base_res)

      zarray = zarrays[mip]

      if zarray is None:
        continue

      self.add_scale(
        factor,
        chunk_size=chunk_size_mip(mip),
        encoding=encoding,
        info=info
      )

    return info

  def key(self, mip:int) -> str:
    datasets = self.ome["multiscales"][0]["datasets"]
    if len(datasets) == 0:
      return ''
    return datasets[mip].get("path", mip+1)

  def fetch_info(self):
    cf = CloudFiles(self.cloudpath, secrets=self.config.secrets)
    self.zinfo = cf.get_json("zarr.json")

    if self.zinfo is None or self.zinfo == {}:
      raise exceptions.InfoUnavailableError("No zarr.json file was found.")

    if self.zarr_format != 3:
      raise exceptions.UnsupportedFormatError(
        f"zarr3 module cannot parse zarr format version {self.zarr_format}."
      )

    datasets = []
    
    if "attributes" in self.zinfo:
      try:
        # Detect OME-Zarr 0.5
        # https://ngff.openmicroscopy.org/0.5/#metadata
        self.ome = self.zinfo["attributes"]["ome"]
        assert self.ome["version"] == "0.5" # < 1 version number means anything can change...
      except KeyError: 
        try:
          # OME-Zarr 0.4
          # https://ngff.openmicroscopy.org/0.4/
          self.ome = self.zinfo["attributes"]
          assert self.ome["version"] == "0.4" # < 1 version number means anything can change...
        except KeyError:
          self.ome = self.default_attributes(3) # make a guess at least...

    datasets = self.ome["multiscales"][0]["datasets"]

    zarray_paths = [
      f"{ds.get('path', i+1)}/zarr.json" for i, ds in enumerate(datasets)
    ]
    res = cf.get_json(zarray_paths)

    if res:
      self.zarrays.extend(res)
    else:
      self.zarrays.append(self.zinfo)

    return self.zarr_to_info(self.zarrays)

  def zarr_chunk_size(self, mip:int):
    cs = self.chunk_size(mip)

    attr = []
    for axis in self.axes():
      if axis["name"] == 't':
        attr.append(self.time_chunk_size(mip))
      elif axis["name"] == 'c':
        attr.append(self.num_channels)
      elif axis["name"] == 'x':
        attr.append(cs[0])
      elif axis["name"] == 'y':
        attr.append(cs[1])
      elif axis["name"] == 'z':
        attr.append(cs[2])

    return attr

  def commit_provenance(self):
    """Zarr doesn't support provenance files."""
    pass

  def fetch_provenance(self):
    """Zarr doesn't support provenance files."""
    pass

# Example zarr.json array file

# {
#   "chunk_grid": {
#     "configuration":{
#       "chunk_shape":[64,64,64]
#     },
#     "name":"regular"
#   },
#   "chunk_key_encoding":{
#     "name":"default"
#   },
#   "codecs":[{
#     "configuration":{"endian":"little"},
#     "name":"bytes"
#   }],
#   "data_type": "uint32",
#   "fill_value": 0,
#   "node_type": "array",
#   "shape": [1460,1491,3281],
#   "zarr_format":3
# }

# Example zarr.json group file

# {
#     "zarr_format": 3,
#     "node_type": "group",
#     "attributes": {
#         "ome": {
#             "version": "0.5",
#             "multiscales": [
#                 {
#                     "axes": [
#                         {
#                             "name": "c",
#                             "type": "channel"
#                         },
#                         {
#                             "name": "x",
#                             "type": "space",
#                             "unit": "nanometer"
#                         },
#                         {
#                             "name": "y",
#                             "type": "space",
#                             "unit": "nanometer"
#                         },
#                         {
#                             "name": "z",
#                             "type": "space",
#                             "unit": "nanometer"
#                         }
#                     ],
#                     "datasets": [
#                         {
#                             "path": "1",
#                             "coordinateTransformations": [
#                                 {
#                                     "type": "scale",
#                                     "scale": [
#                                         1.0,
#                                         650.0,
#                                         748.0,
#                                         748.0
#                                     ]
#                                 }
#                             ]
#                         },
#                         {
#                             "path": "2",
#                             "coordinateTransformations": [
#                                 {
#                                     "type": "scale",
#                                     "scale": [
#                                         1.0,
#                                         1300.0,
#                                         1496.0,
#                                         1496.0
#                                     ]
#                                 }
#                             ]
#                         },
#                         {
#                             "path": "4",
#                             "coordinateTransformations": [
#                                 {
#                                     "type": "scale",
#                                     "scale": [
#                                         1.0,
#                                         2600.0,
#                                         2992.0,
#                                         2992.0
#                                     ]
#                                 }
#                             ]
#                         },
#                         {
#                             "path": "8",
#                             "coordinateTransformations": [
#                                 {
#                                     "type": "scale",
#                                     "scale": [
#                                         1.0,
#                                         5200.0,
#                                         5984.0,
#                                         5984.0
#                                     ]
#                                 }
#                             ]
#                         }
#                     ]
#                 }
#             ]
#         }
#     }
# }

# Example sub zarr.json

# {"chunk_grid":{"configuration":{"chunk_shape":[1,4096,4096,32]},"name":"regular"},"chunk_key_encoding":{"name":"default"},"codecs":[{"configuration":{"chunk_shape":[1,32,32,32],"codecs":[{"configuration":{"order":[3,2,1,0]},"name":"transpose"},{"configuration":{"endian":"little"},"name":"bytes"},{"configuration":{"checksum":true,"level":5},"name":"zstd"}],"index_codecs":[{"configuration":{"endian":"little"},"name":"bytes"},{"name":"crc32c"}]},"name":"sharding_indexed"}],"data_type":"uint16","dimension_names":["c","x","y","z"],"fill_value":0,"node_type":"array","shape":[1,4096,4096,2400],"zarr_format":3}

