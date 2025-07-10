import re
import os

import numpy as np

from cloudfiles import CloudFiles

from cloudvolume.datasource.precomputed.metadata import PrecomputedMetadata
from cloudvolume.lib import jsonify, Vec, Bbox, spatial_unit_in_meters, time_unit_in_seconds

from ... import exceptions
from ...provenance import DataLayerProvenance

CV_TO_ZARR_DTYPE = {
  "int8": "|i1",
  "int16": "<i2",
  "int32": "<i4",
  "int64": "<i8",

  "uint8": "|u1",
  "uint16": "<u2",
  "uint32": "<u4",
  "uint64": "<u8",

  "float32": "<f4",
  "float64": "<f8",
}

class Zarr2Metadata(PrecomputedMetadata):
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
    self.zattrs = self.default_zattrs()
    self._is_group = False

    if orig_info is None:
      self.info = self.fetch_info()
    else:
      self.render_zarr_metadata()

    self.provenance = DataLayerProvenance()

  @property
  def zarr_format(self):
    return self.zinfo["zarr_format"]

  def is_group(self):
    return self._is_group

  def default_zattrs(self, num_dims = 5):
    attrs = {
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
              "unit": "micrometer"
            },
            {
              "name": "y",
              "type": "space",
              "unit": "micrometer"
            },
            {
              "name": "x",
              "type": "space",
              "unit": "micrometer"
            }
          ],
          "datasets": [],
          "name": "/",
          "version": "0.4"
        }
      ],
      "omero": {
        "channels": [ {"active": True} ] * self.num_channels
      }
    }

    attrs["multiscales"][0]["axes"] = attrs["multiscales"][0]["axes"][5-num_dims:]

    return attrs

  def default_datasets(self, num_datasets):
    ds = []
    for i in range(num_datasets):
      path = ""
      if self.is_group():
        path = f"{i}"

      ds.append(
        {
          "coordinateTransformations": [
            {
              "scale": [ 1 ] * len(self.zarr_chunk_size(0)),
              "type": "scale"
            }
          ],
          "path": path
        }
      )
    return ds

  def spatial_resolution_in_nm(self, mip, zattrs = None, zarrays = None):
    if zarrays is None:
      zarrays = self.zarrays
    if zattrs is None:
      zattrs = self.zattrs

    scale_factors = np.ones([3], dtype=np.float32)
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

    resolution = self.datasets()[mip]["coordinateTransformations"][0]["scale"]
    resolution = np.array([
      resolution[positions[0]],
      resolution[positions[1]],
      resolution[positions[2]]
    ], dtype=np.float32)

    return resolution * (scale_factors / 1e-9)

  def time_resolution_in_seconds(self, mip:int) -> float:
    i = 0
    unit = None
    for axis in self.axes():
      if axis["type"] == "time":
        unit = axis["unit"]
        break
      i += 1

    if unit is None:
      return 1.0

    scale_factor = time_unit_in_seconds(unit)
    resolution = self.datasets()[mip]["coordinateTransformations"][0]["scale"]
    return resolution[i] * scale_factor

  def has_time_axis(self):
    try:
      return self.time_index() is not None
    except ValueError:
      return False

  def datasets(self):
    return self.zattrs["multiscales"][0]["datasets"]

  def axes(self):
    return self.zattrs["multiscales"][0]["axes"]

  def time_index(self):
    for i, axis in enumerate(self.axes()):
      if axis["type"] == "time":
        return i
    raise ValueError("No time axis.")

  def time_chunk_size(self, mip):
    i = self.time_index()
    return self.zarrays[mip]["chunks"][i]

  def num_time_chunks(self, mip):
    nframes = self.num_frames(mip)
    try:
      t_chunk_size = self.time_chunk_size(mip)
    except ValueError:
      return 1
    return int(np.ceil(nframes / t_chunk_size))

  def num_frames(self, mip):
    try:
      i = self.time_index()
      return self.zarrays[mip]["shape"][i]
    except ValueError:
      return 1

  def chunk_name(self, mip, *args):
    seq = self.cv_axes_to_zarr_axes()
    sep = self.dimension_separator(mip)

    values = [ str(args[val]) for val in seq ]

    if self.order(mip) == "F":
      values.reverse()

    if self.is_group():
      return sep.join([ self.key(mip), *values ])
    else:
      return sep.join([ *values ])

  def duration_in_seconds(self):
    return self.time_resolution_in_seconds(0) * self.num_frames(0)

  def compressor(self, mip):
    return self.zarrays[mip].get("compressor", {}).get("id", "blosc")

  def order(self, mip):
    return self.zarrays[mip]["order"]

  def background_color(self, mip):
    return self.zarrays[mip].get("fill_value", 0)

  def set_background_color(self, mip):
    self.zarrays[mip]["fill_value"] = 0

  def filename_regexp(self, mip):
      scale = self.zattrs["multiscales"][0]
      axes = scale["axes"]

      cf = CloudFiles(self.cloudpath)
      dsep = '/'
      if cf.protocol == "file":
        dsep = os.path.sep  
      if dsep == '\\':
        dsep = '\\\\' # compensate for regexp escaping

      regexp = rf""
      if self.is_group():
        regexp = rf"(?P<mip>\d+){dsep}"

      groups = []
      for axis in axes:
        groups.append(fr"(?P<{axis['name']}>-?\d+)")

      regexp += self.dimension_separator(mip).join(groups)

      return re.compile(regexp)

  def dimension_separator(self, mip):
    return self.zarrays[mip].get("dimension_separator", ".")

  def commit_info(self):
    self.render_zarr_metadata()

    cf = CloudFiles(self.cloudpath, secrets=self.config.secrets)

    to_upload = []
    for i, zarray in enumerate(self.zarrays):
      to_upload.append(
        (cf.join(str(i), ".zarray"), zarray)
      )

    to_upload.append(
      ( ".zattrs", self.zattrs )
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
      dataset = {
        "coordinateTransformations": [
          {
            "scale": [
              1,
              self.num_channels,
              scale["resolution"][2] / 1000,
              scale["resolution"][1] / 1000,
              scale["resolution"][0] / 1000
            ],
            "type": "scale"
          }
        ],
        "path": str(mip),
      }
      datasets.append(dataset)

      zscale = self.zarrays[mip] or {}

      zscale["dtype"] = CV_TO_ZARR_DTYPE[self.data_type]
      zscale["chunks"] = [ 1, self.num_channels ] + list(scale["chunk_sizes"][0][::-1])
      zscale["shape"] = self.to_zarr_volume_size(mip)

      zscale["fill_value"] = zscale.get("fill_value", 0)
      zscale["order"] = zscale.get("order", 'C')
      zscale["zarr_format"] = zscale.get("zarr_format", 2)

      zscale["compressor"] = zscale.get("compressor", {
        "blocksize": 0,
        "clevel": 5,
        "cname": "lz4",
        "id": "blosc",
        "shuffle": 1,
      })
      zscale["filters"] = zscale.get("filters", None)
      
      self.zarrays[mip] = zscale

    self.zattrs["multiscales"][0]["datasets"] = datasets

  def zarr_to_info(self, zarrays, zattrs):
    def extract_spatial(attr, dtype):
      scale = zattrs["multiscales"][0]
      axes = scale["axes"]
      
      spatial = np.ones([3], dtype=dtype)
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

    def extract_spatial_size(mip):
      shape = zarrays[mip]["shape"]
      return extract_spatial(shape, int)
      
    def get_full_resolution(mip):
      scale = zattrs["multiscales"][0]
      axes = scale["axes"]
      return np.array(scale["datasets"][mip]["coordinateTransformations"][0]["scale"])

    try:
      num_channels = len([ 
        chan for chan in zattrs["omero"]["channels"] if chan["active"] 
      ])
    except KeyError:
      num_channels = 1

    if not zarrays:
      raise exceptions.InfoUnavailableError()

    base_res = self.spatial_resolution_in_nm(0, zattrs, zarrays)

    def extract_chunk_size(chunk_size):
      chunk_size = list(chunk_size)
      if zarrays[0]["order"] == "C":
        while len(chunk_size) > 3:
          chunk_size.pop(0)
        chunk_size.reverse()
      else:
        chunk_size = chunk_size[:3]
      return chunk_size

    info = PrecomputedMetadata.create_info(
      num_channels=num_channels,
      layer_type='image',
      data_type=str(np.dtype(zarrays[0]["dtype"])),
      encoding=zarrays[0]["compressor"].get("id", "raw"),
      resolution=base_res,
      voxel_offset=[0,0,0],
      volume_size=extract_spatial_size(0),
      chunk_size=extract_chunk_size(zarrays[0]["chunks"]),
    )

    num_mips = len(zattrs["multiscales"][0]["datasets"])

    for mip in range(1, num_mips):
      res = self.spatial_resolution_in_nm(mip, zattrs, zarrays)
      factor = np.round(res / base_res)

      zarray = zarrays[mip]

      if zarray is None:
        continue

      chunk_size = extract_chunk_size(zarray["chunks"])
      self.add_scale(
        factor,
        chunk_size=chunk_size,
        encoding=zarray["compressor"].get("id", "raw"),
        info=info
      )

    return info

  def key(self, mip):
    datasets = self.zattrs["multiscales"][0]["datasets"]
    return datasets[mip].get("path", mip+1)

  def fetch_info(self):
    cf = CloudFiles(self.cloudpath, secrets=self.config.secrets)
    metadata = cf.get_json([ "0/.zarray", ".zattrs" ])

    if metadata[0] is not None:
      self.zarrays.append(metadata[0])
      self._is_group = True
    else:
      self.zarrays = [ cf.get_json(".zarray") ]
      self._is_group = False
      if self.zarrays[0] is None:
        self.zarrays = []
    
    if metadata[1] is not None:
      self.zattrs = metadata[1]
    else:
      self.zattrs = {}

    num_dims = 5
    if len(self.zarrays):
      num_dims = len(self.zarrays[0]["chunks"])

    if self.zattrs is None or "multiscales" not in self.zattrs:
      self.zattrs.update(self.default_zattrs(num_dims))

    datasets = self.datasets()
    
    if datasets == []:
      self.zattrs["multiscales"][0]["datasets"] = self.default_datasets(len(self.zarrays))
      datasets = self.datasets()

    zarray_paths = [
      f"{ds.get('path', i+1)}/.zarray" for i, ds in enumerate(datasets)
    ]
    res = cf.get_json(zarray_paths)
    
    if any([ x is not None for x in res ]):
      self.zarrays.extend(res)

    return self.zarr_to_info(self.zarrays, self.zattrs)

  def zarr_chunk_size(self, mip):
    axes = self.zattrs["multiscales"][0]["axes"]
    cs = self.chunk_size(mip)

    attr = []
    for axis in axes:
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

# Example .zarray JSON file
# {
#     "chunks": [
#         64,
#         128,
#         128
#     ],
#     "compressor": {
#         "blocksize": 0,
#         "clevel": 5,
#         "cname": "lz4",
#         "id": "blosc",
#         "shuffle": 1
#     },
#     "dtype": "|u1",
#     "fill_value": 0,
#     "filters": null,
#     "order": "C",
#     "shape": [
#         512,
#         512,
#         512
#     ],
#     "zarr_format": 2
# }

# Example .zattrs JSON file

# {
#   "multiscales": [
#     {
#       "axes": [
#         {
#           "name": "t",
#           "type": "time",
#           "unit": "millisecond"
#         },
#         {
#           "name": "c",
#           "type": "channel"
#         },
#         {
#           "name": "z",
#           "type": "space",
#           "unit": "micrometer"
#         },
#         {
#           "name": "y",
#           "type": "space",
#           "unit": "micrometer"
#         },
#         {
#           "name": "x",
#           "type": "space",
#           "unit": "micrometer"
#         }
#       ],
#       "datasets": [
#         {
#           "coordinateTransformations": [
#             {
#               "scale": [
#                 1,
#                 1,
#                 1,
#                 0.748,
#                 0.748
#               ],
#               "type": "scale"
#             }
#           ],
#           "path": "0"
#         },
#         {
#           "coordinateTransformations": [
#             {
#               "scale": [
#                 1,
#                 1,
#                 2,
#                 1.496,
#                 1.496
#               ],
#               "type": "scale"
#             }
#           ],
#           "path": "1"
#         },
#         {
#           "coordinateTransformations": [
#             {
#               "scale": [
#                 1,
#                 1,
#                 4,
#                 2.992,
#                 2.992
#               ],
#               "type": "scale"
#             }
#           ],
#           "path": "2"
#         },
#         {
#           "coordinateTransformations": [
#             {
#               "scale": [
#                 1,
#                 1,
#                 8,
#                 5.984,
#                 5.984
#               ],
#               "type": "scale"
#             }
#           ],
#           "path": "3"
#         },
#         {
#           "coordinateTransformations": [
#             {
#               "scale": [
#                 1,
#                 1,
#                 16,
#                 11.968,
#                 11.968
#               ],
#               "type": "scale"
#             }
#           ],
#           "path": "4"
#         },
#         {
#           "coordinateTransformations": [
#             {
#               "scale": [
#                 1,
#                 1,
#                 32,
#                 23.936,
#                 23.936
#               ],
#               "type": "scale"
#             }
#           ],
#           "path": "5"
#         },
#         {
#           "coordinateTransformations": [
#             {
#               "scale": [
#                 1,
#                 1,
#                 64,
#                 47.872,
#                 47.872
#               ],
#               "type": "scale"
#             }
#           ],
#           "path": "6"
#         },
#         {
#           "coordinateTransformations": [
#             {
#               "scale": [
#                 1,
#                 1,
#                 128,
#                 95.744,
#                 95.744
#               ],
#               "type": "scale"
#             }
#           ],
#           "path": "7"
#         }
#       ],
#       "name": "/",
#       "version": "0.4"
#     }
#   ],
#   "omero": {
#     "channels": [
#       {
#         "active": true,
#         "coefficient": 1,
#         "color": "000000",
#         "family": "linear",
#         "inverted": false,
#         "label": "Channel:s3://aind-open-data/exaSPIM_653980_2023-08-10_20-08-29_fusion_2023-08-24/fused.zarr/:0",
#         "window": {
#           "end": 1,
#           "max": 1,
#           "min": 0,
#           "start": 0
#         }
#       }
#     ],
#     "id": 1,
#     "name": "s3://aind-open-data/exaSPIM_653980_2023-08-10_20-08-29_fusion_2023-08-24/fused.zarr/",
#     "rdefs": {
#       "defaultT": 0,
#       "defaultZ": 11573,
#       "model": "color"
#     },
#     "version": "0.4"
#   }
# }
