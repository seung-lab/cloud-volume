import re

import numpy as np

from cloudfiles import CloudFiles

from cloudvolume.datasource.precomputed.metadata import PrecomputedMetadata
from cloudvolume.lib import jsonify, Vec, Bbox

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

class ZarrMetadata(PrecomputedMetadata):
  def __init__(self, cloudpath, config, cache,  info=None):
    
    # some default values, to be overwritten
    info = PrecomputedMetadata.create_info(
      num_channels=1, layer_type='image', data_type='uint8', 
      encoding='raw', resolution=[1,1,1], voxel_offset=[0,0,0], 
      volume_size=[1,1,1]
    )

    super().__init__(
      cloudpath, config, cache, info=info, provenance=None
    )

    self.zarrays = []
    self.zattrs = {}

    self.info = self.fetch_info()
    self.provenance = DataLayerProvenance()

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

      regexp = rf"(?P<mip>\d+){dsep}"

      groups = []
      for axis in axes:
        groups.append(fr"(?P<{axis['name']}>\d+)")

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
        [cf.join(str(i), ".zarray"), zarray]
      )

    to_upload.append(
      [ ".zattrs", self.zattrs ]
    )

    cf.put_jsons(to_upload, compress='br')

  def to_zarr_volume_size(self, mip):
    axes = self.zattrs["multiscales"][0]["axes"]
      
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
    axes = self.zattrs["multiscales"][0]["axes"]

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

  def render_zarr_metadata(self):
    """Convert the current info file into zarr metadata."""
    datasets = []
    for mip, scale in enumerate(info["scales"]):
      dataset = {
        "coordinateTransformations": [
          {
            "scale": [
              1,
              self.num_channels,
              scale["resolution"][2],
              scale["resolution"][1],
              scale["resolution"][0]
            ],
            "type": "scale"
          }
        ],
        "path": str(mip),
      }
      datasets.append(dataset)

      zscale = self.zarrays[mip]
      zscale["dtype"] = CV_TO_ZARR_DTYPE[info["dtype"]]
      zscale["chunks"] = scale["chunk_size"]
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

      # zscale["filters"] = None

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

    def extract_spatial_resolution(mip):
      scale = zattrs["multiscales"][0]
      resolution = scale["datasets"][mip]["coordinateTransformations"][0]["scale"]
      spatial_res = extract_spatial(resolution, np.float32)
      return spatial_res * 1000.0 # um -> nm

    def extract_spatial_size(mip):
      shape = zarrays[mip]["shape"]
      return extract_spatial(shape, int)
      
    def get_full_resolution(mip):
      scale = zattrs["multiscales"][0]
      axes = scale["axes"]
      return np.array(scale["datasets"][mip]["coordinateTransformations"][0]["scale"])

    num_channels = len([ 
      chan for chan in zattrs["omero"]["channels"] if chan["active"] 
    ])

    info = PrecomputedMetadata.create_info(
      num_channels=num_channels,
      layer_type='image',
      data_type=str(np.dtype(zarrays[0]["dtype"])),
      encoding=zarrays[0]["compressor"]["id"],
      resolution=extract_spatial_resolution(0),
      voxel_offset=[0,0,0],
      volume_size=extract_spatial_size(0),
      chunk_size=zarrays[0]["chunks"],
    )

    num_mips = len(zattrs["multiscales"][0]["datasets"])

    for mip in range(1, num_mips):
      prev_res = extract_spatial_resolution(mip-1)
      res = extract_spatial_resolution(mip)

      factor = np.round(res / prev_res)

      self.add_scale(
        factor,
        chunk_size=zarrays[mip]["chunks"],
        encoding=zarrays[mip]["compressor"]["id"],
        info=info
      )

    return info

  def fetch_info(self):
    cf = CloudFiles(self.cloudpath, secrets=self.config.secrets)
    metadata = cf.get_json([ "0/.zarray", ".zattrs" ])

    self.zarrays.append(metadata[0])
    self.zattrs = metadata[1]

    num_mips = len(self.zattrs["multiscales"][0]["datasets"])

    zarray_paths = [
      f"{i}/.zarray" for i in range(1, num_mips)
    ]
    res = cf.get_json(zarray_paths)

    if res is not None:
      self.zarrays.extend(res)

    return self.zarr_to_info(self.zarrays, self.zattrs)

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