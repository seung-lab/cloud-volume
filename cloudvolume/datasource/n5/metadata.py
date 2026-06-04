from cloudfiles import CloudFiles

from cloudvolume.datasource.precomputed.metadata import PrecomputedMetadata
from cloudvolume.lib import jsonify, Vec, Bbox

from ...provenance import DataLayerProvenance

class N5Metadata(PrecomputedMetadata):
  def __init__(self, cloudpath, config, cache, info=None):
    
    # some default values, to be overwritten
    tmp_info = PrecomputedMetadata.create_info(
      num_channels=1, layer_type='image', data_type='uint8', 
      encoding='raw', resolution=[1,1,1], voxel_offset=[0,0,0], 
      volume_size=[1,1,1]
    )

    super().__init__(
      cloudpath, config, cache, info=tmp_info, provenance=None
    )
    self.attributes = {
      "root": {},
      "scales": {},
    }

    if info:
      self.info = info
    else:
      self.info = self.fetch_info()

    self.provenance = DataLayerProvenance()

  def info_to_attributes(self):
    return {
      'root': {
        'pixelResolution': {
          'unit': 'nm',
          'dimensions': self.info['scales'][0]['resolution'],
        },
        'scales': [
          list(self.downsample_ratio(i)) for i in self.available_mips
        ],
      },
      'scales': [
        {
          'dataType': self.data_type,
          'compression': { 
            'type': self.config.compress,
            'level': self.config.compress_level,
          },
          'blockSize': list(scale['chunk_sizes'][0]),
          'dimensions': list(scale['volume_size']),
        }
        for scale in self.scales
      ],
    }

  def commit_info(self):
    """We only are supporing read-only."""
    self.attributes = self.info_to_attributes()

    cf = CloudFiles(self.cloudpath, secrets=self.config.secrets)
    cf.put_json("attributes.json", self.attributes["root"])

    cf.put_jsons([
      (cf.join(f"s{i}", "attributes.json"), scale)
      for i, scale in enumerate(self.attributes["root"]["scales"]) 
    ])

  def fetch_info(self):
    cf = CloudFiles(self.cloudpath, secrets=self.config.secrets)
    self.attributes["root"] = cf.get_json("attributes.json")

    if 'pixelResolution' in self.attributes["root"]:
      resolution = self.attributes["root"]["pixelResolution"]["dimensions"]
    else:
      resolution = self.attributes["root"]["resolution"]

    try:
      num_scales = len(self.attributes["root"]["scales"])
    except KeyError:
      num_scales = len(self.attributes["root"]["downsamplingFactors"])

    scale_dirs = [ 
      cf.join(f"s{i}", "attributes.json") 
      for i in range(num_scales) 
    ]
    scale_attrs = cf.get_json(scale_dirs)
    self.attributes["scales"] = scale_attrs
    
    # glossing over that each scale can have 
    # a different data type, but usually it 
    # should all be the same
    data_type = scale_attrs[0]["dataType"] 

    info = PrecomputedMetadata.create_info(
      num_channels=1,
      layer_type="image",
      data_type=data_type,
      encoding="raw",
      resolution=resolution,
      voxel_offset=[0,0,0],
      volume_size=scale_attrs[0]["dimensions"][:3],
      chunk_size=scale_attrs[0]["blockSize"],
    )
    
    for i, scale in enumerate(scale_attrs[1:]):
      try:
        ds_factor = scale["downsamplingFactors"]
      except KeyError:
        ds_factor = self.attributes["root"]["downsamplingFactors"][i+1]

      self.add_scale(
        ds_factor,
        chunk_size=scale["blockSize"],
        encoding="raw",
        info=info
      )

    return info

  def compression_type(self, mip:int) -> str:
    """
    Encoding in n5 is overloaded compared to precomputed.
    This includes bitstream compression. The reason is
    that the file has an uncompressed header whereas precomputed,
    the compression is handled by content-encoding.
    """
    compression_dict = self.attributes["scales"][mip].get("compression", { "type": "raw" })
    return compression_dict.get("type", "raw")

  def commit_provenance(self):
    """N5 doesn't support provenance files."""
    pass

  def fetch_provenance(self):
    """N5 doesn't support provenance files."""
    pass