from intern.remote.boss import BossRemote
from intern.resource.boss.resource import ChannelResource, ExperimentResource, CoordinateFrameResource
from ...secrets import boss_credentials

from cloudvolume.datasource.precomputed.metadata import PrecomputedMetadata

class BossMetadata(PrecomputedMetadata):
  def __init__(self, cloudpath, cache, info=None):
    super(BossMetadata, self).__init__(
      cloudpath, cache, info=info, provenance=None
    )

  def commit_info(self):
    """BOSS doesn't support editing metadata after creation."""
    pass 

  def fetch_info(self):
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
      chunk_size=(512, 512, 16), # fixed in s3 implementation
    )

    each_factor = Vec(2,2,1)
    if experiment.hierarchy_method == 'isotropic':
      each_factor = Vec(2,2,2)
    
    factor = each_factor.clone()
    for _ in range(1, experiment.num_hierarchy_levels):
      self.add_scale(factor, info=info)
      factor *= each_factor

    return info

  def commit_provenance(self):
    """BOSS doesn't support provenance files."""
    pass

  def fetch_provenance(self):
    """BOSS doesn't support provenance files."""
    pass