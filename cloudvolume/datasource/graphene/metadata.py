import posixpath
from collections import namedtuple
import json
import re
import requests
from six.moves import urllib

import numpy as np

from ...lib import Vec, Bbox
from ... import exceptions
from ... import paths
from ...secrets import cave_credentials
from ..precomputed import PrecomputedMetadata

VERSION_ORDERING = [  
  '1.0', 'v1'
]
VERSION_MAP = {
  version: i for i, version in enumerate(VERSION_ORDERING)
}

uint64 = np.uint64
GrapheneLabel = namedtuple('GrapheneLabel', ('level', 'x', 'y', 'z', 'segid'))


class GrapheneApiVersion():
  def __init__(self, version):
    self.version = version.lower()
    if self.version == 'table':
      self.version = VERSION_ORDERING[-1]
    elif self.version not in VERSION_MAP:
      raise ValueError("Unknown Graphene API version {}".format(self.version))

  def __eq__(self, rhs):
    return self.version == rhs.version
  def __ne__(self, rhs):
    return self.version != rhs.version
  def __lt__(self, rhs):
    return self.sequence_number() < rhs.sequence_number()
  def __gt__(self, rhs):
    return self.sequence_number() > rhs.sequence_number()
  def __le__(self, rhs):
    return self.sequence_number() <= rhs.sequence_number()
  def __ge__(self, rhs):
    return self.sequence_number() >= rhs.sequence_number()
  def __str__(self):
    return self.version
  def __repr__(self):
    return "GrapheneApiVersion('{}')".format(self.version)

  def sequence_number(self):
    return VERSION_MAP[self.version]

  def path(self, graphene_path):
    if self.version == '1.0':
      return self.legacy_path(graphene_path)
    return self.api_vx_path(graphene_path)

  def table_path(self, graphene_path):
    return posixpath.join(graphene_path.modality, 'table', graphene_path.dataset)

  def legacy_path(self, graphene_path):
    """All /segmentation/1.0/$DATASET paths"""
    return posixpath.join(graphene_path.modality, '1.0', graphene_path.dataset)

  def api_vx_path(self, graphene_path):
    """
    All /segmentation/api/v1/$DATASET paths.

    As of Feb. 2020, these were the latest paths.
    """
    return posixpath.join( 
      graphene_path.modality, 'api', self.version, 'table', graphene_path.dataset
    )

class GrapheneMetadata(PrecomputedMetadata):
  def __init__(
    self, cloudpath, use_https=False, 
    use_auth=True, auth_token=None, 
    agglomerate=False,
    *args, **kwargs
  ):
    self.server_url = cloudpath.replace('graphene://', '')
    self.server_path = extract_graphene_path(self.server_url)
    self.use_https = use_https
    self.agglomerate = agglomerate
    self.auth_header = None
    self.spatial_index = None
    if use_auth:
      self.auth_header = {
        "Authorization": "Bearer %s" % self.parse_token(auth_token)
      }
    kwargs['use_https'] = bool(use_https)
    super(GrapheneMetadata, self).__init__(cloudpath, *args, **kwargs)

    version = self.server_path.version
    if version == 'table':
      version = self.supported_api_versions[-1].version

    self.api_version = GrapheneApiVersion(version)

  def parse_token(self, auth_token):
    token = None
    if auth_token:
      token = auth_token
    else:
      token = cave_credentials(self.server_path.fqdn)

    if isinstance(token, str):
      try:
        token = json.loads(token)
      except json.decoder.JSONDecodeError:
        pass

    if isinstance(token, dict):
      token = token.get("token", None)

    if not token:
      raise exceptions.AuthenticationError(
        "No Graphene authentication token was provided. "
        "Does ~/.cloudvolume/secrets/cave-secret.json exist?"
      )
    elif not (re.match(r'^[0-9a-f]+$', token) or re.match(r'^[A-Za-z0-9+/]+={0,2}$', token)):
      raise exceptions.AuthenticationError("Graphene authentication token was not formatted correctly. It should either be a hexadecimal or base64 string.")
    return token

  def supports_api(self, version):
    return GrapheneApiVersion(version) in self.supported_api_versions

  @property  
  def supported_api_versions(self):
    versions = [ 
      GrapheneApiVersion(VERSION_ORDERING[i]) \
      for i in self.info['app']['supported_api_versions'] 
    ]
    versions.sort(key=lambda ver: ver.sequence_number())
    return versions

  @property
  def base_path(self):
    path = self.server_path
    return f"{path.scheme}://{path.fqdn}/"

  @property
  def table_path(self):
    return posixpath.join(self.base_path, self.server_path.modality, 'table', self.server_path.dataset)

  @property
  def info_path(self):
    """e.g. https://SUBDOMAIN.dynamicannotationframework.com/segmentation/table/DATASET/info"""
    return posixpath.join(self.table_path, 'info')

  def fetch_info(self):
    """
    Reads info from chunkedgraph endpoint and extracts relevant information
    """
    r = requests.get(self.info_path, headers=self.auth_header)
    r.raise_for_status()
    return r.json()

  @property
  def mesh_path(self):
    if 'mesh' in self.info:
      return self.info['mesh']
    return 'mesh'

  @property
  def cloudpath(self):
    data_dir = self.info['data_dir']
    if self.use_https:
      data_dir = paths.to_https_protocol(data_dir)
    return data_dir

  @property
  def fan_out(self):
    """Number of chunks agglomerated into a new chunk per a level increase in the graph."""
    graph_object = self.info['graph']
    return int(graph_object.get('fan_out', 2))

  def decode_label(self, label):
    level = self.decode_layer_id(label)
    x,y,z = self.decode_chunk_position(label)
    segid = self.decode_segid(label)
    return GrapheneLabel(level, x, y, z, segid)

  def decode_segid(self, label):
    label = uint64(label)
    level = self.decode_layer_id(label)
    segid_bits = self.segid_bits(level)

    mask = uint64(2 ** segid_bits) - uint64(1)
    
    return label & mask

  def decode_chunk_id(self, label):
    """The chunk id is a graphene label with the segid portion zeroed out."""
    label = uint64(label)
    level = self.decode_layer_id(label)
    bits = self.segid_bits(level)
    mask = uint64(2 ** bits) - uint64(1)
    return label & ~mask

  def decode_chunk_position_number(self, label):
    """Returns the chunk position X,Y,Z as a packed uint64."""
    label = uint64(label)
    level = self.decode_layer_id(label)
    ct = self.spatial_bit_count(level)
    label = label & uint64(0x00ffffffffffffff)
    return label >> uint64(self.segid_bits(level))

  def decode_chunk_position(self, label):
    """Returns the chunk position as a tuple (X,Y,Z)"""
    label = uint64(label)
    level = self.decode_layer_id(label)
    ct = self.spatial_bit_count(level)
    label = label & uint64(0x00ffffffffffffff)
    masks = self.spatial_bit_masks(level)
    segid_bits = self.segid_bits(level)

    x = (label & masks[0]) >> uint64(segid_bits + 2 * ct)
    y = (label & masks[1]) >> uint64(segid_bits + 1 * ct)
    z = (label & masks[2]) >> uint64(segid_bits + 0 * ct)

    return Vec(x,y,z)

  def point_to_chunk_position(self, pt, mip=None):
    """
    Convert a point into the chunk position.

    pt: x,y,z triple
    mip: 
      if None, pt is in physical coordinates
      else pt is in the coordinates of the indicated mip level

    Returns: Vec(chunk_x,chunk_y,chunk_z)
    """
    pt = Vec(*pt, dtype=np.float)

    if mip is not None:
      pt *= self.resolution(mip)

    pt /= self.resolution(self.watershed_mip)

    if self.chunks_start_at_voxel_offset:
      pt -= self.voxel_offset(self.watershed_mip)

    return (pt // self.graph_chunk_size).astype(np.int32)

  def point_to_chunk_bbox(self, pt, mip=None):
    """
    For a given point, get the Bbox of the containing 
    chunk.

    pt: x,y,z triple
    mip: 
      if None, pt is in physical coordinates
      else pt is in the coordinates of the indicated mip level

    Returns: Bbox in voxels
    """
    pos = self.point_to_chunk_position(pt, mip)
    pos *= self.graph_chunk_size
    if self.chunks_start_at_voxel_offset:
      pos += self.voxel_offset(self.watershed_mip)

    return Bbox(pos, pos + self.graph_chunk_size)

  def segid_bits(self, level):
    ct = self.spatial_bit_count(level)
    return uint64(64 - self.n_bits_for_layer_id - 3 * ct)

  def decode_layer_id(self, label):
    return uint64(label) >> uint64(64 - self.n_bits_for_layer_id)

  def encode_label(self, layer, x, y, z, segid):
    """
    Create a graphene label from the specified values.

    Another way to use this:

    glabel = GrapheneLabel(2,1,1,1,777)
    meta.encode_label(*glabel)
    """
    if layer > self.n_layers:
      raise ValueError("Provided layer %d is greater than the number of layers in the dataset: %d" % layer, self.n_layers)

    layer_offset = uint64(64 - self.n_bits_for_layer_id)
    bits_per_dim = uint64(self.spatial_bit_count(layer))
    x_offset = uint64(layer_offset - bits_per_dim)
    y_offset = uint64(x_offset - bits_per_dim)
    z_offset = uint64(y_offset - bits_per_dim)

    if not (
      x < 2 ** bits_per_dim and y < 2 ** bits_per_dim and z < 2 ** bits_per_dim
    ):
      raise ValueError(
        "Chunk coordinate is out of range for "
        "this graph on layer %d with %d bits/dim. "
        "[%d, %d, %d]; max = %d."
        % (layer, bits_per_dim, x, y, z, 2 ** bits_per_dim)
      )

    if segid >= 2 ** self.segid_bits(layer):
      raise ValueError(
        "segid {} provided is out of range. It must be less than {}".format(
          segid, 2 ** self.segid_bits(layer)
      ))

    layer = uint64(layer)
    x, y, z = uint64(x), uint64(y), uint64(z)
    segid = uint64(segid)

    return uint64(
      layer << layer_offset | x << x_offset | y << y_offset | z << z_offset | segid
    )

  def spatial_bit_masks(self, level):
    ct = self.spatial_bit_count(level)

    mask = uint64(2 ** ct) - uint64(1)
    segid_bits = 64 - self.n_bits_for_layer_id - 3 * ct

    return [
      mask << uint64(segid_bits + 2 * ct),
      mask << uint64(segid_bits + 1 * ct),
      mask << uint64(segid_bits + 0 * ct)
    ]

  def spatial_bit_count(self, level):
    """
    64-bit labels

    8-bit  chunk coord
    layer | X | Y | Z | segid

    This method returns how many bits in X,Y,Z
    """
    return int(self.info['graph']['spatial_bit_masks'][str(level)])

  @property
  def n_bits_for_layer_id(self):
    return int(self.info['graph'].get('n_bits_for_layer_id', 8))

  @property
  def n_layers(self):
    return int(self.info['graph']['n_layers'])

  @property
  def graph_chunk_size(self):
    return self.info['graph']['chunk_size']

  @property
  def uses_new_draco_bin_size(self):
    graph_object = self.info['graph']
    return int(graph_object.get('uses_new_draco_bin_size', False))
  
  @property
  def mesh_chunk_size(self):
    # TODO: add this as new parameter to the info as it can be different from the chunkedgraph chunksize
    return self.graph_chunk_size

  @property
  def manifest_endpoint(self):
    pth = self.server_path
    pth = GraphenePath(
      pth.scheme, pth.fqdn,
      'meshing', pth.version, pth.dataset
    )

    url = self.api_version.path(pth)
    return posixpath.join(self.base_path, url, 'manifest')

  @property
  def chunks_start_at_voxel_offset(self):
    """
    Boolean property specifying whether ChunkedGraph chunks begin
    at voxel offset or at origin.
    """
    if 'chunks_start_at_voxel_offset' in self.info:
      return self.info["chunks_start_at_voxel_offset"]
    return False

  @property
  def mesh_metadata(self):
    if 'mesh_metadata' in self.info:
      return self.info["mesh_metadata"]
    return None

  @property
  def uniform_draco_grid_size(self):
    """
    If not None, a number that specifies the draco_grid_size at every ChunkedGraph level.
    """
    if self.mesh_metadata and 'uniform_draco_grid_size' in self.mesh_metadata:
      return self.mesh_metadata["uniform_draco_grid_size"]
    return None

  @property
  def max_meshed_layer(self):
    """
    The highest level in the ChunkedGraph that we create meshes for in this dataset.
    """
    if self.mesh_metadata and 'max_meshed_layer' in self.mesh_metadata:
      return self.mesh_metadata["max_meshed_layer"]
    return None

  @property
  def watershed_mip(self):
    """mip level of the base segmentation that all chunk graph operations remap."""
    return self.info["graph"]["cv_mip"]

  def get_draco_grid_size(self, level):
    """
    Returns the draco_grid_size for specified ChunkedGraph level.
    """
    if self.mesh_metadata is None:
      raise ValueError('This layer is not draco meshed')
    if self.uniform_draco_grid_size is not None:
      return self.uniform_draco_grid_size
    if self.mesh_metadata["max_meshed_layer"] < level:
      raise ValueError(
        "Request level",
        level,
        ". But the maximum meshed level is ",
        self.mesh_metadata["max_meshed_layer"],
      )
    return self.mesh_metadata["draco_grid_sizes"][str(level)]

GraphenePath = namedtuple('GraphenePath', ('scheme', 'fqdn', 'modality', 'version', 'dataset'))
LEGACY_EXTRACTION_RE = re.compile(r'/?(\w+)/([\d\.]+)/([\w\d\.\_\-]+)/?')
API_VX_EXTRACTION_RE = re.compile(r'/?(\w+)/api/(v[\d\.]+)/([\w\d\.\_\-]+)/?')
LATEST_API_EXTRACTION_RE = re.compile(r'/?(\w+)/(table)/([\w\d\.\_\-]+)/?')

def extract_graphene_path(url):
  """
  Examples:
  Legacy endpoint:
    graphene://https://SUBDOMAIN.dynamicannotationframework.com/segmentation/1.0/DATASET
  Newer endpoint:
    graphene://https://SUBDOMAIN.dynamicannotationframework.com/segmentation/api/v1/DATASET
  Latest endpoint:
    graphene://https://SUBDOMAIN.DOMAIN_DOT_COM/segmentation/table/DATASET
  """
  parse = urllib.parse.urlparse(url)

  schemes = [ 
    LATEST_API_EXTRACTION_RE, API_VX_EXTRACTION_RE, LEGACY_EXTRACTION_RE 
  ]

  for scheme in schemes:
    match = re.match(scheme, parse.path)
    if match:
      break
  else:
    raise exceptions.UnsupportedFormatError("Unable to parse Graphene URL: " + url)

  modality, version, dataset = match.groups()
  return GraphenePath(parse.scheme, parse.netloc, modality, version, dataset)

