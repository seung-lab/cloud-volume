import posixpath
from collections import namedtuple
import json
import re
import requests
from six.moves import urllib

from ... import exceptions
from ... import paths
from ...secrets import chunkedgraph_credentials
from ..precomputed import PrecomputedMetadata

class GrapheneMetadata(PrecomputedMetadata):
  def __init__(self, cloudpath, use_https=False, use_auth=True, auth_token=None, *args, **kwargs):
    self.server_url = cloudpath.replace('graphene://', '')
    self.server_path = extract_graphene_path(self.server_url)
    self.use_https = use_https
    self.auth_header = None
    if use_auth:
      token = None
      if chunkedgraph_credentials:
        token = chunkedgraph_credentials["token"]
      if auth_token:
        token = auth_token
      self.auth_header = {
        "Authorization": "Bearer %s" % token
      }
    super(GrapheneMetadata, self).__init__(cloudpath, *args, **kwargs)

  def fetch_info(self):
    """
    Reads info from chunkedgraph endpoint and extracts relevant information
    """
    r = requests.get(posixpath.join(self.server_url, "info"), headers=self.auth_header)
    r.raise_for_status()
    return r.json()

  @property
  def cloudpath(self):
    data_dir = self.info['data_dir']
    if self.use_https:
      data_dir = paths.to_https_protocol(data_dir)
    return data_dir

  @property
  def graph_chunk_size(self):
    return self.info["graph"]["chunk_size"]
  
  @property
  def mesh_chunk_size(self):
    # TODO: add this as new parameter to the info as it can be different from the chunkedgraph chunksize
    return self.graph_chunk_size

  @property
  def manifest_endpoint(self):
    pth = self.server_path
    url = pth.scheme + '://' + pth.subdomain + '.' + pth.domain
    return url + '/' + posixpath.join('meshing', pth.version, pth.dataset, 'manifest')

GraphenePath = namedtuple('GraphenePath', ('scheme', 'subdomain', 'domain', 'modality', 'version', 'dataset'))
EXTRACTION_RE = re.compile(r'/?(\w+)/([\d.]+)/([\w\d\.\_\-]+)/?')

def extract_graphene_path(url):
  parse = urllib.parse.urlparse(url)
  subdomain = parse.netloc.split('.')[0]
  domain = '.'.join(parse.netloc.split('.')[1:])

  match = re.match(EXTRACTION_RE, parse.path)
  if not match:
    raise exceptions.UnsupportedFormatError("Unable to parse Graphene URL: " + url)

  modality, version, dataset = match.groups()
  return GraphenePath(parse.scheme, subdomain, domain, modality, version, dataset)





