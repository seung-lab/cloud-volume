import posixpath
from collections import namedtuple
import json
import re
import requests
from six.moves import urllib

from ... import exceptions
from ... import paths
from ..precomputed import PrecomputedMetadata

class GrapheneMetadata(PrecomputedMetadata):
  def __init__(self, cloudpath, *args, **kwargs):
    self.server_url = cloudpath.replace('graphene://', '')
    self.server_path = extract_graphene_path(self.server_url)
    super(GrapheneMetadata, self).__init__(cloudpath, *args, **kwargs)

  def fetch_info(self):
    """
    Reads info from chunkedgraph endpoint and extracts relevant information
    """
    r = requests.get(posixpath.join(self.server_url, "info"))
    r.raise_for_status()
    return json.loads(r.content)

  @property
  def cloudpath(self):
    return self.info['data_dir']

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





