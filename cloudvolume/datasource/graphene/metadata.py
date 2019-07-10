import posixpath

import json
import requests

from ..precomputed import PrecomputedMetadata

class GrapheneMetadata(PrecomputedMetadata):
  def __init__(self, cloudpath, *args, **kwargs):
    self.server_url = cloudpath.replace('graphene://', '')
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
  def manifest_endpoint(self):
    return posixpath.join(self.server_url, 'meshing/manifest')
