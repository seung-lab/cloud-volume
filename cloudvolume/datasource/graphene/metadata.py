import posixpath

import json
import requests

from ..precomputed import PrecomputedMetadata

class GrapheneMetadata(PrecomputedMetadata):
  
  def fetch_info(self):
    """
    Reads info from chunkedgraph endpoint and extracts relevant information
    """
    r = requests.get(posixpath.join(self.cloudpath, "info"))
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
    return posixpath.join(self.base_cloudpath, 'meshing/manifest')
