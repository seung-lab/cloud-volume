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