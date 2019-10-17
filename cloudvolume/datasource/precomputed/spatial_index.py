import numpy as np

from ...storage import Storage, SimpleStorage
from ... import paths
from ...lib import Bbox, Vec, xyzrange, min2

class SpatialIndex(object):
  def __init__(self, cloudpath, bounds, chunk_size):
    self.cloudpath = cloudpath
    self.path = paths.extract(cloudpath)
    self.bounds = Bbox.create(bounds)
    self.chunk_size = Vec(*chunk_size)

  def join(self, *paths):
    if self.path.protocol == 'file':
      return os.path.join(*paths)
    else:
      return posixpath.join(*paths)    

  def query(self, bbox):
    bbox = Bbox.create(bbox, context=self.bounds, autocrop=True)
    original_bbox = bbox.clone()
    bbox = bbox.expand_to_chunk_size(self.chunk_size, offset=self.bounds.minpt)

    if bbox.subvoxel():
      return []

    index_files = []
    for pt in xyzrange(bbox.minpt, bbox.maxpt, self.chunk_size):
      search = Bbox( pt, min2(pt + self.chunk_size, self.bounds) )
      index_files.append(search.to_filename() + '.spatial')

    with Storage(self.cloudpath, progress=True) as stor:
      results = stor.get_files(index_files)

    labels = set()
    for i, res in enumerate(results):
      if res['error'] is not None:
        raise LookupError(res['error'])

      res = json.loads(res['content'])
      for label, label_bbx in res.items():
        label = int(label)
        label_bbx = Bbox.from_list(label_bbx)

        if Bbox.intersects(label_bbx, original_bbox):
          labels.update(label)

    return labels








    



