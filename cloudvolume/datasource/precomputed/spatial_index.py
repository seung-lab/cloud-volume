from collections import defaultdict
import simdjson
import os 
import sqlite3

import numpy as np
from tqdm import tqdm

from cloudfiles import CloudFiles

from ...exceptions import SpatialIndexGapError
from ... import paths
from ...lib import (
  Bbox, Vec, xyzrange, min2, 
  toiter, sip, nvl, getprecision
)

class SpatialIndex(object):
  """
  Implements the client side reader of the 
  spatial index. During data generation, the
  labels in a given task are enumerated and 
  assigned their bounding box as JSON:

  {
    SEGID: [ x,y,z, x,y,z ],
    ...
  }

  The filename is the physical bounding box of the
  task dot spatial.

  e.g. "0-1024_0-1024_0-500.spatial" where the bbox 
  units are nanometers.

  The info file of the data type can then be augmented
  with:

  {
    "spatial_index": { "chunk_size": [ sx, sy, sz ] }
  }

  Where sx, sy, and sz are given in physical dimensions.
  """
  def __init__(
    self, cloudpath, bounds, chunk_size, 
    config=None, sqlite_db=None, resolution=None
  ):
    self.cloudpath = cloudpath
    self.path = paths.extract(cloudpath)
    self.bounds = Bbox.create(bounds)
    self.chunk_size = Vec(*chunk_size)
    self.sqlite_db = sqlite_db # optional DB for higher performance
    self.resolution = None
    self.precision = None

    if resolution is None:
      self.physical_bounds = self.bounds.clone()
    else:
      self.resolution = Vec(*resolution, dtype=float)
      self.precision = max(map(getprecision, resolution))
      if self.precision == 0:
        self.resolution = Vec(*resolution, dtype=int)

      self.physical_bounds = self.bounds.astype(self.resolution.dtype) * self.resolution

    if config is None:
      self.config = {}
      self.config.progress = None
    else:
      self.config = config

  def join(self, *paths):
    if self.path.protocol == 'file':
      return os.path.join(*paths)
    else:
      return posixpath.join(*paths)    

  def fetch_index_files(self, index_files, progress=None):
    progress = nvl(progress, self.config.progress)
    results = CloudFiles(self.cloudpath, progress=progress).get(index_files)

    for res in results:
      if res['error'] is not None:
        raise SpatialIndexGapError(res['error'])

    return { res['filename']: res['content'] for res in results }

  def fetch_all_index_files(self, allow_missing=False, progress=None):
    """Generator returning batches of (filename, json)"""
    all_index_paths = self.index_file_paths_for_bbox(self.bounds)
    
    progress = nvl(progress, self.config.progress)

    N = 500
    pbar = tqdm( 
      total=len(all_index_paths), 
      disable=(not progress), 
      desc="Processing Index"
    )

    for index_paths in sip(all_index_paths, N):
      index_files = self.fetch_index_files(index_paths, progress=False)

      for filename, content in index_files.items():
        if content is None:
          if allow_missing:
            continue
          else:
            raise SpatialIndexGapError(filename + " was not found.")

      yield index_files

      pbar.update(N)
    pbar.close()

  def index_file_paths_for_bbox(self, bbox):
    bbox = bbox.expand_to_chunk_size(self.chunk_size, offset=self.physical_bounds.minpt)

    if bbox.subvoxel():
      return []

    chunk_size = self.chunk_size
    bounds = self.bounds
    resolution = self.resolution
    precision = self.precision

    class IndexPathIterator():
      def __len__(self):
        return bbox.num_chunks(chunk_size)
      def __iter__(self):
        for pt in xyzrange(bbox.minpt, bbox.maxpt, chunk_size):
          search = Bbox( pt, min2(pt + chunk_size, bounds.maxpt) )
          search *= resolution
          yield search.to_filename(precision) + '.spatial'

    return IndexPathIterator()

  def to_sqlite(
    self, database_name="spatial_index.db", 
    create_indices=True, progress=None
  ):
    """
    Create a sqlite database of labels and filenames
    from the JSON spatial_index for faster performance.

    Depending on the dataset size, this could take a while.
    With a dataset with ~140k index files, the DB took over
    an hour to build and was 42 GB.
    """
    progress = nvl(progress, self.config.progress)

    conn = sqlite3.connect(database_name)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE index_files (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      filename TEXT NOT NULL
    )
    """)
    cur.execute("CREATE INDEX idxfname ON index_files (filename)")

    cur.execute("""
    CREATE TABLE file_lookup (
      label INTEGER NOT NULL,
      fid INTEGER NOT NULL REFERENCES index_files(id),
      PRIMARY KEY(label,fid)
    )
    """)

    cur.execute("PRAGMA journal_mode = MEMORY")
    cur.execute("PRAGMA synchronous = OFF")

    parser = simdjson.Parser()

    for index_files in self.fetch_all_index_files(progress=progress):
      for filename, content in index_files.items():
        index_labels = parser.parse(content).keys()
        filename = os.path.basename(filename)
        cur.execute("INSERT INTO index_files(filename) VALUES (?)", (filename,))
        cur.execute("SELECT id from index_files where filename = ?", (filename,))
        fid = cur.fetchone()[0]
        values = ( (int(label), fid) for label in index_labels )
        cur.executemany("INSERT INTO file_lookup(label, fid) VALUES (?,?)", values)
      conn.commit()

    cur.execute("PRAGMA journal_mode = DELETE")
    cur.execute("PRAGMA synchronous = FULL")

    if create_indices:
      if progress:
        print("Creating labels index...")
      cur.execute("CREATE INDEX file_lbl ON file_lookup (label)")

      if progress:
        print("Creating filename index...")
      cur.execute("CREATE INDEX fname ON file_lookup (fid)")

    conn.close()

  def get_bbox(self, label):
    """
    Given a label, compute an enclosing bounding box for it.

    Returns: Bbox in physical coordinates
    """
    locations = defaultdict(list)
    parser = simdjson.Parser()

    label = str(label)
    bbox = None

    if self.sqlite_db:
      conn = sqlite3.connect(self.sqlite_db)
      cur = conn.cursor()
      cur.execute("""
        select index_files.filename  
        from file_lookup, index_files
        where file_lookup.fid = index_files.id
          and file_lookup.label = ?
      """, (label,))
      iterator = [ self.fetch_index_files(( row[0] for row in cur.fetchall() )) ]
      conn.close()
    else:
      iterator = self.fetch_all_index_files()

    for index_files in iterator:
      for filename, content in index_files.items():
        segid_bbox_dict = parser.parse(content)
        filename = os.path.basename(filename)

        if label not in segid_bbox_dict: 
          continue 

        current_bbox = Bbox.from_list(
          np.frombuffer(segid_bbox_dict[label].as_buffer(of_type="i"), dtype=np.int64)
        )

        if bbox is None:
          bbox = current_bbox
        else:
          bbox = Bbox.expand(bbox, current_bbox)

    return bbox

  def file_locations_per_label(self, labels=None, allow_missing=False):
    """
    Queries entire dataset to find which spatial index files the 
    given labels are located in. Can be expensive. If labels is not 
    specified, all labels are fetched.

    If the spatial_index.sqlite_db attribute is specified, attempt
    to use the database instead of querying the json files.

    Returns: { filename: [ labels... ], ... }
    """
    if labels is not None:
      labels = toiter(labels)
    
    if self.sqlite_db:
      return self.file_locations_per_label_sql(labels)
    return self.file_locations_per_label_json(labels, allow_missing)
  
  def file_locations_per_label_json(self, labels, allow_missing=False):
    locations = defaultdict(list)
    parser = simdjson.Parser()
    if labels is not None:
      labels = set(toiter(labels))

    for index_files in self.fetch_all_index_files():
      for filename, content in index_files.items():
        index_labels = set(parser.parse(content).keys())
        filename = os.path.basename(filename)

        if labels is None:
          for label in index_labels:
            locations[int(label)].append(filename)
        elif len(labels) > len(index_labels):
          for label in index_labels:
            if int(label) in labels:
              locations[int(label)].append(filename)
        else:
          for label in labels:
            if str(label) in index_labels:
              locations[int(label)].append(filename)

    return locations

  def file_locations_per_label_sql(self, labels, sqlite_db=None):
    sqlite_db = nvl(sqlite_db, self.sqlite_db)
    if sqlite_db is None:
      raise ValueError("An sqlite database file must be specified.")

    locations = defaultdict(list)
    conn = sqlite3.connect(sqlite_db)
    cur = conn.cursor()

    where_clause = ""
    if labels:
      where_clause = "and file_lookup.label in ({})".format(
        ",".join(( str(int(lbl)) for lbl in labels ))
      )

    cur.execute("""
      select file_lookup.label, index_files.filename  
      from file_lookup, index_files
      where file_lookup.fid = index_files.id
        {}
    """.format(where_clause))
    while True:
      rows = cur.fetchmany(2**20)
      if len(rows) == 0:
        break
      for label, filename in rows:
        locations[int(label)].append(filename)
    conn.close()
    return locations      

  def query(self, bbox, allow_missing=False):
    """
    For the specified bounding box (or equivalent representation),
    list all segment ids enclosed within it.

    If allow_missing is set, then don't raise an error if an index
    file is missing.

    Returns: set(labels)
    """
    bbox = Bbox.create(bbox, context=self.physical_bounds, autocrop=True)
    original_bbox = bbox.clone()
    bbox = bbox.expand_to_chunk_size(self.chunk_size.astype(self.physical_bounds.dtype), offset=self.physical_bounds.minpt)

    if bbox.subvoxel():
      return []

    labels = set()
    fast_path = bbox.contains_bbox(self.physical_bounds)

    if self.sqlite_db and fast_path:
      conn = sqlite3.connect(self.sqlite_db)
      cur = conn.cursor()
      cur.execute("select label from file_lookup")
      while True:
        rows = cur.fetchmany(size=2**20)
        if len(rows) == 0:
          break
        labels.update(( int(row[0]) for row in rows ))
      conn.close()
      return labels

    index_files = self.index_file_paths_for_bbox(bbox)
    results = self.fetch_index_files(index_files)

    parser = simdjson.Parser()
    for filename, content in tqdm(results.items(), desc="Decoding Labels", disable=(not self.config.progress)):
      if content is None:
        if allow_missing:
          continue
        else:
          raise SpatialIndexGapError(filename + " was not found.")

      # The bbox test saps performance a lot
      # but we can skip it if we know 100% that
      # the labels are going to be inside. This
      # optimization is important for querying 
      # entire datasets, which is contemplated
      # for shard generation.
      if fast_path:
        res = parser.parse(content).keys()
        labels.update( (int(label) for label in res ) ) # fast path: 16% CPU
      else:
        res = simdjson.loads(content)
        for label, label_bbx in res.items():
          label = int(label)
          label_bbx = Bbox.from_list(label_bbx)

          if Bbox.intersects(label_bbx, original_bbox):
            labels.add(label)

    return labels

class CachedSpatialIndex(SpatialIndex):
  def __init__(
    self, cache, config, 
    cloudpath, bounds, chunk_size,
    resolution
  ):
    self.cache = cache
    self.subdir = os.path.relpath(cloudpath, cache.meta.cloudpath)

    super(CachedSpatialIndex, self).__init__(
      cloudpath, bounds, chunk_size, config, 
      resolution=resolution
    )

  def fetch_index_files(self, index_files, progress=None):
    progress = nvl(progress, self.config.progress)
    index_files = [ self.cache.meta.join(self.subdir, fname) for fname in index_files ]
    return self.cache.download(index_files, progress=progress)
