from collections import defaultdict
import itertools
import re
import urllib.parse
import os 
import queue
import sqlite3
import threading
import time

import tenacity
import numpy as np
import simdjson
from tqdm import tqdm

from cloudfiles import CloudFiles

from ...secrets import mysql_credentials
from ...exceptions import SpatialIndexGapError
from ... import paths
from ...lib import (
  Bbox, Vec, xyzrange, min2, 
  toiter, sip, nvl, getprecision
)

retry = tenacity.retry(
  reraise=True, 
  stop=tenacity.stop_after_attempt(7), 
  wait=tenacity.wait_random_exponential(0.5, 60.0),
)

def tostr(x):
  if isinstance(x, bytearray):
    return bytes(x).decode("utf8")
  elif isinstance(x, bytes):
    return x.decode("utf8")
  else:
    return x

def parse_db_path(path):
  """
  sqlite paths: filename.db
  mysql paths: mysql://{user}:{pwd}@{host}/{database}

  database defaults to "spatial_index"
  """
  result = urllib.parse.urlparse(path)
  scheme = result.scheme or "sqlite"

  if scheme == "sqlite":
    path = path.replace("sqlite://", "")
    return {
      "scheme": scheme,
      "username": None,
      "password": None,
      "hostname": None,
      "port": None,
      "path": path,
    }

  path = "spatial_index"
  if result.path:
    path = result.path.replace('/', '')

  return {
    "scheme": scheme,
    "username": result.username,
    "password": result.password,
    "hostname": result.hostname,
    "port": result.port,
    "path": path,
  }

def connect(path, use_database=True):
  result = parse_db_path(path)

  if result["scheme"] == "sqlite":
    return sqlite3.connect(result["path"])

  if result["scheme"] != "mysql":
    raise ValueError(
      f"{result['scheme']} is not a supported "
      f"spatial database connector."
    )

  if any([ result[x] is None for x in ("username", "password") ]):
    credentials = mysql_credentials(result["hostname"])
    if result["password"] is None:
      result["password"] = credentials["password"]
    if result["username"] is None:
      result["username"] = credentials["username"]

  import mysql.connector
  return mysql.connector.connect(
    host=result["hostname"],
    user=result["username"],
    passwd=result["password"],
    port=(result["port"] or 3306), # default MySQL port
    database=(result["path"] if use_database else None),
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
    config=None, sql_db=None, resolution=None
  ):
    self.cloudpath = cloudpath
    self.path = paths.extract(cloudpath)
    self.bounds = Bbox.create(bounds)
    self.chunk_size = Vec(*chunk_size)
    self.sql_db = config.spatial_index_db # optional DB for higher performance
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
    all_index_paths = self.index_file_paths_for_bbox(self.physical_bounds)
    
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
    """
    Returns an iterator over the spatial index filenames
    that overlap with the bounding box.
    """
    bbox = bbox.expand_to_chunk_size(self.chunk_size, offset=self.physical_bounds.minpt)
    if bbox.subvoxel():
      return []

    chunk_size = self.chunk_size
    bounds = self.physical_bounds.clone()
    precision = self.precision

    class IndexPathIterator():
      def __len__(self):
        return bbox.num_chunks(chunk_size)
      def __iter__(self):
        for pt in xyzrange(bbox.minpt, bbox.maxpt, chunk_size):
          search = Bbox( pt, min2(pt + chunk_size, bounds.maxpt) )
          yield search.to_filename(precision) + '.spatial'

    return IndexPathIterator()

  def _to_sql_common(
    self, conn, cur, path,
    create_indices, allow_missing, 
    progress, mysql_syntax=False, parallel=1
  ):
    # handle SQLite vs MySQL syntax quirks
    BIND = '%s' if mysql_syntax else '?'
    AUTOINC = "AUTO_INCREMENT" if mysql_syntax else "AUTOINCREMENT"
    INTEGER = "BIGINT UNSIGNED" if mysql_syntax else "INTEGER"

    progress = nvl(progress, self.config.progress)
    if parallel < 1 or parallel != int(parallel):
      raise ValueError(f"parallel must be an integer >= 1. Got: {parallel}")

    cur.execute("""DROP TABLE IF EXISTS index_files""")
    cur.execute("""DROP TABLE IF EXISTS file_lookup""")

    cur.execute(f"""
      CREATE TABLE index_files (
        id {INTEGER} PRIMARY KEY {AUTOINC},
        filename VARCHAR(100) NOT NULL
      )
    """)
    cur.execute("CREATE INDEX idxfname ON index_files (filename)")

    cur.execute(f"""
      CREATE TABLE file_lookup (
        label {INTEGER} NOT NULL,
        fid {INTEGER} NOT NULL REFERENCES index_files(id),
        PRIMARY KEY(label,fid)
      )
    """)

    finished_loading_evt = threading.Event()
    query_lock = threading.Lock()

    qu = queue.Queue(maxsize=(2 * parallel + 1))
    threads = [ 
      threading.Thread(
        target=thread_safe_insert, 
        args=(path, query_lock, finished_loading_evt, qu, progress, mysql_syntax)
      )
      for i in range(parallel) 
    ]
    for t in threads:
      t.start()

    for index_files in self.fetch_all_index_files(progress=progress, allow_missing=allow_missing):
      qu.put(index_files)

    finished_loading_evt.set()
    qu.join()

    if create_indices:
      if progress:
        print("Creating labels index...")
      cur.execute("CREATE INDEX file_lbl ON file_lookup (label)")

      if progress:
        print("Creating filename index...")
      cur.execute("CREATE INDEX fname ON file_lookup (fid)")

  def to_sql(
    self, path=None, create_indices=True, 
    allow_missing=False,
    progress=None, parallel=1
  ):
    path = path or self.sql_db
    parse = parse_db_path(path)
    if parse["scheme"] == "sqlite":
      if parallel != 1:
        raise ValueError("sqlite supports only one writer at a time.")
      return self.to_sqlite(parse["path"], create_indices, progress, allow_missing)
    elif parse["scheme"] == "mysql":
      return self.to_mysql(path, create_indices, allow_missing, progress, parallel)
    else:
      raise ValueError(
        f"Unsupported database type. {path}\n"
        "Supported types: sqlite:// and mysql://"
      )

  def to_mysql(
    self, path, 
    create_indices=True, allow_missing=False, 
    progress=None, parallel=1
  ):
    """
    Create a mysql database of labels and filenames
    from the JSON spatial_index for faster performance.
    """
    progress = nvl(progress, self.config.progress)
    parse = parse_db_path(path)
    conn = connect(path, use_database=False) # in case no DB exists yet
    cur = conn.cursor()

    database_name = parse["path"] or "spatial_index"
    # GRANT CREATE, SELECT, INSERT, DELETE, INDEX ON database TO user@localhost

    # Can't sanitize table names easily using a bind.
    # Therefore check to see if there are any non alphanumerics + underscore
    # https://stackoverflow.com/questions/3247183/variable-table-name-in-sqlite
    if re.search(r'[^a-zA-Z0-9_]', database_name):
      raise ValueError(
        f"Invalid characters in database name. "
        f"Only alphanumerics and underscores are allowed. "
        f"Got: {database_name}"
      )

    cur.execute(f"""
      CREATE DATABASE IF NOT EXISTS {database_name}
      CHARACTER SET utf8 COLLATE utf8_bin
    """)
    cur.execute(f"use {database_name}")

    self._to_sql_common(
      conn, cur, path, 
      create_indices, allow_missing, progress, 
      mysql_syntax=True, parallel=parallel,
    )
    
    cur.close()
    conn.close()

  def to_sqlite(
    self, path="spatial_index.db", 
    create_indices=True, allow_missing=False,
    progress=None
  ):
    """
    Create a sqlite database of labels and filenames
    from the JSON spatial_index for faster performance.

    Depending on the dataset size, this could take a while.
    With a dataset with ~140k index files, the DB took over
    an hour to build and was 42 GB.
    """
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    set_journaling_to_performance_mode(cur, mysql_syntax=False)
    self._to_sql_common(
      conn, cur, path, 
      create_indices, allow_missing, progress, 
      mysql_syntax=False
    )
    cur.execute("PRAGMA journal_mode = DELETE")
    cur.execute("PRAGMA synchronous = FULL")
    cur.close()
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

    if self.sql_db:
      conn = connect(self.sql_db)
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

    If the spatial_index.sql_db attribute is specified, attempt
    to use the database instead of querying the json files.

    Returns: { filename: [ labels... ], ... }
    """
    if labels is not None:
      labels = toiter(labels)
    
    if self.sql_db:
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

  def file_locations_per_label_sql(self, labels, sql_db=None):
    sql_db = nvl(sql_db, self.sql_db)
    if sql_db is None:
      raise ValueError("An sqlite database file must be specified.")

    locations = defaultdict(list)
    conn = connect(sql_db)
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
        locations[int(label)].append(tostr(filename))
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

    if self.sql_db and fast_path:
      conn = connect(self.sql_db)
      cur = conn.cursor()
      cur.execute("select distinct label from file_lookup")
      while True:
        rows = cur.fetchmany(size=2**20)
        if len(rows) == 0:
          break
        # Sqlite only stores signed integers, so we need to coerce negative
        # integers back into unsigned with a bitwise and.
        labels.update(( int(row[0]) & 0xffffffffffffffff for row in rows ))
      cur.close()
      conn.close()
      return labels

    index_files = self.index_file_paths_for_bbox(bbox)

    num_blocks = int(np.ceil(len(index_files) / 10000))
    for index_files_subset in tqdm(sip(index_files, 10000), total=num_blocks, desc="Block", disable=((not self.config.progress) or (num_blocks == 1))):
      results = self.fetch_index_files(index_files_subset)

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

def thread_safe_insert(path, lock, evt, qu, progress, mysql_syntax):
  conn = connect(path)
  cur = conn.cursor()

  set_journaling_to_performance_mode(cur, mysql_syntax)

  print("started thread", threading.current_thread().native_id)
  try:
    while not evt.is_set() or not qu.empty():
      try:
        index_files = qu.get(block=False)
      except queue.Empty:
        time.sleep(0.1)
        continue
      insert_index_files(index_files, lock, conn, cur, progress, mysql_syntax)
      qu.task_done()
  finally:
    cur.close()
    conn.close()

  print('finished', threading.current_thread().native_id)

def insert_index_files(index_files, lock, conn, cur, progress, mysql_syntax):
  # handle SQLite vs MySQL syntax quirks
  BIND = '%s' if mysql_syntax else '?'
  AUTOINC = "AUTO_INCREMENT" if mysql_syntax else "AUTOINCREMENT"
  INTEGER = "BIGINT UNSIGNED" if mysql_syntax else "INTEGER"

  values = ( (os.path.basename(filename),) for filename in index_files.keys() )
  if mysql_syntax:
    values = list(values) # doesn't support generators in v8.0.26

  # This is a critical region as if multiple inserts and queries occur outside
  # a transaction, you could crash the following code. Fortunately, these 
  # take much less time than the final insert, so shouldn't be a problem
  # if a thread blocks.
  # The select could be rewritten to be order independent at the cost of 
  # sending more data for a (probably) slower query.
  with lock:
    cur.executemany(f"INSERT INTO index_files(filename) VALUES ({BIND})", values)
    cur.execute(f"SELECT filename,id from index_files ORDER BY id desc LIMIT {len(index_files)}")

  filename_id_map = { 
    tostr(fname): int(row_id) 
    for fname, row_id in cur.fetchall() 
  }

  parser = simdjson.Parser()

  all_values = []
  for filename, content in index_files.items():
    if content is None:
      continue
    index_labels = parser.parse(content).keys()
    filename = os.path.basename(filename)
    fid = filename_id_map[filename]
    all_values.extend(( (int(label), fid) for label in index_labels ))
  
  @retry
  def insert_file_lookup_values(cur, chunked_values):
    cur.executemany(f"INSERT INTO file_lookup(label, fid) VALUES ({BIND},{BIND})", chunked_values)

  for chunked_values in sip(all_values, 500000):
    insert_file_lookup_values(cur, chunked_values)
    conn.commit()

def set_journaling_to_performance_mode(cur, mysql_syntax):
  if mysql_syntax:
    cur.execute("SET SESSION TRANSACTION ISOLATION LEVEL READ UNCOMMITTED")
  else: # sqlite
    cur.execute("PRAGMA journal_mode = MEMORY")
    cur.execute("PRAGMA synchronous = OFF")
  return cur

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
