from collections import defaultdict
import itertools
import re
import urllib.parse
import os 
import io
import math
import struct
import queue
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
try:
  from enum import StrEnum
except ImportError:
  from enum import Enum
  class StrEnum(str, Enum):
    pass

import tenacity
import numpy as np
from tqdm import tqdm

from cloudfiles import CloudFiles

from ...secrets import mysql_credentials, psql_credentials
from ...exceptions import SpatialIndexGapError
from ... import paths
from ...lib import (
  Bbox, Vec, xyzrange, min2, 
  toiter, sip, nvl, getprecision
)

class DbType(StrEnum):
  SQLITE = "sqlite"
  MYSQL = "mysql"
  POSTGRES = "postgres"

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

# 11-byte signature for PostgreSQL binary COPY format.
# See: https://www.postgresql.org/docs/current/sql-copy.html#SQL-COPY-FILE-FORMATS
PG_BINARY_COPY_SIGNATURE = b'PGCOPY\n\377\r\n\0'

# SQL template for parallel range-partitioned distinct label queries.
# Each thread fills in {low}/{high} to scan a non-overlapping slice
# of the PK B-tree index on file_lookup(label, fid).
# Hash aggregation handles dedup within each range.
PG_RANGE_DISTINCT_SQL = """
  COPY (
    SELECT DISTINCT label FROM file_lookup
    WHERE label >= {low} AND label < {high}
    ORDER BY label
  ) TO STDOUT WITH BINARY
"""

def _parse_pg_binary_copy_bigint(data):
  """Parse PostgreSQL binary COPY output for a single BIGINT column
  into a numpy uint64 array.

  PG binary COPY format:
    Header: 11-byte signature + 4-byte flags + 4-byte ext_len + ext_data
    Per row: 2 (field count=1) + 4 (byte length=8) + 8 (big-endian int64) = 14 bytes
    Trailer: 2 bytes (-1 as int16)

  See: https://www.postgresql.org/docs/current/sql-copy.html#SQL-COPY-FILE-FORMATS
  """
  if len(data) < 19:
    return np.array([], dtype=np.uint64)

  mv = memoryview(data)
  ext_len = int(np.frombuffer(mv[15:19], dtype='>u4')[0])
  header_size = 19 + ext_len
  body = mv[header_size:-2]  # strip trailer (zero-copy slice)
  if len(body) == 0:
    return np.array([], dtype=np.uint64)

  ROW_SIZE = 14  # 2 + 4 + 8 bytes per row
  n_rows = len(body) // ROW_SIZE
  row_dt = np.dtype([('nfields', '>i2'), ('length', '>i4'), ('value', '>i8')])
  rows = np.frombuffer(body, dtype=row_dt, count=n_rows)
  return rows['value'].astype(np.uint64)

def _build_pg_binary_copy_two_bigints(col1, col2):
  """Build a PostgreSQL binary COPY buffer for two BIGINT columns.

  Accepts two array-like inputs (numpy arrays or lists) of equal length.
  Returns a BytesIO positioned at the start, ready for copy_expert().

  PG binary COPY format:
    Header: 11-byte signature + 4-byte flags + 4-byte ext_len
    Per row: 2 (nfields=2) + 4 (len=8) + 8 (val1) + 4 (len=8) + 8 (val2) = 26 bytes
    Trailer: 2 bytes (-1 as int16)
  """
  col1 = np.asarray(col1, dtype=np.int64)
  col2 = np.asarray(col2, dtype=np.int64)
  n = len(col1)

  if n == 0:
    buf = io.BytesIO()
    buf.write(PG_BINARY_COPY_SIGNATURE)
    buf.write(struct.pack('>ii', 0, 0))
    buf.write(struct.pack('>h', -1))
    buf.seek(0)
    return buf

  row_dt = np.dtype([
    ('nfields', '>i2'),
    ('len1', '>i4'), ('val1', '>i8'),
    ('len2', '>i4'), ('val2', '>i8'),
  ])
  rows = np.empty(n, dtype=row_dt)
  rows['nfields'] = 2
  rows['len1'] = 8
  rows['val1'] = col1
  rows['len2'] = 8
  rows['val2'] = col2

  buf = io.BytesIO()
  buf.write(PG_BINARY_COPY_SIGNATURE)
  buf.write(struct.pack('>ii', 0, 0))    # flags + ext_len
  buf.write(rows.tobytes())
  buf.write(struct.pack('>h', -1))       # trailer
  buf.seek(0)
  return buf

def _pg_parallel_distinct_labels(db_path, n_threads=8):
  """Query distinct labels from file_lookup using parallel range scans.

  Splits the label keyspace into n_threads non-overlapping ranges,
  queries each on a separate Postgres connection (= separate backend
  process = separate CPU core), then concatenates the sorted results.
  """
  conn = connect(db_path)
  cur = conn.cursor()
  cur.execute("SELECT MIN(label), MAX(label) FROM file_lookup")
  row = cur.fetchone()
  cur.close()
  conn.close()

  if row is None or row[0] is None:
    return np.array([], dtype=np.uint64)

  min_label, max_label = int(row[0]), int(row[1])

  if min_label == max_label:
    return np.array([min_label], dtype=np.uint64)

  # Split into non-overlapping [low, high) ranges.
  boundaries = np.linspace(min_label, max_label + 1, n_threads + 1, dtype=np.int64)
  # Deduplicate in case range is smaller than n_threads
  boundaries = np.unique(boundaries)
  actual_threads = len(boundaries) - 1

  def _worker(low, high):
    wconn = connect(db_path)
    wcur = wconn.cursor()
    wcur.execute("SET work_mem = '256MB'")
    buf = io.BytesIO()
    sql = PG_RANGE_DISTINCT_SQL.format(low=int(low), high=int(high))
    wcur.copy_expert(sql, buf)
    wcur.close()
    wconn.close()
    return _parse_pg_binary_copy_bigint(buf.getvalue())

  with ThreadPoolExecutor(max_workers=actual_threads) as executor:
    futures = [
      executor.submit(_worker, boundaries[i], boundaries[i + 1])
      for i in range(actual_threads)
    ]
    arrays = [f.result() for f in futures]

  non_empty = [a for a in arrays if len(a) > 0]
  if not non_empty:
    return np.array([], dtype=np.uint64)

  return np.concatenate(non_empty)

def parse_db_path(path):
  """
  sqlite paths: filename.db
  mysql paths: mysql://{user}:{pwd}@{host}/{database}
  postgres paths: postgres://{user}:{pwd}@{host}/{database}

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
  elif result["scheme"] == "mysql":
    if any([ result[x] is None for x in ("username", "password") ]):
      credentials = mysql_credentials(result["hostname"])
      if result["password"] is None:
        result["password"] = credentials["password"]
      if result["username"] is None:
        result["username"] = credentials["username"]

    try:
      import mysql.connector
    except ImportError:
      raise ImportError("""`mysql-connector-python` is not installed. Please install it via `pip install cloud-volume[mysql]`""")

    return mysql.connector.connect(
      host=result["hostname"],
      user=result["username"],
      passwd=result["password"],
      port=(result["port"] or 3306), # default MySQL port
      database=(result["path"] if use_database else None),
    )
  elif result["scheme"] in ("postgres", "postgresql"):
    if any([ result[x] is None for x in ("username", "password") ]):
      credentials = psql_credentials(result["hostname"])
      if result["password"] is None:
        result["password"] = credentials["password"]
      if result["username"] is None:
        result["username"] = credentials["username"]

    try:
      import psycopg2
    except ImportError:
      raise ImportError("""`psycopg2` is not installed. Please install it via `pip install cloud-volume[psql]`""")
    kwargs = {
      "host": result["hostname"],
      "user": result["username"],
      "password": result["password"],
      "port": (result["port"] or 5432),
    }
    if use_database:
      kwargs["database"] = result["path"]

    # psycopg2 doesn't like None values for keyword arguments
    kwargs = { k: v for k, v in kwargs.items() if v is not None }

    return psycopg2.connect(**kwargs)
  else:
    raise ValueError(
      f"{result['scheme']} is not a supported "
      f"spatial database connector. Supported: sqlite, mysql, postgres"
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

  def fetch_all_index_files(self, allow_missing=False, progress=None, parallel=1):
    """Generator returning batches of (filename, json)

    The parallel parameter here affects the chunking of files fetched
    from the cloud, not the direct activation of threads or processes.
    This is a bit exceptional as 'parallel' usually implies direct
    threading/multiprocessing at the leaf level of the call tree.
    """

    all_index_paths = self.index_file_paths_for_bbox(self.physical_bounds)
    total_files = len(all_index_paths)

    progress = nvl(progress, self.config.progress)

    # Heuristic for chunk size
    num_chunks = 4 * parallel
    if num_chunks == 0:
        num_chunks = 1

    N = int(math.ceil(total_files / num_chunks))
    N = max(10, min(N, 500)) # Clamp to a reasonable range

    pbar = tqdm(
      total=total_files,
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

      pbar.update(len(index_paths))
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
    progress, db_type, parallel=1
  ):
    # handle SQLite vs MySQL syntax quirks
    if db_type == DbType.MYSQL:
      BIND = '%s'
      AUTOINC = "AUTO_INCREMENT"
      INTEGER = "BIGINT UNSIGNED"
      ID_INTEGER = INTEGER
      CREATE_TABLE = "CREATE TABLE"
    elif db_type == DbType.POSTGRES:
      BIND = '%s'
      AUTOINC = ""
      INTEGER = "BIGINT" # no unsigned
      ID_INTEGER = "BIGSERIAL"
      CREATE_TABLE = "CREATE UNLOGGED TABLE"
    else: # sqlite
      BIND = '?'
      AUTOINC = "AUTOINCREMENT"
      INTEGER = "INTEGER"
      ID_INTEGER = INTEGER
      CREATE_TABLE = "CREATE TABLE"

    progress = nvl(progress, self.config.progress)
    if parallel < 1 or parallel != int(parallel):
      raise ValueError(f"parallel must be an integer >= 1. Got: {parallel}")

    if db_type == DbType.POSTGRES:
      cur.execute("""DROP TABLE IF EXISTS file_lookup CASCADE""")
      cur.execute("""DROP TABLE IF EXISTS index_files CASCADE""")
    else:
      cur.execute("""DROP TABLE IF EXISTS file_lookup""")
      cur.execute("""DROP TABLE IF EXISTS index_files""")

    cur.execute(f"""
      {CREATE_TABLE} index_files (
        id {ID_INTEGER} PRIMARY KEY {AUTOINC},
        filename VARCHAR(100) NOT NULL
      )
    """)
    cur.execute("CREATE INDEX idxfname ON index_files (filename)")

    # Create file_lookup without PK/FK constraints during bulk load.
    # Constraints are added post-load via sort+bulk-build which is
    # ~10x faster than incremental B-tree maintenance per row.
    cur.execute(f"""
      {CREATE_TABLE} file_lookup (
        label {INTEGER} NOT NULL,
        fid {INTEGER} NOT NULL
      )
    """)

    conn.commit()

    finished_loading_evt = threading.Event()
    query_lock = threading.Lock()

    qu = queue.Queue(maxsize=(2 * parallel + 1))
    threads = [ 
      threading.Thread(
        target=thread_safe_insert, 
        args=(path, query_lock, finished_loading_evt, qu, progress, db_type)
      )
      for i in range(parallel) 
    ]
    for t in threads:
      t.start()

    for index_files in self.fetch_all_index_files(progress=progress, allow_missing=allow_missing, parallel=parallel):
      qu.put(index_files)

    finished_loading_evt.set()
    qu.join()

    if create_indices:
      # Now that all data is loaded, build PK and FK via sort+bulk-load.
      # This is ~10x faster than maintaining the B-tree incrementally
      # during insertion.
      if db_type in (DbType.POSTGRES, DbType.MYSQL):
        if progress:
          print("Building primary key (label, fid)...")
        cur.execute("ALTER TABLE file_lookup ADD PRIMARY KEY (label, fid)")
        conn.commit()
        if progress:
          print("Adding foreign key constraint...")
        cur.execute(
          "ALTER TABLE file_lookup "
          "ADD CONSTRAINT file_lookup_fid_fkey "
          "FOREIGN KEY (fid) REFERENCES index_files(id)"
        )
        conn.commit()
      elif db_type == DbType.SQLITE:
        # SQLite doesn't support ALTER TABLE ADD PRIMARY KEY or FOREIGN KEY.
        # Use a unique index to enforce the same constraint.
        if progress:
          print("Building unique index (label, fid)...")
        cur.execute("CREATE UNIQUE INDEX pk_label_fid ON file_lookup (label, fid)")
        conn.commit()

      if db_type != DbType.POSTGRES:
        # For Postgres, the PK (label, fid) already covers label-only lookups.
        # The separate index is redundant and wastes disk/RAM.
        if progress:
          print("Creating labels index...")
        cur.execute("CREATE INDEX file_lbl ON file_lookup (label)")

      if progress:
        print("Creating filename index...")
      cur.execute("CREATE INDEX fname ON file_lookup (fid)")

      if progress:
        print("Running ANALYZE...")
      if db_type == DbType.MYSQL:
        cur.execute("ANALYZE TABLE file_lookup")
      else:
        # Works for both Postgres and SQLite
        cur.execute("ANALYZE file_lookup")
      conn.commit()

  def to_sql(
    self, path=None, create_indices=True, 
    allow_missing=False,
    progress=None, parallel=1
  ):
    path = path or self.sql_db
    parse = parse_db_path(path)
    scheme = parse["scheme"]

    if scheme == DbType.SQLITE:
      if parallel != 1:
        raise ValueError("sqlite supports only one writer at a time.")
      return self.to_sqlite(
        parse["path"], 
        create_indices=create_indices, 
        allow_missing=allow_missing, 
        progress=progress,
      )
    elif scheme == DbType.MYSQL:
      return self.to_mysql(path, 
        create_indices=create_indices, 
        allow_missing=allow_missing, 
        progress=progress,
        parallel=parallel,
      )
    elif scheme in (DbType.POSTGRES, "postgresql"):
      return self.to_postgres(
        path,
        create_indices=create_indices,
        allow_missing=allow_missing,
        progress=progress,
        parallel=parallel,
      )
    else:
      raise ValueError(
        f"Unsupported database type. {path}\n"
        "Supported types: sqlite://, mysql://, and postgres://"
      )

  def to_postgres(
    self, path,
    create_indices=True, allow_missing=False,
    progress=None, parallel=1
  ):
    """
    Create a postgres database of labels and filenames
    from the JSON spatial_index for faster performance.
    """
    try:
      import psycopg2
      import psycopg2.extensions
    except ImportError:
      raise ImportError("""`psycopg2` is not installed. Please install it via `pip install cloud-volume[psql]`""")

    progress = nvl(progress, self.config.progress)
    parse = parse_db_path(path)

    database_name = parse["path"] or "spatial_index"

    if re.search(r'[^a-zA-Z0-9_]', database_name):
      raise ValueError(
        f"Invalid characters in database name. "
        f"Only alphanumerics and underscores are allowed. "
        f"Got: {database_name}"
      )

    try:
      # Connect to default 'postgres' db to create new db
      # Use urllib.parse for robust URI manipulation
      original_uri_parts = urllib.parse.urlparse(path)
      new_uri_parts = original_uri_parts._replace(path='/postgres')
      postgres_db_path_uri = urllib.parse.urlunparse(new_uri_parts)

      conn = connect(postgres_db_path_uri, use_database=True)
      conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
      cur = conn.cursor()
    except psycopg2.Error as e:
      raise ConnectionError(
        f"Failed to connect to default 'postgres' database to create '{database_name}'. "
        f"Check that the server is running and that you have permissions to connect. "
        f"Original error: {e}"
      ) from e

    try:
      cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database_name,))
      exists = cur.fetchone()

      if not exists:
          try:
            cur.execute(f'CREATE DATABASE "{database_name}"')
          except psycopg2.Error as e:
            raise PermissionError(
              f"Failed to create database '{database_name}'. "
              f"The user in the connection string must have CREATEDB privileges. "
              f"Original error: {e}"
            ) from e
    finally:
      cur.close()
      conn.close()

    # now connect to the newly created database
    conn = connect(path, use_database=True)
    cur = conn.cursor()

    self._to_sql_common(
      conn, cur, path,
      create_indices, allow_missing, progress,
      db_type=DbType.POSTGRES, parallel=parallel,
    )

    cur.close()
    conn.close()

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
      db_type=DbType.MYSQL, parallel=parallel,
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
    set_journaling_to_performance_mode(cur, db_type=DbType.SQLITE)
    self._to_sql_common(
      conn, cur, path,
      create_indices, allow_missing, progress,
      db_type=DbType.SQLITE
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
    import simdjson
    locations = defaultdict(list)
    parser = simdjson.Parser()

    label = str(label)
    bbox = None

    if self.sql_db:
      conn = connect(self.sql_db)
      cur = conn.cursor()

      db_type_str = parse_db_path(self.sql_db)["scheme"]
      db_type = DbType.SQLITE
      if db_type_str == DbType.MYSQL:
        db_type = DbType.MYSQL
      elif db_type_str in (DbType.POSTGRES, "postgresql"):
        db_type = DbType.POSTGRES

      BIND = '?'
      if db_type in (DbType.MYSQL, DbType.POSTGRES):
        BIND = '%s'

      cur.execute(f"""
        select index_files.filename  
        from file_lookup, index_files
        where file_lookup.fid = index_files.id
          and file_lookup.label = {BIND}
      """, (label,))
      iterator = [ self.fetch_index_files(( row[0] for row in cur.fetchall() )) ]
      conn.close()
    else:
      iterator = self.fetch_all_index_files()

    for index_files in iterator:
      for filename, content in index_files.items():

        # Need to delete segid_bbox_dict to avoid this error:
        # RuntimeError: Tried to re-use a parser while simdjson.Object 
        #   and/or simdjson.Array objects still exist referencing the 
        #   old parser.
        segid_bbox_dict = parser.parse(content)
        filename = os.path.basename(filename)

        if label not in segid_bbox_dict:
          del segid_bbox_dict
          continue 

        current_bbox = Bbox.from_list(
          np.frombuffer(segid_bbox_dict[label].as_buffer(of_type="i"), dtype=np.int64)
        )

        if bbox is None:
          bbox = current_bbox
        else:
          bbox = Bbox.expand(bbox, current_bbox)

        del segid_bbox_dict

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
    import simdjson
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

  def query(self, bbox, allow_missing=False, nthread=1):
    """
    For the specified bounding box (or equivalent representation),
    list all segment ids enclosed within it.

    If allow_missing is set, then don't raise an error if an index
    file is missing.

    nthread: number of threads to use for the Postgres fast path
    (when the query covers the full dataset). Has no effect for
    non-Postgres databases or non-fast-path queries.

    Returns: iterable
    """
    import simdjson
    bbox = Bbox.create(bbox, context=self.physical_bounds, autocrop=True)
    original_bbox = bbox.clone()
    bbox = bbox.expand_to_chunk_size(self.chunk_size.astype(self.physical_bounds.dtype), offset=self.physical_bounds.minpt)

    if bbox.subvoxel():
      return []

    fast_path = bbox.contains_bbox(self.physical_bounds)

    if nthread > 1 and not (self.sql_db and fast_path):
      print(
        "WARNING: nthread > 1 has no effect: requires a Postgres "
        "sql_db and a bounding box that covers the full dataset (fast path)."
      )

    if self.sql_db and fast_path:
      conn = connect(self.sql_db)
      db_type = parse_db_path(self.sql_db)["scheme"]

      if db_type in ("postgres", "postgresql"):
        # Parallel range-partitioned distinct label query.
        # Splits the label space into ranges, each queried on a
        # separate connection (separate PG backend = separate core).
        conn.close()
        return _pg_parallel_distinct_labels(self.sql_db, n_threads=nthread)
      else:
        if nthread > 1:
          print(
            "WARNING: nthread > 1 has no effect for non-Postgres databases."
          )
        cur = conn.cursor()
        cur.execute("select distinct label from file_lookup")

        labels_list = []

        while True:
          rows = cur.fetchmany(size=2**24)
          if len(rows) == 0:
            break
          # Sqlite only stores signed integers, so we need to coerce negative
          # integers back into unsigned.
          labels_list.append(np.fromiter((row[0] for row in rows), dtype=np.uint64, count=len(rows)))

        cur.close()
        conn.close()

        labels = np.concatenate(labels_list)
        del labels_list

        labels.sort()
        return labels

    labels = set()
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

def thread_safe_insert(path, lock, evt, qu, progress, db_type):
  conn = connect(path)
  cur = conn.cursor()

  set_journaling_to_performance_mode(cur, db_type)

  print("started thread", threading.current_thread().ident)
  try:
    while not evt.is_set() or not qu.empty():
      try:
        index_files = qu.get(block=False)
      except queue.Empty:
        time.sleep(0.1)
        continue
      insert_index_files(index_files, lock, conn, cur, progress, db_type)
      qu.task_done()
  finally:
    cur.close()
    conn.close()

  print('finished', threading.current_thread().ident)

def insert_index_files(index_files, lock, conn, cur, progress, db_type):
  import simdjson
  # handle SQLite vs MySQL syntax quirks
  if db_type in (DbType.MYSQL, DbType.POSTGRES):
    BIND = '%s'
  else:
    BIND = '?'

  @retry
  def insert_file_lookup_values(cur, chunked_values):
    nonlocal BIND
    bindlist = ",".join([f"({BIND},{BIND})"] * len(chunked_values))
    flattened_values = []
    for label, fid in chunked_values:
      flattened_values.append(label)
      flattened_values.append(fid)
    cur.execute(f"INSERT INTO file_lookup(label, fid) VALUES {bindlist}", flattened_values)

  values = [ os.path.basename(filename) for filename in index_files.keys() ]

  # This is a critical region as if multiple inserts and queries occur outside
  # a transaction, you could crash the following code. Fortunately, these 
  # take much less time than the final insert, so shouldn't be a problem
  # if a thread blocks.
  # The select could be rewritten to be order independent at the cost of 
  # sending more data for a (probably) slower query.
  with lock:
    bindlist = ",".join([f"({BIND})"] * len(values))
    cur.execute(f"INSERT INTO index_files(filename) VALUES {bindlist}", values)
    cur.execute(f"SELECT filename,id from index_files ORDER BY id desc LIMIT {len(index_files)}")

  filename_id_map = { 
    tostr(fname): int(row_id) 
    for fname, row_id in cur.fetchall() 
  }

  parser = simdjson.Parser()

  if db_type == DbType.POSTGRES:
    # Vectorized path: build flat numpy arrays, then binary COPY.
    # Avoids millions of Python tuple allocations and StringIO writes.
    all_labels = []
    all_fids = []
    for filename, content in index_files.items():
      if content is None:
        continue
      index_labels = parser.parse(content).keys()
      filename = os.path.basename(filename)
      fid = filename_id_map[filename]
      chunk_labels = np.array([int(label) for label in index_labels], dtype=np.int64)
      all_labels.append(chunk_labels)
      all_fids.append(np.full(len(chunk_labels), fid, dtype=np.int64))

    if not all_labels:
      return

    labels_arr = np.concatenate(all_labels)
    fids_arr = np.concatenate(all_fids)
    total = len(labels_arr)

    del all_labels, all_fids

    pbar = tqdm(
      desc="Inserting File Lookups",
      disable=(not progress),
      total=total
    )

    block_size = 5000000
    conn.commit()
    with pbar:
      for start in range(0, total, block_size):
        end = min(start + block_size, total)
        buf = _build_pg_binary_copy_two_bigints(
          labels_arr[start:end], fids_arr[start:end]
        )
        cur.copy_expert(
          "COPY file_lookup(label, fid) FROM STDIN WITH BINARY", buf
        )
        conn.commit()
        pbar.update(end - start)
  else:
    all_values = []
    for filename, content in index_files.items():
      if content is None:
        continue
      index_labels = parser.parse(content).keys()
      filename = os.path.basename(filename)
      fid = filename_id_map[filename]
      all_values.extend(( (int(label), fid) for label in index_labels ))

    pbar = tqdm(
      desc="Inserting File Lookups",
      disable=(not progress),
      total=len(all_values)
    )

    if db_type == DbType.MYSQL:
      block_size = 500000
    else: # sqlite
      block_size = 15000

    with pbar:
      for chunked_values in sip(all_values, block_size):
        insert_file_lookup_values(cur, chunked_values)
        conn.commit()
        pbar.update(len(chunked_values))
def set_journaling_to_performance_mode(cur, db_type):
  if db_type in (DbType.MYSQL, DbType.POSTGRES):
    cur.execute("SET SESSION TRANSACTION ISOLATION LEVEL READ UNCOMMITTED")
    if db_type == DbType.POSTGRES:
        cur.execute("SET synchronous_commit=OFF")
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
