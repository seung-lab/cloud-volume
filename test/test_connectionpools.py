from tqdm import tqdm
import boto.s3.key

from cloudvolume.connectionpools import S3ConnectionPool, GCloudConnectionPool
from cloudvolume.threaded_queue import ThreadedQueue
from cloudvolume.storage import Storage

S3_POOL = S3ConnectionPool()
GC_POOL = GCloudConnectionPool()

def test_gc_stresstest():
  with Storage('gs://neuroglancer/removeme/connection_pool/', n_threads=0) as stor:
    stor.put_file('test', 'some string')

  n_trials = 500
  pbar = tqdm(total=n_trials)

  def create_conn(interface):
    conn = GC_POOL.get_connection()
    # assert GC_POOL.total_connections() <= GC_POOL.max_connections * 5
    bucket = conn.get_bucket('neuroglancer')
    blob = bucket.get_blob('removeme/connection_pool/test')
    blob.download_as_string()
    GC_POOL.release_connection(conn)
    pbar.update()

  with ThreadedQueue(n_threads=20) as tq:
    for _ in xrange(n_trials):
      tq.put(create_conn)

  pbar.close()

def test_s3_stresstest():
  with Storage('s3://neuroglancer/removeme/connection_pool/', n_threads=0) as stor:
    stor.put_file('test', 'some string')

  n_trials = 500
  pbar = tqdm(total=n_trials)

  def create_conn(interface):
    conn = S3_POOL.get_connection()  
    # assert S3_POOL.total_connections() <= S3_POOL.max_connections * 5
    bucket = conn.get_bucket('neuroglancer')
    k = boto.s3.key.Key(bucket)
    k.key = 'removeme/connection_pool/test'
    k.get_contents_as_string()
    S3_POOL.release_connection(conn)
    pbar.update()

  with ThreadedQueue(n_threads=20) as tq:
    for _ in xrange(n_trials):
      tq.put(create_conn)

  pbar.close()
