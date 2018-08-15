from __future__ import print_function
from six.moves import range

import tenacity
from tqdm import tqdm

from cloudvolume.connectionpools import S3ConnectionPool, GCloudBucketPool
from cloudvolume.threaded_queue import ThreadedQueue
from cloudvolume.storage import Storage

S3_POOL = S3ConnectionPool('s3', 'seunglab-test')
GC_POOL = GCloudBucketPool('seunglab-test')

retry = tenacity.retry(
    reraise=True, 
    stop=tenacity.stop_after_attempt(7), 
    wait=tenacity.wait_full_jitter(0.5, 60.0),
)

def test_gc_stresstest():
  with Storage('gs://seunglab-test/cloudvolume/connection_pool/', n_threads=0) as stor:
    stor.put_file('test', 'some string')

  n_trials = 500
  pbar = tqdm(total=n_trials)

  @retry
  def create_conn(interface):
    # assert GC_POOL.total_connections() <= GC_POOL.max_connections * 5
    bucket = GC_POOL.get_connection()
    blob = bucket.get_blob('cloudvolume/connection_pool/test')
    blob.download_as_string()
    GC_POOL.release_connection(bucket)
    pbar.update()

  with ThreadedQueue(n_threads=20) as tq:
    for _ in range(n_trials):
      tq.put(create_conn)

  pbar.close()

def test_s3_stresstest():
  with Storage('s3://seunglab-test/cloudvolume/connection_pool/', n_threads=0) as stor:
    stor.put_file('test', 'some string')

  n_trials = 500
  pbar = tqdm(total=n_trials)

  @retry
  def create_conn(interface):
    conn = S3_POOL.get_connection()  
    # assert S3_POOL.total_connections() <= S3_POOL.max_connections * 5
    bucket = conn.get_object(
      Bucket='seunglab-test',
      Key='cloudvolume/connection_pool/test',
    )
    S3_POOL.release_connection(conn)
    pbar.update()

  with ThreadedQueue(n_threads=20) as tq:
    for _ in range(n_trials):
      tq.put(create_conn)

  pbar.close()
