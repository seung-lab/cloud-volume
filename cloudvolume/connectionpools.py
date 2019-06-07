from six.moves import queue as Queue
import threading
import time
from functools import partial

import boto3 
from google.cloud.storage import Client
import tenacity

from .secrets import google_credentials, aws_credentials
from .exceptions import UnsupportedProtocolError

retry = tenacity.retry(
  reraise=True, 
  stop=tenacity.stop_after_attempt(7), 
  wait=tenacity.wait_random_exponential(0.5, 60.0),
)

class ConnectionPool(object):
  """
  This class is intended to be subclassed. See below.
  
  Creating fresh client or connection objects
  for Google or Amazon eventually starts causing
  breakdowns when too many connections open.
  
  To promote efficient resource use and prevent
  containers from dying, we create a ConnectionPool
  that allows for the reuse of connections.
  
  Storage interfaces may acquire and release connections 
  when they need or finish using them. 
  
  If the limit is reached, additional requests for
  acquiring connections will block until they can
  be serviced.
  """
  def __init__(self):
    self.pool = Queue.Queue(maxsize=0)
    self.outstanding = 0
    self._lock = threading.Lock()

  def total_connections(self):
    return self.pool.qsize() + self.outstanding

  def _create_connection(self):
    raise NotImplementedError

  def get_connection(self):    
    with self._lock:
      try:        
        conn = self.pool.get(block=False)
        self.pool.task_done()
      except Queue.Empty:
        conn = self._create_connection()
      finally:
        self.outstanding += 1

    return conn

  def release_connection(self, conn):
    if conn is None:
      return

    self.pool.put(conn)
    with self._lock:
      self.outstanding -= 1

  def close(self, conn):
    return 

  def reset_pool(self):
    while True:
      if not self.pool.qsize():
        break
      try:
        conn = self.pool.get()
        self.close(conn)
        self.pool.task_done()
      except Queue.Empty:
        break

    with self._lock:
      self.outstanding = 0

  def __del__(self):
    self.reset_pool()

class S3ConnectionPool(ConnectionPool):
  def __init__(self, service, bucket):
    self.service = service
    self.bucket = bucket
    self.credentials = aws_credentials(bucket, service)
    super(S3ConnectionPool, self).__init__()

  @retry
  def _create_connection(self):
    if self.service in ('aws', 's3'):
      return boto3.client(
        's3',
        aws_access_key_id=self.credentials['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=self.credentials['AWS_SECRET_ACCESS_KEY'],
        region_name='us-east-1',
      )
    elif self.service == 'matrix':
      return boto3.client(
        's3',
        aws_access_key_id=self.credentials['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=self.credentials['AWS_SECRET_ACCESS_KEY'],
        endpoint_url='https://s3-hpcrc.rc.princeton.edu',
      )
    else:
      raise UnsupportedProtocolError("{} unknown. Choose from 's3' or 'matrix'.", self.service)
      
  def close(self, conn):
    try:
      return conn.close()
    except AttributeError:
      pass # AttributeError: 'S3' object has no attribute 'close' on shutdown


class GCloudBucketPool(ConnectionPool):
  def __init__(self, bucket):
    self.bucket = bucket
    self.project, self.credentials = google_credentials(bucket)
    super(GCloudBucketPool, self).__init__()

  @retry
  def _create_connection(self):
    client = Client(
      credentials=self.credentials,
      project=self.project,
    )

    return client.bucket(self.bucket)
