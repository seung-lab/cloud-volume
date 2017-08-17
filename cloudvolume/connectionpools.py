import Queue
import threading
import time
import signal
from functools import partial

from google.cloud.storage import Client
from boto.s3.connection import S3Connection

from secrets import PROJECT_NAME, google_credentials, aws_credentials

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

        def handler(signum, frame):
            self.reset_pool()

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

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

    def _close_function(self):
        return lambda x: x # no-op

    def reset_pool(self):
        closefn = self._close_function()
        while True:
            if not self.pool.qsize():
                break
            try:
                conn = self.pool.get()
                closefn(conn)
                self.pool.task_done()
            except Queue.Empty:
                break

        with self._lock:
            self.outstanding = 0

    def __del__(self):
        self.reset_pool()

class S3ConnectionPool(ConnectionPool):
    def _create_connection(self):
        return S3Connection(
            aws_credentials['AWS_ACCESS_KEY_ID'],
            aws_credentials['AWS_SECRET_ACCESS_KEY']
        )

    def _close_function(self):
        return lambda conn: conn.close()

class GCloudConnectionPool(ConnectionPool):
    def _create_connection(self):
        return Client(
            credentials=google_credentials,
            project=PROJECT_NAME
        )
