"""
Storage is a multithreaded key-value object
management client that supports GET, PUT, DELETE,
and LIST operations.

It can support any key-value storage system and 
currently supports local filesystem, Google Cloud Storage,
and Amazon S3 interfaces.

Single threaded, Python (preemptive) threads and 
green (cooperative) threads are available as

SimpleStorage, ThreadedStorage, and GreenStorage respectively.

Storage is an alias for ThreadedStorage
"""

from .storage import (
  SimpleStorage, ThreadedStorage, GreenStorage,  
  DEFAULT_THREADS
)
from .storage_interfaces import reset_connection_pools

# For backwards compatibility

class Storage(object):
  def __new__(kls, *args, **kwargs):
    green = False
    if 'green' in kwargs:
      green = bool(kwargs['green'])
      del kwargs['green']

    if green:
      return GreenStorage(*args, **kwargs)
    return ThreadedStorage(*args, **kwargs)
