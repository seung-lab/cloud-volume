from .storage import (
  ThreadedStorage, GreenStorage, SimpleStorage, 
  DEFAULT_THREADS
)
from .storage_interfaces import reset_connection_pools

Storage = ThreadedStorage # For backwards compatibility

