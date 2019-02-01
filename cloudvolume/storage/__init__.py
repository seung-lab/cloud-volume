from .storage import (
  ThreadedStorage, GreenStorage, SimpleStorage, 
  DEFAULT_THREADS
)
from .storage_interfaces import reset_connection_pools

import sys

if sys.version_info[0] < 3:
  Storage = ThreadedStorage
else:
  Storage = GreenStorage