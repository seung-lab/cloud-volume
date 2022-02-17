import pytest

from cloudvolume.lru import LRU

import time
import random
import sys

@pytest.mark.parametrize("size_in_bytes", (False,True))
def test_lru(size_in_bytes):
  base_size = 5
  size = base_size
  small_int_bytes = sys.getsizeof(100)
  if size_in_bytes:
    size *= small_int_bytes

  lru = LRU(size, size_in_bytes=size_in_bytes)

  assert len(lru) == 0
  for i in range(5):
    lru[i] = i
  assert len(lru) == 5
  assert 0 < lru.nbytes <= small_int_bytes * base_size

  for i in range(5):
    lru[i] = i
  assert 0 < lru.nbytes <= small_int_bytes * base_size

  for i in range(100):
    lru[i] = i
  assert 0 < lru.nbytes <= small_int_bytes * base_size

  assert len(lru) == 5

  lru.resize(size * 2)
  for i in range(5):
    lru[i] = i
  assert 0 < lru.nbytes <= small_int_bytes * base_size * 2

  assert lru.queue.tolist() == [ 
    (4,4), (3,3), (2,2), (1,1), (0,0),
    (99,99), (98,98), (97,97), (96,96), (95,95)
  ]

  lru.resize(size)
  assert lru.queue.tolist() == [ 
    (4,4), (3,3), (2,2), (1,1), (0,0)
  ]
  assert 0 < lru.nbytes <= small_int_bytes * base_size

  assert lru[0] == 0
  assert lru.queue.tolist() == [ 
    (0,0), (4,4), (3,3), (2,2), (1,1)
  ]

def test_lru_chaos():
  lru = LRU(10)

  seed = time.time()
  print("seed", seed)

  random.seed(seed)
  for i in range(150):
    rand = random.randint(-5000, 5000)
    lru[rand] = random.randint(-5000, 5000)

    if rand % 17 == 0:
      keys = lru.keys()
      key = random.choice(list(keys))
      lru[key]






