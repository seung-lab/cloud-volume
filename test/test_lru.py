import pytest

from cloudvolume.lru import LRU

import time
import random
import sys

def test_lru_size_w_tuple():
  lru = LRU(int(1e6), size_in_bytes=True)
  assert lru.nbytes == 0
  val = (1, b'0' * 1000, b'1' * 1000)
  lru[1] = val

  ans = (
    sys.getsizeof(val) 
    + sys.getsizeof(val[0])
    + sys.getsizeof(val[1])
    + sys.getsizeof(val[2])
  )
  assert 2000 < ans < 2200
  assert lru.nbytes == ans
  val = b'2' * 500
  lru[2] = val
  ans += sys.getsizeof(val)
  assert lru.nbytes == ans

  val2 = b'3' * 1000
  lru[2] = val2
  ans -= sys.getsizeof(val)
  ans += sys.getsizeof(val2)
  assert lru.nbytes == ans

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






