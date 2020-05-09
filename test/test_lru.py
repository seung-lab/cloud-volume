import pytest

from cloudvolume.lru import LRU

import time
import random

def test_lru():
  lru = LRU(5)

  assert len(lru) == 0
  for i in range(5):
    lru[i] = i
  assert len(lru) == 5

  for i in range(5):
    lru[i] = i

  for i in range(100):
    lru[i] = i

  assert len(lru) == 5

  lru.resize(10)
  for i in range(5):
    lru[i] = i

  assert lru.queue.tolist() == [ 
    (4,4), (3,3), (2,2), (1,1), (0,0),
    (99,99), (98,98), (97,97), (96,96), (95,95)
  ]

  lru.resize(5)
  assert lru.queue.tolist() == [ 
    (4,4), (3,3), (2,2), (1,1), (0,0)
  ]

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






