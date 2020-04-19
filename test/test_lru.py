from cloudvolume.lru import LRU

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
