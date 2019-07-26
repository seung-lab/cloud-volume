import numpy as np

uint64 = np.uint64

def murmurHash3_x86_128Mix(h):
  h = uint64(h)
  h ^= h >> uint64(16)
  h *= uint64(0x85ebca6b)
  h ^= h >> uint64(13)
  h *= uint64(0xc2b2ae35)
  h ^= h >> uint64(16)
  return h

def rotl32(x, r):
  x, r = uint64(x), uint64(r)
  return uint64(x << r) | (x >> (uint64(32) - r))


# MurmurHash3_x86_128, specialized for 8 bytes of input.
#
# Only the low 8 bytes of output are returned.
#
# Ported from Google's Apache2 Licensed Typescript
# https://github.com/google/neuroglancer/blob/master/src/neuroglancer/util/hash.ts
#
#
def murmurHash3_x86_128Hash64Bits(x, seed):
  low_mask = uint64(0x00000000ffffffff)
  low = uint64(x & low_mask)
  high = uint64(x) >> uint64(32)
  seed = uint64(seed)

  h1, h2, h3, h4 = uint64(seed), uint64(seed), uint64(seed), uint64(seed) 
  c1 = uint64(0x239b961b)
  c2 = uint64(0xab0e9789)
  c3 = uint64(0x38b34ae5)
  # c4 = uint64(0xa1e38b93)

  k2 = high * c2
  k2 = rotl32(k2, uint64(16))
  k2 *= c3
  h2 ^= k2

  k1 = low * c1
  k1 = rotl32(k1, uint64(15))
  k1 *= c2
  h1 ^= k1

  length = uint64(8)

  h1 ^= length
  h2 ^= length
  h3 ^= length
  h4 ^= length

  zero = uint64(0)

  h1 = (h1 + h2) >> zero
  h1 = (h1 + h3) >> zero
  h1 = (h1 + h4) >> zero
  h2 = (h2 + h1) >> zero
  h3 = (h3 + h1) >> zero
  h4 = (h4 + h1) >> zero

  h1 = murmurHash3_x86_128Mix(h1)
  h2 = murmurHash3_x86_128Mix(h2)
  h3 = murmurHash3_x86_128Mix(h3)
  h4 = murmurHash3_x86_128Mix(h4)

  h1 = (h1 + h2) >> zero
  h1 = (h1 + h3) >> zero
  h1 = (h1 + h4) >> zero
  h2 = (h2 + h1) >> zero

  # h3 = (h3 + h1) >> zero
  # h4 = (h4 + h1) >> zero

  return uint64((h1 & low_mask) | (h2 << uint64(32)))

