from __future__ import annotations

import numpy as np
import numpy.typing as npt


def apply_transform(vertices: npt.NDArray, transform: npt.NDArray) -> npt.NDArray:
  """
  Scales mesh vertices from stored coordinate
  space to physical coordinates.

  vertices: a Nx3 array of X,Y,Z floats
  transform: a 4x4 affine transform matrix
    from stored to physical coordinates.
  """
  if (transform == np.eye(4)).all():
    return vertices

  n_verts = len(vertices)
  vertices = np.hstack([vertices, np.ones((n_verts,1)) ]).T
  return np.matmul(transform[:3,:], vertices).T
