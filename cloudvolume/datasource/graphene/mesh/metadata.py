from __future__ import annotations

from typing import Any

from ...precomputed.mesh import PrecomputedMeshMetadata

class GrapheneMeshMetadata(PrecomputedMeshMetadata):
  def sharding(self, layer_id: int) -> Any:
    return self.info['sharding'][str(layer_id)]

  @property
  def sharded_mesh_dir(self) -> str:
    return "initial"

  @property
  def unsharded_mesh_dir(self) -> str:
    return self.meta.unsharded_mesh_dir
