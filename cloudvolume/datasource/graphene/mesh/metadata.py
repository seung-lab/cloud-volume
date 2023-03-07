from ...precomputed.mesh import PrecomputedMeshMetadata

class GrapheneMeshMetadata(PrecomputedMeshMetadata):
  def sharding(self, layer_id):
    return self.info['sharding'][str(layer_id)]

  @property
  def sharded_mesh_dir(self):
    return "initial"
  
  @property
  def unsharded_mesh_dir(self):
    return self.meta.unsharded_mesh_dir
