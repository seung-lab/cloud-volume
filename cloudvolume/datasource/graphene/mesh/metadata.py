from ...precomputed.mesh import PrecomputedMeshMetadata

class GrapheneMeshMetadata(PrecomputedMeshMetadata):
  def sharding(self, layer_id):
    return self.info['sharding'][str(layer_id)]

  @property
  def initial_mesh_cloudpath(self):
    mesh_meta = self.info.get("mesh_metadata", {})
    return mesh_meta.get(
      "initial_mesh_path", 
      self.meta.join(self.cloudpath, self.mesh_path, self.sharded_mesh_dir)
    )

  @property
  def dynamic_mesh_cloudpath(self):
    mesh_meta = self.info.get("mesh_metadata", {})
    return mesh_meta.get(
      "dynamic_mesh_path", 
      self.meta.join(self.cloudpath, self.mesh_path, self.unsharded_mesh_dir)
    )

  @property
  def sharded_mesh_dir(self):
    return "initial"
  
  @property
  def unsharded_mesh_dir(self):
    return self.meta.unsharded_mesh_dir
