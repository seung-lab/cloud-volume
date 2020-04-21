from ...precomputed.mesh import PrecomputedMeshMetadata

class GrapheneMeshMetadata(PrecomputedMeshMetadata):
  def sharding(self, layer_id):
    return self.info['sharding'][str(layer_id)]