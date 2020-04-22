from ..precomputed.sharding import ShardingSpecification, ShardReader

class GrapheneShardReader(ShardReader):
  def compute_shard_location(self, label):
    shard_loc = self.spec.compute_shard_location(label)
    chunk_pos_num = self.meta.meta.decode_chunk_position_number(label)
    filename = str(chunk_pos_num) + '-' + str(shard_loc.shard_number) + '.shard'
    return (filename, shard_loc.minishard_number)


 