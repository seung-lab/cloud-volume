/**
 * @license LICENSE_JANELIA.txt
 *
 * \author Stephen Plaza (plazas@janelia.hhmi.org)
*/
 
/*!
 * Decompresses segmentation encoded using the format described at
 * https://github.com/google/neuroglancer/tree/master/src/neuroglancer/sliceview/compressed_segmentation.
 *
 * User must know the block size for their compressed data and the final
 * volume dimensions.
*/

#ifndef DECOMPRESS_SEGMENTATION_H_
#define DECOMPRESS_SEGMENTATION_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

namespace compress_segmentation {


// Decodes a single channel.
//
// Args:
//   input: Pointer to compressed data.
//
//   volume_size: Extent of the x, y, and z dimensions.
//
//   block_size: Extent of the x, y, and z dimensions of the block.
//
//   output: Vector to which output will be appended.
//
//   returns input pointer location
template <class Label>
void DecompressChannel(const uint32_t* input,
                     const ptrdiff_t volume_size[3],
                     const ptrdiff_t block_size[3],
                     std::vector<Label>* output);

// Encodes multiple channels.
//
// Each channel is decoded independently.
//
// The output starts with num_channels (=volume_size[3]) uint32 values
// specifying the starting offset of the encoding of each channel (the first
// offset will always equal num_channels).
//
// Args:
//
//   input: Pointer to compressed data.
//
//   volume_size: Extent of the x, y, z, and channel dimensions.
//
//   block_size: Extent of the x, y, and z dimensions of the block.
//
//   output: Vector where output will be appended.
template <class Label>
void DecompressChannels(const uint32_t* input,
                      const ptrdiff_t volume_size[4],
                      const ptrdiff_t block_size[3],
                      std::vector<Label>* output);

}  // namespace compress_segmentation

#endif  // DECOMPRESS_SEGMENTATION_H_
