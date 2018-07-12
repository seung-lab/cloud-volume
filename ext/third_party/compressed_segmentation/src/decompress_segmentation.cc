/**
 * @license LICENSE_JANELIA.txt
 *
 * \author Stephen Plaza (plazas@janelia.hhmi.org)
*/
 

#include "decompress_segmentation.h"

#include <algorithm>
#include <unordered_map>
#include <iostream>

using std::min;

namespace compress_segmentation {

constexpr size_t kBlockHeaderSize = 2;

template <class Label>
void DecompressChannel(const uint32_t* input,
                     const ptrdiff_t volume_size[3],
                     const ptrdiff_t block_size[3],
                     std::vector<Label>* output) 
{
  const size_t base_offset = output->size();
  const size_t num_elements = volume_size[0]*volume_size[1]*volume_size[2]; 
  output->resize(base_offset + num_elements);

  // determine number of grids for volume specified and block size
  // (must match what was encoded) 
  ptrdiff_t grid_size[3];
  for (size_t i = 0; i < 3; ++i) {
    grid_size[i] = (volume_size[i] + block_size[i] - 1) / block_size[i];
  }
  
  ptrdiff_t block[3];
  for (block[2] = 0; block[2] < grid_size[2]; ++block[2]) {
    for (block[1] = 0; block[1] < grid_size[1]; ++block[1]) {
      for (block[0] = 0; block[0] < grid_size[0]; ++block[0]) {
        const size_t block_offset =
            block[0] + grid_size[0] * (block[1] + grid_size[1] * block[2]);
        
        size_t encoded_bits, tableoffset, encoded_value_start;
        tableoffset = input[block_offset * kBlockHeaderSize] & 0xffffff;
        encoded_bits = (input[block_offset * kBlockHeaderSize] >> 24) & 0xff;
        encoded_value_start = input[block_offset * kBlockHeaderSize + 1];
        size_t table_entry_size = sizeof(Label)/4;

        // find absolute positions in output array (+ base_offset)
        size_t xmin = block[0]*block_size[0];
        size_t xmax = min(xmin + block_size[0], size_t(volume_size[0]));

        size_t ymin = block[1]*block_size[1];
        size_t ymax = min(ymin + block_size[1], size_t(volume_size[1]));

        size_t zmin = block[2]*block_size[2];
        size_t zmax = min(zmin + block_size[2], size_t(volume_size[2]));

        uint64_t bitmask = (1<<encoded_bits)-1;
        for (size_t z = zmin; z < zmax; ++z) {
            for (size_t y = ymin; y < ymax; ++y) {
                size_t outindex = (z*(volume_size[1]) + y)*volume_size[0] + xmin + base_offset;
                size_t bitpos = block_size[0] * ((z-zmin) * (block_size[1]) +
                         (y-ymin)) * encoded_bits;
                for (size_t x = xmin; x < xmax; ++x, ++outindex) {
                    size_t bitshift = bitpos % 32;

                    size_t arraypos = bitpos / (32);
                    size_t bitval = 0;
                    if (encoded_bits > 0) {
                        bitval = (input[encoded_value_start + arraypos] >> bitshift) & bitmask; 
                    }
                    Label val = input[tableoffset + bitval*table_entry_size];
                    if (table_entry_size == 2) {
                        val |=  uint64_t(input[tableoffset + bitval*table_entry_size+1]) << 32;
                    }
                    (*output)[outindex] = val;
                    bitpos += encoded_bits; 
                }
            }
        }
      }
    }
  }
}

template <class Label>
void DecompressChannels(const uint32_t* input,
                      const ptrdiff_t volume_size[4],
                      const ptrdiff_t block_size[3],
                      std::vector<Label>* output)
{
  for (size_t channel_i = 0; channel_i < volume_size[3]; ++channel_i) {
    DecompressChannel(input + input[channel_i], volume_size, block_size, output);
  }
}

#define DO_INSTANTIATE(Label)                                        \
  template void DecompressChannel<Label>(                              \
      const uint32_t* input, const ptrdiff_t volume_size[3],       \
      const ptrdiff_t block_size[3], \
      std::vector<Label>* output);                                \
  template void DecompressChannels<Label>(                             \
      const uint32_t* input, const ptrdiff_t volume_size[4],            \
      const ptrdiff_t block_size[3], \
      std::vector<Label>* output);                                \
/**/

DO_INSTANTIATE(uint32_t)
DO_INSTANTIATE(uint64_t)

#undef DO_INSTANTIATE

}  // namespace compress_segmentation
