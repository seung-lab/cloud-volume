/**
 * @license LICENSE_JANELIA.txt
 *
 * \author Stephen Plaza (plazas@janelia.hhmi.org)
*/


#include "compress_segmentation.h"
#include "decompress_segmentation.h"

#include "gtest/gtest.h"

namespace compress_segmentation {

namespace {

TEST(DecompressChannelTest, Basic) {
  std::vector<uint64_t> input{4, 3, 5, 4, 50130210214, 3, 3, 3};
  const ptrdiff_t input_strides[3] = {1, 2, 4};
  const ptrdiff_t volume_size[3] = {2, 2, 2};
  const ptrdiff_t block_size[3] = {2, 2, 1};
  std::vector<uint32_t> temp_output;
  std::vector<uint64_t> decompress_output;

  CompressChannel(input.data(), input_strides, volume_size, block_size,
                  &temp_output);

  DecompressChannel(temp_output.data(), volume_size, block_size, &decompress_output);
  
  ASSERT_EQ(input, decompress_output);
}

TEST(DecompressChannelTest, Basic2) {
  std::vector<uint64_t> input{
      4, 3, 5, 4,  //
      1, 3, 3, 3,  //
      3, 1, 1, 1,  //
      5, 5, 3, 4,  //
  };
  const ptrdiff_t input_strides[3] = {1, 2, 4};
  const ptrdiff_t volume_size[3] = {2, 2, 4};
  const ptrdiff_t block_size[3] = {2, 2, 1};
  std::vector<uint32_t> temp_output;
  std::vector<uint64_t> decompress_output;
  CompressChannel(input.data(), input_strides, volume_size, block_size,
                  &temp_output);
  DecompressChannel(temp_output.data(), volume_size, block_size, &decompress_output);
  
  ASSERT_EQ(input, decompress_output);
}

TEST(DecompressChannelTest, Basic32) {
  std::vector<uint32_t> input{
      4, 3, 5, 4,  //
      1, 3, 3, 3,  //
      3, 1, 1, 1,  //
      5, 5, 3, 4,  //
  };
  const ptrdiff_t input_strides[3] = {1, 2, 4};
  const ptrdiff_t volume_size[3] = {2, 2, 4};
  const ptrdiff_t block_size[3] = {2, 2, 1};
  std::vector<uint32_t> temp_output;
  std::vector<uint32_t> decompress_output;
  CompressChannel(input.data(), input_strides, volume_size, block_size,
                  &temp_output);
  DecompressChannel(temp_output.data(), volume_size, block_size, &decompress_output);
  ASSERT_EQ(input, decompress_output);
}

TEST(DecompressChannelTest, BasicNonBlockAligned) {
  std::vector<uint32_t> input{
      4, 3, 5, 4,  //
      1, 3, 3, 3  //
  };

  const ptrdiff_t input_strides[3] = {1, 2, 4};
  const ptrdiff_t volume_size[3] = {2, 2, 2};
  const ptrdiff_t block_size[3] = {1, 2, 1};
  std::vector<uint32_t> temp_output;
  std::vector<uint32_t> decompress_output;
  CompressChannel(input.data(), input_strides, volume_size, block_size,
                  &temp_output);
  DecompressChannel(temp_output.data(), volume_size, block_size, &decompress_output);
  ASSERT_EQ(input, decompress_output);
}

}  // namespace
}  // namespace compress_segmentation
