"""

"""


class AlignmentError(Exception):
  """Signals that an operation requiring chunk alignment was not aligned."""
  pass 

class EmptyVolumeException(Exception):
  """Raised upon finding a missing chunk."""
  pass

class EmptyRequestException(Exception):
  """
  Requesting uploading or downloading 
  a bounding box of less than one cubic voxel
  is impossible.
  """
  pass

class DecompressionError(Exception):
  """
  Decompression failed. This exception is used for codecs 
  that are naieve to data contents like gzip, lzma, etc. as opposed
  to codecs that are aware of array shape like fpzip or compressed_segmentation.
  """
  pass

class CompressionError(Exception):
  """
  Compression failed. This exception is used for codecs 
  that are naieve to data contents like gzip, lzma, etc. as opposed
  to codecs that are aware of array shape like fpzip or compressed_segmentation.
  """
  pass