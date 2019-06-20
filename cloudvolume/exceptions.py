
class InfoUnavailableError(ValueError):
  """CloudVolume was unable to access this layer's info file."""
  pass

class ScaleUnavailableError(IndexError):
  """The info file is not configured to support this scale / mip level."""
  pass

class AlignmentError(ValueError):
  """Signals that an operation requiring chunk alignment was not aligned."""
  pass 

class EmptyVolumeException(Exception):
  """Raised upon finding a missing chunk."""
  pass

class EmptyRequestException(ValueError):
  """
  Requesting uploading or downloading 
  a bounding box of less than one cubic voxel
  is impossible.
  """
  pass

class DecodingError(Exception):
  """Generic decoding error. Applies to content aware and unaware codecs."""
  pass

class EncodingError(Exception):
  """Generic decoding error. Applies to content aware and unaware codecs."""
  pass

class OutOfBoundsError(ValueError):
  """
  Raised upon trying to obtain or assign to a bbox of a volume outside
  of the volume's bounds
  """

# Inheritance below done for backwards compatibility reasons.

class DecompressionError(DecodingError):
  """
  Decompression failed. This exception is used for codecs 
  that are naieve to data contents like gzip, lzma, etc. as opposed
  to codecs that are aware of array shape like fpzip or compressed_segmentation.
  """
  pass

class CompressionError(EncodingError):
  """
  Compression failed. This exception is used for codecs 
  that are naieve to data contents like gzip, lzma, etc. as opposed
  to codecs that are aware of array shape like fpzip or compressed_segmentation.
  """
  pass

class SkeletonUnassignedEdgeError(Exception):
  """This skeleton has an edge to a vertex that doesn't exist."""
  pass

class SkeletonDecodeError(Exception):
  """Unable to decode a binary skeleton into a Python object."""
  pass

class SkeletonEncodeError(Exception):
  """Unable to encode a PrecomputedSkeleton into a binary object."""
  pass

class UnsupportedProtocolError(ValueError):
  """Unknown protocol extension."""
  pass