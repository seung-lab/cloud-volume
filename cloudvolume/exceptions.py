from cloudfiles.exceptions import CompressionError, DecompressionError
from osteoid.exceptions import (
  SkeletonUnassignedEdgeError, 
  SkeletonDecodeError,
  SkeletonEncodeError,
  SkeletonTransformError,
  SkeletonAttributeMixingError,
)

class AuthenticationError(Exception):
  """Incorrect credentials."""
  pass

class DimensionError(Exception):
  """Wrong number of dimensions."""
  pass

class InfoError(Exception):
  pass

class InfoUnavailableError(InfoError):
  """CloudVolume was unable to access this layer's info file."""
  pass

class RedirectError(InfoError):
  pass

class TooManyRedirects(RedirectError):
  """The chain of redirects became unconvincingly long."""
  pass

class CyclicRedirect(RedirectError):
  """Unable to resolve redirects due to cycle."""
  pass

class ScaleUnavailableError(IndexError):
  """The info file is not configured to support this scale / mip level."""
  pass

class ReadOnlyException(Exception):
  """Attempted to write to a readonly data source."""
  pass

class AlignmentError(ValueError):
  """Signals that an operation requiring chunk alignment was not aligned."""
  pass 

class EmptyVolumeException(Exception):
  """Raised upon finding a missing chunk."""
  pass

class EmptyFileException(Exception):
  """File was zero bytes."""
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

class SubvoxelVolumeError(ValueError):
  """
  Raised upon requesting a bounding box with less than one voxel 
  of volume.
  """

class UnsupportedCompressionType(ValueError):
  """
  Raised when attempting to use a compression type which is unsupported
  by the storage interface.
  """

# Inheritance below done for backwards compatibility reasons.

class MeshDecodeError(ValueError):
  """Unable to decode a mesh object."""
  pass

class UnsupportedFormatError(Exception):
  """Unable to interpret the format of this URI. e.g. precomputed://"""
  pass

class UnsupportedProtocolError(ValueError):
  """Unknown protocol extension."""
  pass

class UnsupportedGrapheneAPIVersionError(Exception):
  """This dataset does not support the specified api version."""
  pass

class SpecViolation(Exception):
  """The values held by this object violate its written specification."""
  pass

class SpatialIndexGapError(Exception):
  """Part of the spatial index was not found. A complete result set cannot be fetched."""
  pass

class WriteLockAcquisitionError(Exception):
  """Unable to obtain a lock on this data layer element."""
  pass

class WriteLockReleaseError(Exception):
  """Unable to release a lock on this data layer element."""
  pass