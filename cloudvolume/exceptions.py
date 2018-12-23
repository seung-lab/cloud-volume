from six.moves import http_client
import six

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


# Lookup tables for mapping exceptions from HTTP Storage interface.
# Populated by _HTTPInterfaceErrorMeta
_HTTP_CODE_TO_EXCEPTION = {}

class _HTTPInterfaceErrorMeta(type):
  """Metaclass for registering HTTPInterfaceError subclasses."""

  def __new__(mcs, name, bases, class_dict):
    cls = type.__new__(mcs, name, bases, class_dict)
    if cls.status_code is not None:
      _HTTP_CODE_TO_EXCEPTION.setdefault(cls.status_code, cls)
    return cls


@six.python_2_unicode_compatible
@six.add_metaclass(_HTTPInterfaceErrorMeta)
class HTTPInterfaceError(Exception):
  """Base class for exceptions raised by the HTTP Storage interface.

  Args:
    message (str): The exception message.
    response (Union[hyper.HTTP11Response, hyper.HTTP20Response]): response object, optional.
  """

  status_code = None
  """Optional[int]: The HTTP status code associated with this error.

  This may be ``None`` if the exception does not have a direct mapping
  to an HTTP error.

  See http://www.w3.org/Protocols/rfc2616/rfc2616-sec10.html
  """

  def __init__(self, message, response=None):
    super(HTTPInterfaceError, self).__init__(message)
    self.message = message
    self._response = response

  def __str__(self):
    return '[%s]: %s' % (self.status_code, self.message)

  @property
  def response(self):
    return self._response


class HTTPClientError(HTTPInterfaceError):
  """Base class for all client error (HTTP 4xx) responses."""


class BadRequest(HTTPClientError):
  """Exception mapping a ``400 Bad Request`` response."""
  status_code = http_client.BAD_REQUEST


class Unauthorized(HTTPClientError):
  """Exception mapping a ``401 Unauthorized`` response."""
  status_code = http_client.UNAUTHORIZED


class Forbidden(HTTPClientError):
  """Exception mapping a ``403 Forbidden`` response."""
  status_code = http_client.FORBIDDEN


class NotFound(HTTPClientError):
  """Exception mapping a ``404 Not Found`` response."""
  status_code = http_client.NOT_FOUND


class MethodNotAllowed(HTTPClientError):
  """Exception mapping a ``405 Method Not Allowed`` response."""
  status_code = http_client.METHOD_NOT_ALLOWED


class Conflict(HTTPClientError):
  """Exception mapping a ``409 Conflict`` response."""
  status_code = http_client.CONFLICT


class LengthRequired(HTTPClientError):
  """Exception mapping a ``411 Length Required`` response."""
  status_code = http_client.LENGTH_REQUIRED


class PreconditionFailed(HTTPClientError):
  """Exception mapping a ``412 Precondition Failed`` response."""
  status_code = http_client.PRECONDITION_FAILED


class RequestedRangeNotSatisfiable(HTTPClientError):
  """Exception mapping a ``416 Range Not Satisfiable`` response."""
  status_code = http_client.REQUESTED_RANGE_NOT_SATISFIABLE


class TooManyRequests(HTTPClientError):
  """Exception mapping a ``429 Too Many Requests`` response."""
  try:
    status_code = http_client.TOO_MANY_REQUESTS
  except AttributeError:  # Python 2.7
    status_code = (429, 'Too Many Requests',
      'The user has sent too many requests in '
      'a given amount of time ("rate limiting")')


class HTTPServerError(HTTPInterfaceError):
  """Base for 5xx responses."""


class InternalServerError(HTTPServerError):
  """Exception mapping a ``500 Internal Server Error`` response."""
  status_code = http_client.INTERNAL_SERVER_ERROR


class MethodNotImplemented(HTTPServerError):
  """Exception mapping a ``501 Not Implemented`` response."""
  status_code = http_client.NOT_IMPLEMENTED


class BadGateway(HTTPServerError):
  """Exception mapping a ``502 Bad Gateway`` response."""
  status_code = http_client.BAD_GATEWAY


class ServiceUnavailable(HTTPServerError):
  """Exception mappping a ``503 Service Unavailable`` response."""
  status_code = http_client.SERVICE_UNAVAILABLE


class GatewayTimeout(HTTPServerError):
  """Exception mapping a ``504 Gateway Timeout`` response."""
  status_code = http_client.GATEWAY_TIMEOUT


def from_http_status(status_code, message, **kwargs):
  """Create a :class:`HTTPInterfaceError` from an HTTP status code.
  Args:
    status_code (int): The HTTP status code.
    message (str): The exception message.
    kwargs: Additional arguments passed to the :class:`HTTPInterfaceError`
      constructor.
  Returns:
    HTTPInterfaceError: An instance of the appropriate subclass of
      :class:`HTTPInterfaceError`.
  """
  error_class = _HTTP_CODE_TO_EXCEPTION.get(status_code, HTTPInterfaceError)
  error = error_class(message, **kwargs)

  if error.status_code is None:
    error.status_code = status_code

  return error
