from typing import Dict, Tuple, Sequence, Union, Optional

import re
import os

import numpy as np
from cloudfiles import CloudFiles

from .. import autocropfn, readonlyguard, ImageSourceInterface

from ... import compression
from ... import chunks
from ... import exceptions 
from ...lib import ( 
  colorize, red, mkdir, Vec, Bbox,  
  jsonify, generate_random_string,
  xyzrange, BboxLikeType
)
from ... import paths
from ...types import CompressType, MipType
from ...volumecutout import VolumeCutout
from ..precomputed.image.common import shade
from ..precomputed.image import xfer

class N5ImageSource(ImageSourceInterface):
  def __init__(
    self, config, meta, cache,
    autocrop=False, bounded=True,
    non_aligned_writes=False,
    delete_black_uploads=False,
    fill_missing=False,
    readonly=False,
  ):
    self.config = config
    self.meta = meta 
    self.cache = cache 

    self.autocrop = bool(autocrop)
    self.bounded = bool(bounded)
    self.fill_missing = bool(fill_missing)
    self.non_aligned_writes = bool(non_aligned_writes)
    self.readonly = bool(readonly)

  def parse_chunk(self, binary, mip, filename, default_shape):
    if binary is None:
      if self.fill_missing:
        data = np.zeros(default_shape, dtype=self.meta.dtype, order="F")
        shape = list(self.meta.chunk_size(mip)) + [ self.meta.num_channels ]
        return data, shape
      else:
        raise exceptions.EmptyVolumeException(f"{filename} is missing.")

    toint = lambda n: int.from_bytes(n, byteorder="big", signed=False)

    mode = toint(binary[0:2])

    if mode != 0:
      raise exceptions.DecodingError(
        f"This implementation cannot read volumes "
        f"with mode != 0. Got mode: {mode}"
      )

    ndim = toint(binary[2:4])
    dims = [ toint(binary[4+4*i:4+4*(i+1)]) for i in range(ndim) ]
    while len(dims) < 4:
      dims.append(1)
    dims[3] = self.meta.num_channels

    compressed_stream = binary[4+4*ndim:]
    compressed_stream = compression.decompress(
      compressed_stream, self.meta.encoding(mip), filename
    )

    data = chunks.decode(
      compressed_stream, 
      encoding='raw', 
      shape=dims, 
      dtype=self.meta.dtype,
    )
    return data, dims

  def download(self, bbox, mip, parallel=1, renumber=False):
    if parallel != 1:
      raise ValueError("Only parallel=1 is supported for n5.")
    elif renumber != False:
      raise ValueError("Only renumber=False is supported for n5.")

    bounds = Bbox.clamp(bbox, self.meta.bounds(mip))

    if self.autocrop:
      image, bounds = autocropfn(self.meta, image, bounds, mip)
    
    if bounds.subvoxel():
      raise exceptions.EmptyRequestException(f'Requested less than one pixel of volume. {bounds}')

    cf = CloudFiles(self.meta.cloudpath, progress=self.config.progress)
    realized_bbox = bbox.expand_to_chunk_size(self.meta.chunk_size(mip))
    grid_bbox = realized_bbox // self.meta.chunk_size(mip)

    urls = [
      cf.join(f"s{mip}", str(x), str(y), str(z))
      for x,y,z in xyzrange(grid_bbox.minpt, grid_bbox.maxpt)
    ]

    all_chunks = cf.get(urls, parallel=parallel, return_dict=True)
    shape = list(bbox.size3()) + [ self.meta.num_channels ]
    renderbuffer = np.zeros(shape=shape, dtype=self.meta.dtype, order='F')

    sep = '/'
    if cf._path.protocol == "file":
      sep = os.path.sep
    if sep == '\\':
      sep = '\\\\' # compensate for regexp escaping

    regexp = re.compile(rf"s(?P<mip>\d+){sep}(?P<x>\d+){sep}(?P<y>\d+){sep}(?P<z>\d+)")
    for fname, binary in all_chunks.items():
      m = re.search(regexp, fname).groupdict()
      assert mip == int(m["mip"])
      gridpoint = Vec(*[ int(i) for i in [ m["x"], m["y"], m["z"] ] ])
      chunk_bbox = Bbox(gridpoint, gridpoint + 1) * self.meta.chunk_size(mip)
      chunk_bbox = Bbox.clamp(chunk_bbox, self.meta.bounds(mip))
      default_shape = list(chunk_bbox.size3()) + [ self.meta.num_channels ]
      chunk, chunk_shape = self.parse_chunk(binary, mip, fname, default_shape)
      chunk_bbox = Bbox(chunk_bbox.minpt, chunk_bbox.minpt + Vec(*chunk_shape[:3]))
      chunk_bbox = Bbox.clamp(chunk_bbox, self.meta.bounds(mip))
      shade(renderbuffer, bbox, chunk, chunk_bbox)

    return VolumeCutout.from_volume(self.meta, mip, renderbuffer, bbox)

  @readonlyguard
  def upload(self, image, offset, mip, parallel:int = 1):
    raise NotImplementedError()

  def exists(self, bbox, mip=None):
    raise NotImplementedError()

  @readonlyguard
  def delete(self, bbox, mip=None):
    raise NotImplementedError()

  def transfer_to(
    self,
    cloudpath:str, 
    bbox:BboxLikeType, 
    mip:MipType, 
    block_size:Optional[int] = None, 
    compress:CompressType = True, 
    compress_level:Optional[int] = None, 
    encoding:Optional[str] = None,
    sharded:Optional[bool] = None,
  ):
    pth = paths.extract(cloudpath)

    if pth.format != self.meta.path.format and encoding is None:
      if pth.format == "zarr":
        encoding = "blosc"
      elif pth.format == "precomputed":
        encoding = "raw"

    return xfer.transfer_by_rerendering(
      self, cloudpath,
      bbox=bbox,
      mip=mip,
      compress=compress,
      compress_level=compress_level,
      encoding=encoding,
    )