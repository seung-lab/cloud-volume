from typing import Optional

import copy
import re
import os

import numpy as np
from cloudfiles import CloudFiles
import cloudfiles.compression

from .. import (
  autocropfn, readonlyguard, 
  ImageSourceInterface, check_grid_aligned,
  generate_chunks
)

from ...types import CompressType, MipType
from ... import compression
from ... import exceptions 
from ...lib import ( 
  colorize, red, mkdir, Vec, Bbox, BboxLikeType, 
  jsonify, generate_random_string,
  xyzrange
)
from ...paths import extract
from ...volumecutout import VolumeCutout
from ..precomputed.image.common import shade
from ..precomputed.image import xfer

class Zarr3ImageSource(ImageSourceInterface):
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

  def decode_chunk(self, binary, mip, filename, default_shape):
    if binary is None:
      if self.fill_missing:
        return None
      else:
        raise exceptions.EmptyVolumeException(f"{filename} is missing.")

    codecs = self.meta.codecs(mip)
    arr = binary

    transposed = False

    for codec in reversed(codecs):
      encoding = codec["name"]
      if encoding == "bytes":
        arr = np.frombuffer(arr, dtype=self.meta.dtype)
        arr = arr.reshape(default_shape, order='C')
        continue
      elif encoding == "brotli":
        encoding = "br"
      elif encoding == "zlib":
        encoding = "gzip"
      elif encoding == "lzma":
        encoding = "xz"

      if encoding == "blosc":
        import blosc
        arr = blosc.decompress(arr)
      elif encoding == "crc32c":
        import crc32c
        stored_crc = int.from_bytes(arr[-4:], byteorder='little')
        calculated_crc = crc32c.crc32c(arr[:-4])
        if stored_crc != calculated_crc:
          raise ValueError(
            f"Stored crc32c {stored_crc} did not match "
            f"calculated crc32c {calculated_crc} for file {filename}."
          )
        arr = arr[:-4]
      elif encoding in ["zstd", "xz", "br", "gzip"]:
        arr = cloudfiles.compression.decompress(arr, encoding, filename)
      elif encoding == "transpose":
        transposed = True
        arr = np.transpose(arr, axes=codec["configuration"]["order"])
      else:
        raise exceptions.DecodingError(f"Unsupported decoding method: {encoding}")
    
    if not transposed:
      arr = arr.T

    return arr

  def encode_chunk(self, arr:np.ndarray, mip:int) -> bytes:
    codecs = self.meta.codecs(mip)

    transposed = False

    binary = arr
    for codec in codecs:
      encoding = codec["name"]

      if encoding == "bytes":
        order = 'C'
        if not transposed:
          order = 'F'
        binary = binary.tobytes(order)
        continue
      elif encoding == "brotli":
        encoding = "br"
      elif encoding == "zlib":
        encoding = "gzip"
      elif encoding == "lzma":
        encoding = "xz"

      if encoding == "blosc":
        import blosc
        binary = blosc.compress(binary)
      elif encoding == "crc32c":
        import crc32c
        calculated_crc = crc32c.crc32c(binary)
        binary += calculated_crc.to_bytes(4, byteorder='little')
      elif encoding in ["zstd", "xz", "br", "gzip"]:
        compress_level = codec.get("configuration", { "level": 1 })
        compress_level = compress_level.get("level", 1)
        binary = cloudfiles.compression.compress(binary, encoding, compress_level=compress_level)
      elif encoding == "transpose":
        transposed = True
        binary = np.transpose(binary, axes=codec["configuration"]["order"])
      else:
        raise exceptions.DecodingError(f"Unsupported decoding method: {encoding}")

    return binary

  def download(
    self, 
    bbox:BboxLikeType, 
    mip:MipType, 
    parallel:int = 1, 
    renumber:bool = False, 
    label:Optional[int] = None,
    t:int = 0,
  ) -> VolumeCutout:
    if self.meta.is_sharded(mip):
      raise NotImplementedError("sharded volumes are not currently supported.")
    elif parallel != 1:
      raise ValueError("Only parallel=1 is supported for zarr.")
    elif renumber != False:
      raise ValueError("Only renumber=False is supported for zarr.")

    bounds = Bbox.clamp(bbox, self.meta.bounds(mip))

    if self.autocrop:
      image, bounds = autocropfn(self.meta, image, bounds, mip)
    
    if bounds.subvoxel():
      raise exceptions.EmptyRequestException(f'Requested less than one pixel of volume. {bounds}')

    cf = CloudFiles(
      self.meta.cloudpath, 
      progress=self.config.progress, 
      secrets=self.config.secrets,
      green=self.config.green,
    )

    spatial_chunk_size = self.meta.spatial_chunk_size(mip)
    realized_bbox = bounds.expand_to_chunk_size(spatial_chunk_size, offset=self.meta.voxel_offset(mip))

    paths = self._chunknames(
      realized_bbox, self.meta.bounds(mip), 
      mip, spatial_chunk_size,
      t=t,
    )

    all_chunks = cf.get(paths, parallel=parallel, return_dict=True)
    shape = list(bounds.size3()) + [ self.meta.num_channels ]

    dtype = self.meta.dtype
    if label is not None:
      dtype = bool

    if self.meta.background_color(mip) == 0:
      renderbuffer = np.zeros(shape=shape, dtype=dtype, order="F")
    else:
      renderbuffer = np.full(
        shape=shape,
        fill_value=self.meta.background_color(mip),
        dtype=dtype,
        order="F",
      )

    regexp = self.meta.filename_regexp(mip)

    axis_mapping = self.meta.zarr_axes_to_cv_axes()

    tslice = 0
    taxis = self.meta.has_time_axis()

    if taxis:
      tslice = t - tchunk * self.meta.time_chunk_size(mip)

    voxel_offset = self.meta.voxel_offset(mip)

    for fname, binary in all_chunks.items():
      m = re.search(regexp, fname).groupdict()
      assert self.meta.key(mip) == m.get("mip", '')
      gridpoint = Vec(*[ int(i) for i in [ m["x"], m["y"], m["z"] ] ])
      chunk_bbox = Bbox(gridpoint, gridpoint + 1) * spatial_chunk_size
      chunk_bbox += voxel_offset
      chunk_size = chunk_bbox.size()[axis_mapping]
      chunk = self.decode_chunk(binary, mip, fname, chunk_size)
      if chunk is None:
        continue
      if taxis:
        chunk = chunk[...,tslice]
      slcs = (chunk_bbox - chunk_bbox.minpt).to_slices()

      chunk = chunk[slcs]
      if label is not None:
        chunk = chunk == label
      shade(renderbuffer, bounds, chunk, chunk_bbox, channel=int(m.get("c", 0)))

    return VolumeCutout.from_volume(self.meta, mip, renderbuffer, bounds)

  @readonlyguard
  def upload(self, image, offset, mip, parallel=1, t=0):
    if self.meta.is_sharded(mip):
      raise NotImplementedError("sharded volumes are not currently supported.")
    elif not np.issubdtype(image.dtype, np.dtype(self.meta.dtype).type):
      raise ValueError(f"""
        The uploaded image data type must match the volume data type. 

        Volume: {self.meta.dtype}
        Image: {image.dtype}
        """
      )

    shape = Vec(*image.shape)[:3]
    offset = Vec(*offset)[:3]
    bounds = Bbox( offset, shape + offset)

    is_aligned = check_grid_aligned(
      self.meta, image, bounds, mip, 
      throw_error=True, # (self.non_aligned_writes == False)
    )

    spatial_chunk_size = self.meta.spatial_chunk_size(mip)

    expanded = bounds.expand_to_chunk_size(spatial_chunk_size, self.meta.voxel_offset(mip))
    all_chunknames = self._chunknames(
      expanded, self.meta.bounds(mip), 
      mip, spatial_chunk_size,
      t=t,
    )

    all_chunks = generate_chunks(self.meta, image, offset, mip, chunk_size=spatial_chunk_size)

    to_upload = []

    bgcolor = self.meta.background_color(mip)

    def all_chunks_by_channel(all_chunks):
      for ispt, iept, vol_spt, vol_ept in all_chunks:
        for c in range(self.meta.num_channels):
          cutout = image[ ispt.x:iept.x, ispt.y:iept.y, ispt.z:iept.z, c ]
          if np.any(np.array(cutout.shape) < spatial_chunk_size):
            diff = spatial_chunk_size - np.array(cutout.shape)
            pad_width = [ (0, diff[0]), (0, diff[1]), (0, diff[2]) ]
            cutout = np.pad(cutout, pad_width, 'constant', constant_values=bgcolor)
          yield cutout

    for filename, imgchunk in zip(all_chunknames, all_chunks_by_channel(all_chunks)):
      binary = self.encode_chunk(imgchunk, mip)
      to_upload.append(
        (filename, binary)
      )

    CloudFiles(self.meta.cloudpath).puts(to_upload)

  def _chunknames(self, bbox, volume_bbox, mip, chunk_size, t):
    meta = self.meta
    cf = CloudFiles(self.meta.cloudpath)
    num_channels = self.meta.num_channels

    has_t = self.meta.has_time_axis()
    tchunk = 0
    if has_t:
      tchunk = int(t / self.meta.num_time_chunks(mip))

    axes = [ (axis["type"], axis["name"]) for axis in self.meta.axes() ]
    voxel_offset = self.meta.voxel_offset(mip)[:3]

    class ZarrChunkNamesIterator():
      def __len__(self):
        # round up and avoid conversion to float
        n_chunks = (bbox.dx + chunk_size[0] - 1) // chunk_size[0]
        n_chunks *= (bbox.dy + chunk_size[1] - 1) // chunk_size[1]
        n_chunks *= (bbox.dz + chunk_size[2] - 1) // chunk_size[2]
        return n_chunks
      def __iter__(self):
        nonlocal volume_bbox
        volume_bbox = Bbox.expand_to_chunk_size(volume_bbox, chunk_size, offset=voxel_offset)
        bbox_grid = (bbox - voxel_offset) // chunk_size

        for x,y,z in xyzrange(bbox_grid.minpt, bbox_grid.maxpt):
          for c in range(num_channels):
            params = []
            
            for typ, name in axes:
              if typ == "time":
                params.append(str(tchunk))
              elif typ == "space" and name == "x":
                params.append(str(x))
              elif typ == "space" and name == "y":
                params.append(str(y))
              elif typ == "space" and name == "z":
                params.append(str(z))
              elif typ == "channel":
                params.append(str(c))

            yield meta.chunk_name(mip, *params)

    return ZarrChunkNamesIterator()

  def exists(self, bbox, mip=None, t=0):
    if mip is None:
      mip = self.config.mip

    bounds = Bbox.clamp(bbox, self.meta.bounds(mip))

    if self.autocrop:
      image, bounds = autocropfn(self.meta, image, bounds, mip)
    
    if bounds.subvoxel():
      raise exceptions.EmptyRequestException(f'Requested less than one pixel of volume. {bounds}')

    expanded = bounds.expand_to_chunk_size(
      self.meta.chunk_size(mip), self.meta.voxel_offset(mip)
    )
    all_chunknames = self._chunknames(
      expanded, self.meta.bounds(mip), 
      mip, self.meta.chunk_size(mip), 
      t=t
    )

    cf = CloudFiles(
      self.meta.cloudpath, 
      progress=self.config.progress, 
      secrets=self.config.secrets,
      green=self.config.green,
    )

    return cf.exists(all_chunknames)

  @readonlyguard
  def delete(self, bbox, mip=None, t=0):
    if mip is None:
      mip = self.config.mip

    if mip in self.meta.locked_mips():
      raise exceptions.ReadOnlyException(
        f"MIP {mip} is currently write locked. If this should not be the case, "
        f"run vol.meta.unlock_mips({mip})."
      )

    bbox = Bbox.create(bbox, self.meta.bounds(mip), bounded=True)
    realized_bbox = bbox.expand_to_chunk_size(
      self.meta.chunk_size(mip), offset=self.meta.voxel_offset(mip)
    )
    all_chunknames = self._chunknames(
      realized_bbox, self.meta.bounds(mip), 
      mip, self.meta.chunk_size(mip),
      t=t
    )

    cf = CloudFiles(self.meta.cloudpath, progress=self.config.progress, secrets=self.config.secrets)
    cf.delete(all_chunknames)

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
    pth = extract(cloudpath)

    if pth.format != self.meta.path.format and encoding is None:
      if pth.format == "n5":
        encoding = "gzip"
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