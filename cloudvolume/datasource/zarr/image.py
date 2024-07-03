from typing import Optional

import copy
import re
import os

import numpy as np
from cloudfiles import CloudFiles

from .. import (
  autocropfn, readonlyguard, 
  ImageSourceInterface, check_grid_aligned,
  generate_chunks
)

from ...types import CompressType, MipType
from ... import compression
from ... import chunks
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


class ZarrImageSource(ImageSourceInterface):
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

    import blosc
    arr = np.frombuffer(blosc.decompress(binary), dtype=self.meta.dtype)
    return arr.reshape(default_shape, order=self.meta.order(mip))

  def download(
    self, 
    bbox:BboxLikeType, 
    mip:MipType, 
    parallel:int = 1, 
    renumber:bool = False, 
    label:Optional[int] = None,
    t:int = 0,
  ) -> VolumeCutout:
    if parallel != 1:
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

    cv_chunk_size = self.meta.chunk_size(mip)

    realized_bbox = bounds.expand_to_chunk_size(cv_chunk_size)
    grid_bbox = realized_bbox // cv_chunk_size

    sep = self.meta.dimension_separator(mip)

    tchunk = int(t / self.meta.num_time_chunks(mip))

    if self.meta.order(mip) == "C":
      paths = [
        cf.join(str(mip), sep.join([ str(tchunk), str(c), str(z), str(y), str(x) ]))
        for x,y,z in xyzrange(grid_bbox.minpt, grid_bbox.maxpt)
        for c in range(self.meta.num_channels)
      ]
    else:
      paths = [
        cf.join(str(mip), sep.join([ str(x), str(y), str(z), str(c), str(tchunk) ]))
        for x,y,z in xyzrange(grid_bbox.minpt, grid_bbox.maxpt)
        for c in range(self.meta.num_channels)
      ]

    all_chunks = cf.get(paths, parallel=parallel, return_dict=True)
    shape = list(bounds.size3()) + [ self.meta.num_channels ]

    if self.meta.background_color(mip) == 0:
      renderbuffer = np.zeros(shape=shape, dtype=self.meta.dtype, order="F")
    else:
      renderbuffer = np.full(
        shape=shape, fill_value=self.meta.background_color(mip), 
        dtype=self.meta.dtype, order="F",
      )

    regexp = self.meta.filename_regexp(mip)

    axis_mapping = self.meta.zarr_axes_to_cv_axes()

    tslice = t - tchunk * self.meta.time_chunk_size(mip)

    for fname, binary in all_chunks.items():
      m = re.search(regexp, fname).groupdict()
      assert mip == int(m["mip"])
      gridpoint = Vec(*[ int(i) for i in [ m["x"], m["y"], m["z"] ] ])
      chunk_bbox = Bbox(gridpoint, gridpoint + 1) * cv_chunk_size
      chunk_bbox = Bbox.clamp(chunk_bbox, self.meta.bounds(mip))
      chunk = self.decode_chunk(binary, mip, fname, self.meta.zarr_chunk_size(mip))
      if chunk is None:
        continue
      chunk = np.transpose(chunk, axes=axis_mapping)[...,tslice]
      slcs = (chunk_bbox - chunk_bbox.minpt).to_slices()
      shade(renderbuffer, bounds, chunk[slcs], chunk_bbox, channel=int(m["c"]))

    data = VolumeCutout.from_volume(self.meta, mip, renderbuffer, bounds)

    if label is not None:
      return data == label

    return data

  @readonlyguard
  def upload(self, image, offset, mip, parallel=1, t=0):
    import blosc

    if not np.issubdtype(image.dtype, np.dtype(self.meta.dtype).type):
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

    expanded = bounds.expand_to_chunk_size(self.meta.chunk_size(mip), self.meta.voxel_offset(mip))
    all_chunknames = self._chunknames(
      expanded, self.meta.bounds(mip), 
      mip, self.meta.chunk_size(mip),
      t=t,
    )

    all_chunks = generate_chunks(self.meta, image, offset, mip)
    order = self.meta.order(mip)

    to_upload = []

    axis_mapping = self.meta.cv_axes_to_zarr_axes()

    compressor_args = copy.deepcopy(self.meta.zarrays[mip]["compressor"])
    compressor_args.pop("id", None)
    compressor_args.pop("blocksize", None)

    def all_chunks_by_channel(all_chunks):
      for ispt, iept, vol_spt, vol_ept in all_chunks:
        for c in range(self.meta.num_channels):
          yield image[ ispt.x:iept.x, ispt.y:iept.y, ispt.z:iept.z, c ]

    for filename, imgchunk in zip(all_chunknames, all_chunks_by_channel(all_chunks)):
      zarr_imgchunk = np.transpose(imgchunk[..., np.newaxis], axes=axis_mapping)
      binary = zarr_imgchunk.tobytes(order)
      del zarr_imgchunk
      del imgchunk

      binary = blosc.compress(binary, **compressor_args)
      to_upload.append(
        (filename, binary)
      )

    CloudFiles(self.meta.cloudpath).puts(to_upload)

  def _chunknames(self, bbox, volume_bbox, mip, chunk_size, t):
    sep = self.meta.dimension_separator(mip)
    cf = CloudFiles(self.meta.cloudpath)
    num_channels = self.meta.num_channels

    tchunk = int(t / self.meta.num_time_chunks(mip))

    class ZarrChunkNamesIterator():
      def __len__(self):
        # round up and avoid conversion to float
        n_chunks = (bbox.dx + chunk_size[0] - 1) // chunk_size[0]
        n_chunks *= (bbox.dy + chunk_size[1] - 1) // chunk_size[1]
        n_chunks *= (bbox.dz + chunk_size[2] - 1) // chunk_size[2]
        return n_chunks
      def __iter__(self):
        nonlocal volume_bbox
        volume_bbox = Bbox.expand_to_chunk_size(volume_bbox, chunk_size)
        volume_grid = volume_bbox // chunk_size
        bbox_grid = bbox // chunk_size

        for x,y,z in xyzrange(bbox_grid.minpt, bbox_grid.maxpt):
          for c in range(num_channels):
            filename = sep.join([
              tchunk, str(c), str(z), str(y), str(x)
            ])
            yield cf.join(str(mip), filename)

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