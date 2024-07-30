import copy
from typing import Dict, Tuple, Sequence, Union, Optional

import numpy as np
from tqdm import tqdm

from cloudfiles import CloudFiles, compression

from cloudvolume import lib, exceptions
from ....types import CompressType, MipType
from ....lib import Bbox, Vec, sip, first, BboxLikeType, toiter, xyzrange
from .... import chunks

from ... import autocropfn, readonlyguard, ImageSourceInterface
from .. import sharding
from .common import chunknames, gridpoints, compressed_morton_code, morton_code_to_bbox
from . import tx

def create_destination(source, cloudpath, mip, encoding):
  from cloudvolume import CloudVolume

  commit = False
  try:
    destvol = CloudVolume(cloudpath, mip=mip)
  except exceptions.InfoUnavailableError: 
    info = copy.deepcopy(source.meta.info)
    destvol = CloudVolume(cloudpath, mip=mip, info=info, provenance=source.meta.provenance.serialize())
    commit = True
  except exceptions.ScaleUnavailableError:
    destvol = CloudVolume(cloudpath)
    for i in range(len(destvol.scales) + 1, len(source.meta.scales)):
      destvol.scales.append(
        source.meta.scales[i]
      )
    commit = True

  if encoding is not None:
    destvol.meta.encoding(mip, encoding)
    commit = commit or (destvol.meta.encoding(mip) != source.meta.encoding(mip))

  if commit:
    destvol.commit_info()
    destvol.commit_provenance()

  return destvol

def transfer_by_rerendering(
  source,
  cloudpath:str,
  bbox:BboxLikeType,
  mip:int,
  compress:CompressType = None,
  compress_level:Optional[int] = None,
  encoding:Optional[str] = None,
):
  from cloudvolume import CloudVolume

  dest_cv = create_destination(source, cloudpath, mip, encoding)
  dest_cv.commit_info()
  dest_cv.progress = False
  dest_cv.compress = compress
  mip = dest_cv.mip

  progress = source.config.progress

  source.config.progress = False

  shape = np.array(dest_cv.chunk_size * 4)

  bbox = bbox.expand_to_chunk_size(dest_cv.chunk_size, offset=dest_cv.voxel_offset)

  grid_box = bbox / shape
  grid_size = grid_box.size()
  total = int(grid_size[0] * grid_size[1] * grid_size[2])

  for gx,gy,gz in tqdm(xyzrange(grid_size + 1), disable=(not progress), total=total):
    gpt = Vec(gx,gy,gz, dtype=int)
    bbx = Bbox(gpt * shape, (gpt+1) * shape) + bbox.minpt

    if dest_cv.meta.path.format == "precomputed":
      bbx = Bbox.clamp(bbx, dest_cv.bounds)
    dest_cv[bbx] = source.download(bbx, mip=mip)

  source.config.progress = progress

  return dest_cv

def transfer_unsharded_to_sharded(
  source,
  cloudpath:str,
  bbox:BboxLikeType, 
  mip:int,
  compress:CompressType = None, 
  compress_level:Optional[int] = None,
  encoding:Optional[str] = None,
):
  from cloudvolume import CloudVolume

  cv = create_destination(source, cloudpath, mip, encoding)
  cv.image.to_sharded(mip=mip)
  cv.commit_info()
  mip = cv.mip

  src_encoding = source.meta.encoding(mip)
  dest_encoding = cv.meta.encoding(mip)
  shard_shape = cv.image.shard_shape(mip)

  decompress = (src_encoding != dest_encoding) or (compress is not None)

  # To get the decompress info at this level will require
  # significant refactoring. Not great news for "raw" encoding.
  files = source.download_files(bbox, mip, decompress=decompress)

  spec = sharding.ShardingSpecification.from_dict(cv.scale["sharding"])
  gpts, morton_codes = cv.image.morton_codes(
    bbox, 
    mip=mip, 
    same_shard=False, 
    require_aligned=True,
  )
  for code in morton_codes:
    bbx = morton_code_to_bbox(code, cv.bounds, cv.chunk_size)
    path = cv.meta.join(cv.key, bbx.to_filename())
    files[code] = files[path]
    del files[path]

  itr = chunks.transcode(
    files,
    progress=False, 
    src_encoding=src_encoding,
    dest_encoding=dest_encoding,
    chunk_size_fn=lambda _: cv.chunk_size,
    dtype=source.meta.dtype,
    src_block_size=source.meta.compressed_segmentation_block_size(mip),
    dest_block_size=cv.meta.compressed_segmentation_block_size(mip),
    background_color=source.background_color,
  )
  for code, binary in itr:
    files[code] = binary

  shard_binaries = spec.synthesize_shards(files)
  
  basepath = cv.image.meta.join(cv.image.meta.cloudpath, cv.image.meta.key(mip))

  cf = CloudFiles(basepath, progress=cv.config.progress, secrets=cv.config.secrets)
  cf.puts(
    shard_binaries.items(), 
    compress=cv.config.compress, 
    cache_control=cv.config.cdn_cache
  )
  return cv

def transfer_any_to_unsharded(
  source,
  cloudpath:str,
  bbox:BboxLikeType, 
  mip:int,
  compress:CompressType = None, 
  compress_level:Optional[int] = None,
  encoding:Optional[str] = None,
):
  """
  You can specify an alternative encoding and compression 
  settings for the new volume.
  """
  from cloudvolume import CloudVolume

  cv = create_destination(source, cloudpath, mip, encoding)
  cv.scale.pop("sharding", None)
  cv.commit_info()
  mip = cv.mip

  src_encoding = source.meta.encoding(mip)
  dest_encoding = cv.meta.encoding(mip)

  decompress = (src_encoding != dest_encoding) or (compress is not None)

  # To get the decompress info at this level will require
  # significant refactoring. Not great news for "raw" encoding.
  files = source.download_files(bbox, mip, decompress=decompress)

  bounds = source.meta.bounds(mip)
  chunk_size = source.meta.chunk_size(mip)

  filenames = list(files.keys())

  def get_bbx(fname):
    if type(fname) == int:
      bbx = morton_code_to_bbox(fname, bounds, chunk_size)
    else:
      bbx = Bbox.from_filename(fname)
    return bbx

  itr = chunks.transcode(
    img_chunks,
    progress=False, 
    src_encoding=src_encoding,
    dest_encoding=dest_encoding,
    chunk_size_fn=lambda fname: get_bbx(fname).size(),
    dtype=source.meta.dtype,
    src_block_size=source.meta.compressed_segmentation_block_size(mip),
    dest_block_size=cv.meta.compressed_segmentation_block_size(mip),
    background_color=source.background_color,
  )
  for label, binary in itr:
    bbx = get_bbx(label)
    del files[label]
    files[bbx.to_filename()] = binary

  CloudFiles(
    cv.meta.join(cloudpath, cv.key)
  ).puts(
    files.items(), compress=compress, compression_level=compress_level
  )

  return cv

def transfer_sharded_to_sharded(
  source,
  cloudpath:str, 
  bbox:BboxLikeType, 
  mip:MipType,
  block_size:int = 2, 
  compress:CompressType = True, 
  compress_level:Optional[int] = None, 
  encoding:Optional[str] = None,
):
  from cloudvolume import CloudVolume
  if mip is None:
    mip = source.config.mip

  if not source.is_sharded(mip):
    raise exceptions.UnsupportedFormatError(f"Unsharded sources are not supported. got: {source.meta.cloudpath}")

  spec = source.shard_spec(mip)

  chunk_size = source.meta.chunk_size(mip)
  shape = spec.image_shard_shape(source.meta.volume_size(mip), chunk_size)
  bbox = Bbox.create(bbox, source.meta.bounds(mip))
  realized_bbox = bbox.expand_to_chunk_size(
    shape, offset=source.meta.voxel_offset(mip)
  )
  realized_bbox = Bbox.clamp(realized_bbox, source.meta.bounds(mip))

  grid_size = np.ceil(source.meta.bounds(mip).size3() / chunk_size).astype(np.uint32)

  reader = sharding.ShardReader(source.meta, source.cache, spec)
  bounds = source.meta.bounds(mip)
  
  gpts, morton_codes = source.morton_codes(
    bbox, mip=mip, spec=spec, 
    same_shard=False, require_aligned=False,
  )
  shard_filenames = list(set([ 
    reader.get_filename(code) for code in morton_codes 
  ]))

  destvol = create_destination(source, cloudpath, mip, encoding)

  cfsrc = CloudFiles(
    source.meta.join(source.meta.cloudpath, source.meta.key(mip)),
    secrets=source.config.secrets
  )
  cfdest = CloudFiles(destvol.meta.join(
    cloudpath, destvol.meta.key(mip)
  ))

  src_encoding = source.meta.encoding(mip)
  dest_encoding = destvol.meta.encoding(mip)

  if src_encoding == dest_encoding:
    cfsrc.transfer_to(
      cfdest, 
      paths=shard_filenames,
      block_size=block_size,
    )
  else:
    for filename in shard_filenames:
      shard_binary = cfsrc.get(filename, raw=True)
      img_chunks = reader.disassemble_shard(shard_binary)
      del shard_binary

      itr = chunks.transcode(
        img_chunks,
        progress=False, 
        src_encoding=src_encoding,
        dest_encoding=dest_encoding,
        chunk_size_fn=lambda _: chunk_size,
        dtype=source.meta.dtype,
        src_block_size=source.meta.compressed_segmentation_block_size(mip),
        dest_block_size=destvol.meta.compressed_segmentation_block_size(mip),
        background_color=source.background_color,
      )
      for label, binary in itr:
        img_chunks[label] = binary

      shard_binary = spec.synthesize_shard(img_chunks)
      del img_chunks
      cfdest.put(filename, shard_binary, raw=True)

  return destvol

def transfer_unsharded_to_unsharded(
  source, 
  cloudpath:str, 
  bbox:BboxLikeType, 
  mip:MipType, 
  block_size:Optional[int] = None, 
  compress:CompressType = True, 
  compress_level:Optional[int] = None, 
  encoding:Optional[str] = None,
):
  """
  Transfer files from one storage location to another, bypassing
  volume painting. This enables using a single CloudVolume instance
  to transfer big volumes. In some cases, gsutil or aws s3 cli tools
  may be more appropriate. This method is provided for convenience. It
  may be optimized for better performance over time as demand requires.

  cloudpath: path to storage layer
  bbox: ROI to transfer
  mip: resolution level
  block_size: number of file chunks to transfer per I/O batch.
  compress: Set to False to upload as uncompressed
  compress_level: level to feed to compressor (means different things for
    different algorithms)
  encoding: if specified, transcode to this image encoding (otherwise taken
    to be identical to source volume)
  """
  from cloudvolume import CloudVolume
  if mip is None:
    mip = source.config.mip

  if source.is_sharded(mip):
    raise exceptions.UnsupportedFormatError(f"Sharded sources are not supported. got: {source.meta.cloudpath}")

  bbox = Bbox.create(bbox, source.meta.bounds(mip))
  realized_bbox = bbox.expand_to_chunk_size(
    source.meta.chunk_size(mip), offset=source.meta.voxel_offset(mip)
  )
  realized_bbox = Bbox.clamp(realized_bbox, source.meta.bounds(mip))

  if bbox != realized_bbox:
    raise exceptions.AlignmentError(
      "Unable to transfer non-chunk aligned bounding boxes. Requested: {}, Realized: {}".format(
        bbox, realized_bbox
      ))

  default_block_size_MB = 50 # MB
  chunk_MB = source.meta.chunk_size(mip).rectVolume() * np.dtype(source.meta.dtype).itemsize * source.meta.num_channels
  if source.meta.layer_type == 'image':
    # kind of an average guess for some EM datasets, have seen up to 1.9x and as low as 1.1
    # affinites are also images, but have very different compression ratios. e.g. 3x for kempressed
    chunk_MB /= 1.3 
  else: # segmentation
    chunk_MB /= 100.0 # compression ratios between 80 and 800....
  chunk_MB /= 1024.0 * 1024.0

  if block_size:
    step = block_size
  else:
    step = int(default_block_size_MB // chunk_MB) + 1

  destvol = create_destination(source, cloudpath, mip, encoding)

  if destvol.image.is_sharded(mip):
    raise exceptions.UnsupportedFormatError(f"Sharded destinations are not supported. got: {destvol.cloudpath}")

  num_blocks = np.ceil(source.meta.bounds(mip).volume() / source.meta.chunk_size(mip).rectVolume()) / step
  num_blocks = int(np.ceil(num_blocks))

  src_encoding = source.meta.encoding(mip)
  dest_encoding = destvol.meta.encoding(mip)

  cloudpaths = chunknames(
    bbox, source.meta.bounds(mip), 
    source.meta.key(mip), source.meta.chunk_size(mip),
    protocol=source.meta.path.protocol
  )

  pbar = tqdm(
    desc='Transferring Blocks of {} Chunks'.format(step), 
    unit='blocks', 
    disable=(not source.config.progress),
    total=num_blocks,
  )

  cfsrc = CloudFiles(source.meta.cloudpath, secrets=source.config.secrets)
  cfdest = CloudFiles(cloudpath)

  def check(files):
    if source.fill_missing:
      for file in files:
        if file['content'] is None:
          file['content'] = b''
    errors = [
      file for file in files if \
      (file['content'] is None or file['error'] is not None)
    ]
    if errors:
      error_paths = [ f['path'] for f in errors ]
      raise exceptions.EmptyFileException(f"{', '.join(error_paths)} were empty or had IO errors.")
    return files

  content_type = tx.content_type(destvol)

  with pbar:
    for srcpaths in sip(cloudpaths, step):
      if src_encoding == dest_encoding:
          cfdest.transfer_from(
            cfsrc, srcpaths, 
            reencode=compress,
            content_type=content_type,
            allow_missing=source.fill_missing,
          )
      else:
        files = check(cfsrc.get(srcpaths))
        itr = chunks.transcode(
          ( (f["path"], f["content"]) for f in files ),
          progress=False, #source.config.progress,
          total=len(files),
          src_encoding=src_encoding,
          dest_encoding=dest_encoding,
          chunk_size_fn=lambda path: Bbox.from_filename(path).size(),
          dtype=source.meta.dtype,
          src_block_size=source.meta.compressed_segmentation_block_size(mip),
          dest_block_size=destvol.meta.compressed_segmentation_block_size(mip),
          background_color=source.background_color,
        )
        cfdest.puts(
          itr, 
          content_type=content_type,
          compress=compress,
          compression_level=compress_level,
        )
      pbar.update()
  
  return destvol
