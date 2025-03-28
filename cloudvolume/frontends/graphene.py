from typing import Optional, List, Any, Sequence

from collections import defaultdict
from datetime import datetime
import math
import orjson
import os
import pickle
import posixpath
import re
import requests
import sys

import dateutil.parser
import fastremap
import numpy as np
import tenacity

from cloudfiles import CloudFiles

from .. import chunks
from .. import compression
from .. import exceptions
from ..cacheservice import CacheService
from ..lib import Bbox, Vec, toiter, BboxLikeType, sip
from ..storage import SimpleStorage, Storage, reset_connection_pools
from ..volumecutout import VolumeCutout
from ..datasource.graphene.metadata import GrapheneApiVersion
from ..types import CompressType, MipType

from .precomputed import CloudVolumePrecomputed
from tqdm import tqdm

def warn(text):
  print(colorize('yellow', text))

def to_unix_time(timestamp):
  """
  Accepts integer UNIX timestamps, ISO 8601 datetime strings,
  and Python datetime objects and returns them as the equivalent
  UNIX timestamp or None if timestamp is None.
  """
  if timestamp is None:
    return None

  if isinstance(timestamp, str):
    timestamp = dateutil.parser.parse(timestamp) # returns datetime
  if isinstance(timestamp, datetime): # NB. do not change to elif
    timestamp = datetime.timestamp(timestamp)

  if not isinstance(timestamp, (int, float, np.integer, np.floating)) and timestamp is not None:
    raise ValueError("Not able to convert {} to UNIX time.".format(timestamp))
  
  return int(math.ceil(timestamp))

retry = tenacity.retry(
  reraise=True, 
  stop=tenacity.stop_after_attempt(7), 
  wait=tenacity.wait_random_exponential(0.5, 60.0),
)

class CloudVolumeGraphene(CloudVolumePrecomputed):

  @property
  def timestamp(self):
    return self.meta.timestamp
  
  @timestamp.setter
  def timestamp(self, val):
    self.meta.timestamp = int(val)

  @property
  def agglomerate(self):
    return self.meta.agglomerate
  
  @agglomerate.setter
  def agglomerate(self, val):
    self.meta.agglomerate = bool(val)

  @property
  def manifest_endpoint(self):
    return self.meta.manifest_endpoint

  @property
  def graph_chunk_size(self):
    return self.meta.graph_chunk_size 

  @property
  def mesh_chunk_size(self):
    # TODO: add this as new parameter to the info as it can be different from the chunkedgraph chunksize
    return self.meta.mesh_chunk_size

  def scattered_points(
    self, pts, 
    mip=None, coord_resolution=None,
    agglomerate=None, timestamp=None, stop_layer=None,
  ):
    """
    Download one or more single voxel values that may be scattered
    across the dataset. You can accelerate this query with an LRU
    if there is some spatial localization.

    If coord_resolution is not specified, pts are assumed to be specified in mip 0,
    but will request the current mip level.

    pts: iterable of triples
    mip: which resolution level to get (default self.mip)
    coord_resolution: (rx,ry,rz) the coordinate resolution of the input point.
      Sometimes Neuroglancer is working in the resolution of another
      higher res layer and this can help correct that.

    agglomerate: if true, remap all watershed ids in the volume
      and return a flat segmentation.

    if agglomerate is true these options are available:

    timestamp: (agglomerate only) get the roots from this date and time
      formats accepted:
        int: unix timestamp
        datetime: self explainatory
        string: ISO 8601 date
    stop_layer: (agglomerate only) (int) if specified, return the lowest 
      parent at or above that layer. If not specified, go all the way 
      to the root id. 
        Layer 1: Watershed
        Layer 2: Within-Chunk Agglomeration
        Layer 2+: Between chunk interconnections (skip connections possible)

    If agglomerate is None, then the cv.meta.agglomerate controls
    its value.

    Returns: 
    
      { (x,y,z): label, ... }
    """
    pts = list(pts)
    if isinstance(pts[0], int):
      pts = [ pts ]

    if mip is None:
      mip = self.mip
    mip = self.meta.to_mip(mip)

    if coord_resolution is not None:
      factor = self.meta.resolution(0) / Vec(*coord_resolution)
      pts = [ Vec(*pt) / factor for pt in pts ]

    pts = set([ tuple(self.point_to_mip(pt, mip=0, to_mip=mip)) for pt in pts ])
    results = self.image.download_points(pts, mip)

    agglomerate = agglomerate if agglomerate is not None else self.agglomerate
    timestamp = timestamp if timestamp is not None else self.timestamp
    if (agglomerate and stop_layer is not None) and (stop_layer <= 0 or stop_layer > self.meta.n_layers):
      raise ValueError(
        f"Stop layer {stop_layer} must be "
        f"1 <= stop_layer <= {self.meta.n_layers} or None."
      )

    if not agglomerate:
      return results

    labels = np.array(list(results.values()), dtype=np.uint64)
    roots = self.agglomerate_cutout(
      labels, 
      timestamp=timestamp, 
      stop_layer=stop_layer,
      in_place=False,
    )
    mapping = { segid: root for segid, root in zip(labels, roots) }
    return { pt: mapping[label] for pt, label in results.items() }

  def download_point(
    self, pt, size=256, 
    mip=None, parallel=None, 
    coord_resolution=None,
    **kwargs
  ):
    """
    Download to the right of point given in mip 0 coords.
    Useful for quickly visualizing a neuroglancer coordinate
    at an arbitary mip level.

    pt: (x,y,z)
    size: int or (sx,sy,sz)
    mip: int representing resolution level
    parallel: number of processes to launch (0 means all cores)
    coord_resolution: (rx,ry,rz) the coordinate resolution of the input point.
      Sometimes Neuroglancer is working in the resolution of another
      higher res layer and this can help correct that.

    Also accepts the arguments for download such as segids and preserve_zeros.

    Return: image as VolumeCutout(ndarray)
    """
    if isinstance(size, int) or isinstance(size, float):
      size = Vec(size, size, size)
    else:
      size = Vec(*size)

    if mip is None:
      mip = self.mip

    mip = self.meta.to_mip(mip)
    size2 = size // 2

    if coord_resolution is not None:
      factor = self.meta.resolution(0) / Vec(*coord_resolution)
      pt = Vec(*pt) / factor

    pt = self.point_to_mip(pt, mip=0, to_mip=mip)

    if all(size == 1):
      bbox = Bbox(pt, pt + 1).astype(np.int64)
    else:
      bbox = Bbox(pt - size2, pt + size2).astype(np.int64)

    if parallel is None:
      parallel = self.parallel

    return self.download(bbox, mip, parallel=parallel, **kwargs)

  def unique(
    self, 
    bbox:BboxLikeType, 
    mip:Optional[int] = None, 
    segids:Optional[List[int]] = None,
    preserve_zeros:bool = False,
    agglomerate:Optional[bool] = None, 
    timestamp:Optional[int] = None,
    stop_layer:Optional[int] = None,
    coord_resolution:Optional[Sequence[int]] = None,
  ) -> set:
    """
    Extracts unique labels from segmentation and optionally agglomerates
    labels based on information in the graph server.

    This operation can be done with download and np.unique but this
    version scales to much larger sizes and is faster.

    bbox: specifies cutout to fetch
    mip: which resolution level to get (default self.mip)
    coord_resolution: (rx,ry,rz) the coordinate resolution of the input point.
      Sometimes Neuroglancer is working in the resolution of another
      higher res layer and this can help correct that.

    agglomerate: if true, remap all watershed ids in the volume
      and return a flat segmentation.

    if agglomerate is true these options are available:

    timestamp: (agglomerate only) get the roots from this date and time
      formats accepted:
        int: unix timestamp
        datetime: self explainatory
        string: ISO 8601 date
    stop_layer: (agglomerate only) (int) if specified, return the lowest 
      parent at or above that layer. If not specified, go all the way 
      to the root id. 
        Layer 1: Watershed
        Layer 2: Within-Chunk Agglomeration
        Layer 2+: Between chunk interconnections (skip connections possible)

    If agglomerate is None, then the cv.meta.agglomerate controls
    its value.

    If agglomerate is false, these other options come into play:

    segids: agglomerate the leaves of these segids from the graph 
      server and label them with the given segid.
    preserve_zeros: If segids is not None:
      False: mask other segids with zero
      True: mask other segids with the largest integer value
        contained by the image data type and leave zero as is.

    Returns: set of integers
    """
    agglomerate = agglomerate if agglomerate is not None else self.agglomerate
    timestamp = timestamp if timestamp is not None else self.timestamp

    bbox = Bbox.create(
      bbox, context=self.bounds, 
      bounded=(self.bounded and coord_resolution is None), 
      autocrop=self.autocrop
    )
  
    if mip is None:
      mip = self.mip

    if coord_resolution is not None:
      factor = self.meta.resolution(mip) / coord_resolution
      bbox /= factor
      if self.bounded and not self.meta.bounds(mip).contains_bbox(bbox):
        raise exceptions.OutOfBoundsError(f"Computed {bbox} is not contained within bounds {self.meta.bounds(mip)}")

    if bbox.subvoxel():
      raise exceptions.EmptyRequestException("Requested {} is smaller than a voxel.".format(bbox))

    if (agglomerate and stop_layer is not None) and (stop_layer <= 0 or stop_layer > self.meta.n_layers):
      raise ValueError(
        f"Stop layer {stop_layer} must be "
        f"1 <= stop_layer <= {self.meta.n_layers} or None."
      )

    mip0_bbox = self.bbox_to_mip(bbox, mip=mip, to_mip=0)
    # Only ever necessary to make requests within the bounding box
    # to the server. We can fill black in other situations.
    mip0_bbox = bbox.intersection(self.meta.bounds(0), mip0_bbox)

    labels = super().unique(bbox, mip=mip)

    if agglomerate:
      return set(self.get_roots(
        list(labels), 
        timestamp=timestamp, 
        binary=True, 
        stop_layer=stop_layer
      ))

    labels = set(labels)
    if segids is None:
      return labels

    for segid in segids:
      leaves = set(self.get_leaves(segid, mip0_bbox, 0))
      if labels.isdisjoint(leaves):
        continue
      labels -= leaves
      labels.add(segid)

    mask_value = 0
    if preserve_zeros:
      mask_value = np.inf
      if np.issubdtype(self.dtype, np.integer):
        mask_value = np.iinfo(self.dtype).max

      segids.append(0)

    segids = set(segids)
    final_labels = set([ label for label in labels if label in segids ])
    if len(final_labels) < len(labels):
      final_labels.add(mask_value)
    return final_labels

  def download_files(
    self, bbox, mip=None, 
    parallel=None, segids=None,
    agglomerate=None, timestamp=None,
    stop_layer=None,
    coord_resolution=None,
    cache_only=False,
  ):
    agglomerate = agglomerate if agglomerate is not None else self.agglomerate
    timestamp = timestamp if timestamp is not None else self.timestamp

    bbox = Bbox.create(
      bbox, context=self.bounds, 
      bounded=(self.bounded and coord_resolution is None), 
      autocrop=self.autocrop
    )

    if mip is None:
      mip = self.mip

    if coord_resolution is not None:
      factor = self.meta.resolution(mip) / coord_resolution
      bbox /= factor
      if self.bounded and not self.meta.bounds(mip).contains_bbox(bbox):
        raise exceptions.OutOfBoundsError(f"Computed {bbox} is not contained within bounds {self.meta.bounds(mip)}")

    if bbox.subvoxel():
      raise exceptions.EmptyRequestException(f"Requested {bbox} is smaller than a voxel.")

    if (agglomerate and stop_layer is not None) and (stop_layer <= 0 or stop_layer > self.meta.n_layers):
      raise ValueError("Stop layer {} must be 1 <= stop_layer <= {} or None.".format(stop_layer, self.meta.n_layers))

    mip0_bbox = self.bbox_to_mip(bbox, mip=mip, to_mip=0)
    # Only ever necessary to make requests within the bounding box
    # to the server. We can fill black in other situations.
    mip0_bbox = bbox.intersection(self.meta.bounds(0), mip0_bbox)

    files = self.image.download_files(
      bbox, mip=mip, 
      decompress=True, parallel=parallel,
      cache_only=cache_only,
    )

    labels = set([])
    for file in files.values():
      labels.update(chunks.labels(
        file, 
        encoding=self.meta.encoding(mip),
        shape=self.meta.chunk_size(mip),
        dtype=self.meta.dtype,
        block_size=self.meta.compressed_segmentation_block_size(mip),
      ))

    def apply_mapping(mapping):
      for key in files:
        files[key] = chunks.remap(
          files[key], 
          encoding=self.meta.encoding(mip), 
          shape=self.meta.chunk_size(mip),
          dtype=self.meta.dtype,
          block_size=self.meta.compressed_segmentation_block_size(mip),
          mapping=mapping,
          preserve_missing_labels=True,
        )
      return files

    if agglomerate:
      labels = list(labels)
      roots = self.get_roots(labels, timestamp=timestamp, binary=True, stop_layer=stop_layer)
      return apply_mapping({ segid: root for segid, root in zip(labels, roots) })
    elif segids is None:
      return files

    segids = list(toiter(segids))

    remapping = {}
    for segid in segids:
      leaves = self.get_leaves(segid, mip0_bbox, 0)
      remapping.update({ leaf: segid for leaf in leaves })
    
    return apply_mapping(remapping)

  def memory_cutout(
    self, 
    bbox:BboxLikeType, 
    mip:Optional[int] = None,
    encoding:Optional[str] = None, 
    compress:CompressType = None, 
    compress_level:Optional[int] = None,
    agglomerate:bool = False,
    timestamp:Optional[int] = None,
    **kwargs, # absorb graphene arguments
  ):
    """
    Create a disposable in-memory CloudVolume (mem://) containing
    the requested cutout region in the unsharded precomputed
    format. The source volume may be sharded or unsharded.

    You can specify an alternative encoding and compression 
    settings for the new volume.
    """
    agglomerate = agglomerate if agglomerate is not None else self.agglomerate
    timestamp = timestamp if timestamp is not None else self.timestamp

    if mip is None:
      mip = self.config.mip

    mem_cv = super().memory_cutout(
      bbox,
      mip=mip,
      encoding=encoding, 
      compress=compress,
      compress_level=compress_level,
    )

    if not agglomerate:
      return mem_cv

    labels = list(mem_cv.unique(bbox))
    labels.sort()
    mapping = self.get_roots(labels, timestamp=timestamp)
    mapping = { k:v for k,v in zip(labels, mapping) }
    del labels

    cf = CloudFiles(mem_cv.cloudpath)
    for filename in cf.list(prefix=mem_cv.key):
      binary = cf.get(filename)
      binary = chunks.remap(
        binary, 
        encoding=mem_cv.meta.encoding(mip), 
        shape=mem_cv.meta.chunk_size(mip),
        dtype=mem_cv.meta.dtype,
        block_size=mem_cv.meta.compressed_segmentation_block_size(mip),
        mapping=mapping,
        preserve_missing_labels=True,
      )
      cf.put(
        filename, binary, 
        compress=compress, 
        compression_level=compress_level
      )

    return mem_cv

  def download(
    self, 
    bbox:BboxLikeType, 
    mip:MipType = None, 
    parallel:Optional[int] = None,
    segids:Optional[Sequence[int]] = None, 
    preserve_zeros:bool = False,

    agglomerate:Optional[bool] = None, 
    timestamp:Optional[int] = None, 
    stop_layer:Optional[int] = None,

    renumber:bool = False, 
    coord_resolution:Optional[Sequence[int]] = None,
    label:Optional[int] = None,
  ):
    """
    Downloads base segmentation and optionally agglomerates
    labels based on information in the graph server.

    bbox: specifies cutout to fetch
    mip: which resolution level to get (default self.mip)
    parallel: what parallel level to use (default self.parallel)
    coord_resolution: (rx,ry,rz) the coordinate resolution of the input point.
      Sometimes Neuroglancer is working in the resolution of another
      higher res layer and this can help correct that.

    agglomerate: if true, remap all watershed ids in the volume
      and return a flat segmentation.

    if agglomerate is true these options are available:

    timestamp: (agglomerate only) get the roots from this date and time
      formats accepted:
        int: unix timestamp
        datetime: self explainatory
        string: ISO 8601 date
    stop_layer: (agglomerate only) (int) if specified, return the lowest 
      parent at or above that layer. If not specified, go all the way 
      to the root id. 
        Layer 1: Watershed
        Layer 2: Within-Chunk Agglomeration
        Layer 2+: Between chunk interconnections (skip connections possible)

    If agglomerate is None, then the cv.meta.agglomerate controls
    its value.

    If agglomerate is false, these other options come into play:

    segids: agglomerate the leaves of these segids from the graph 
      server and label them with the given segid.
    preserve_zeros: If segids is not None:
      False: mask other segids with zero
      True: mask other segids with the largest integer value
        contained by the image data type and leave zero as is.
    label: similar to segids, but for compatibility with Precomputed
      decodes a to binary image.

    Returns: img as a VolumeCutout
    """
    agglomerate = agglomerate if agglomerate is not None else self.agglomerate
    timestamp = timestamp if timestamp is not None else self.timestamp
    
    if mip is None:
      mip = self.mip
    mip = self.meta.to_mip(mip)

    if isinstance(bbox, Bbox):
      bbox = bbox.convert_units(
        "vx", self.meta.resolution(mip)
      ).astype(int)

    bbox = Bbox.create(
      bbox, context=self.bounds, 
      bounded=(self.bounded and coord_resolution is None), 
      autocrop=self.autocrop
    )

    if coord_resolution is not None:
      factor = self.meta.resolution(mip) / coord_resolution
      bbox /= factor
      if self.bounded and not self.meta.bounds(mip).contains_bbox(bbox):
        raise exceptions.OutOfBoundsError(f"Computed {bbox} is not contained within bounds {self.meta.bounds(mip)}")

    if bbox.subvoxel():
      raise exceptions.EmptyRequestException("Requested {} is smaller than a voxel.".format(bbox))

    if (agglomerate and stop_layer is not None) and (stop_layer <= 0 or stop_layer > self.meta.n_layers):
      raise ValueError("Stop layer {} must be 1 <= stop_layer <= {} or None.".format(stop_layer, self.meta.n_layers))

    mip0_bbox = self.bbox_to_mip(bbox, mip=mip, to_mip=0)
    # Only ever necessary to make requests within the bounding box
    # to the server. We can fill black in other situations.
    mip0_bbox = bbox.intersection(self.meta.bounds(0), mip0_bbox)

    renumber_return = renumber
    if renumber and (segids or agglomerate):
      renumber = False # no point      

    direct_binary_image = not (agglomerate or segids or label is None or preserve_zeros)

    img = super(CloudVolumeGraphene, self).download(
      bbox, 
      mip=mip, 
      parallel=parallel, 
      renumber=renumber,
      label=(label if direct_binary_image else None)
    )
    if direct_binary_image:
      if renumber_return:
        return img, { 0:0, label:1 }
      else:
        return img

    renumber_remap = None
    if renumber:
      img, renumber_remap = img

    if agglomerate:
      img = self.agglomerate_cutout(
        img, 
        timestamp=timestamp, 
        stop_layer=stop_layer,
        label=label,
        bbox=mip0_bbox,
      )
      img = VolumeCutout.from_volume(self.meta, mip, img, bbox)
      if label is not None and not preserve_zeros:
        return img

    if segids is None or agglomerate:
      if renumber_return: 
        return img, renumber_remap
      return img

    segids = list(toiter(segids))

    remapping = {}
    for segid in segids:
      leaves = self.get_leaves(segid, mip0_bbox, 0)
      remapping.update({ leaf: segid for leaf in leaves })
    
    # Issue #434: Do not write img = fastremap.FN(in_place=True) as this allows
    # the underlying buffer to get garbage collected. Make sure to carefully
    # manage the buffer's references when making any changes.
    fastremap.remap(img, remapping, preserve_missing_labels=True, in_place=True)

    mask_value = 0
    if preserve_zeros:
      mask_value = np.inf
      if np.issubdtype(self.dtype, np.integer):
        mask_value = np.iinfo(self.dtype).max

      segids.append(0)

    fastremap.mask_except(img, segids, in_place=True, value=mask_value)
    if renumber_return:
      return img, renumber_remap
    return img
  
  def agglomerate_cutout(
    self, 
    img, 
    timestamp:Optional[int] = None, 
    stop_layer:Optional[int] = None, 
    in_place:bool = True,
    label:Optional[int] = None,
    bbox:Optional[Bbox] = None,
    mip:Optional[int] = None, # mip 0 bbox
  ):
    """
    Remap a graphene volume to the indicidated layer ids (default root ids). 
    This creates a flat segmentation.
    """
    timestamp = timestamp if timestamp is not None else self.timestamp

    if np.all(img == self.image.background_color) or stop_layer == 1:
      if not in_place:
        return np.copy(img, order="F")
      else:
        return img

    labels = fastremap.unique(img)
    if labels.size and labels[0] == 0:
      labels = labels[1:]

    if label is not None:
      watershed_domains = list(self.get_leaves(label, bbox, mip=0, stop_layer=None))
      fastremap.mask_except(img, watershed_domains, in_place=True, value=0)
      del watershed_domains
      return img > 0
    else:
      roots = self.get_roots(labels, timestamp=timestamp, binary=True, stop_layer=stop_layer)
      mapping = { segid: root for segid, root in zip(labels, roots) }
      return fastremap.remap(img, mapping, preserve_missing_labels=True, in_place=in_place)

  def coordinate_indexing(self, slices):
    res = super().coordinate_indexing(slices)
    if not self.agglomerate:
      return res
    return self.agglomerate_cutout(res)

  def __getitem__(self, slices):
    if isinstance(slices, Bbox):
      slices = slices.convert_units(
        "vx", self.meta.resolution(self.mip)
      ).astype(int).to_slices()
    elif (
      hasattr(slices, "__len__") 
      and len(slices) == 3
      and all([ isinstance(slc, (list, tuple, np.ndarray)) for slc in slices ])
    ):
      return self.coordinate_indexing(slices)

    return self.download(
      slices, mip=self.mip,
      preserve_zeros=True,
      parallel=self.parallel, 
      agglomerate=self.agglomerate,
    )

  def get_chunk_layer(self, node_or_chunk_id):
    """
    Extract Layer from Node ID or Chunk ID
    
    Returns: (int) layer number
    """
    return int(self.meta.decode_layer_id(node_or_chunk_id))

  def get_root(self, segid, *args, **kwargs):
    """Deprecated. Get a single root id for a single segid."""
    return self.get_roots(segid, *args, **kwargs)[0]

  def get_roots(self, segids, timestamp=None, binary=True, stop_layer=None):
    """
    Get the root ids for these labels.

    segids: (int or iterable) one or more segids to remap
    timestamp: get the roots from this date and time
      formats accepted:
        int: unix timestamp
        datetime: self explainatory
        string: ISO 8601 date
    binary: if true, send and receive segids as a 
      binary stream else, use JSON. The difference can
      be a 2x difference in bandwidth used.
    stop_layer: (int) if specified, return the lowest parent at or above 
      that layer. If not specified, go all the way to the root id. 
        Layer 1: Watershed
        Layer 2: Within-Chunk Agglomeration
        Layer 2+: Between chunk interconnections (skip connections possible)
    """
    timestamp = timestamp if timestamp is not None else self.timestamp

    segids = toiter(segids)
    input_segids = np.fromiter(segids, dtype=self.meta.dtype)

    if input_segids.size == 0:
      return np.array([], dtype=self.meta.dtype)

    segids = fastremap.unique(input_segids)

    base_remap = { 0: 0 }
    # skip ids that are already root IDs
    for segid in segids:
      layer_id = self.meta.decode_layer_id(segid)
      if layer_id in (stop_layer, self.meta.n_layers):
        base_remap[segid] = segid

    segids = np.array(
      [ segid for segid in segids if segid not in base_remap ], 
      dtype=self.meta.dtype
    )

    timestamp = to_unix_time(timestamp)

    if stop_layer is not None:
      stop_layer = int(stop_layer)
      if stop_layer < 1 or stop_layer > self.meta.n_layers:
        raise ValueError("stop_layer ({}) must be between 1 and {} inclusive.".format(
          stop_layer, self.meta.n_layers
        ))

    if self.meta.supports_api('v1'):
      roots = self._get_roots_v1(segids, timestamp, binary, stop_layer)
    elif self.meta.supports_api('1.0'):
      roots = self._get_roots_legacy(segids, timestamp)
    else:
      raise exceptions.UnsupportedGrapheneAPIVersionError(
        "{} is not a supported API version. Supported versions: ".format(self.meta.api_version) \
        + ", ".join([ str(_) for _ in self.meta.supported_api_versions ])
      )

    for segid, root_id in zip(segids, roots):
      base_remap[segid] = root_id

    return fastremap.remap(input_segids, base_remap)

  def get_chunk_mappings(self, chunk_id, timestamp=None):
    """
    Get the mapping of segments in a chunk at a given chunk graph layer 
    to their L1 watershed components.

    NOTE: Only L2 chunks are supported at this time.

    Required:
      chunk_id: uint64 chunk id (ie. an graphene label with a zeroed segid component)
        NOTE: This function actually accepts any graphene label and automatically converts
        it to a chunk ID before querying the graph server by zeroing out its segid component.
    Optional:
      timestamp: query the state of the graph server at the time point specified
        by a UNIX timestamp, ISO 8601 datetime string, or a python datetime object.

    Returns: {  
      chunk_label: [ watershed labels ],
      ... e.g. ...
      173729460028178433: [79450023430979610, 79450023431072298, ... ]
    }
    """
    timestamp = timestamp if timestamp is not None else self.timestamp
    timestamp = to_unix_time(timestamp)

    if not self.meta.supports_api('v1'):
      raise exceptions.UnsupportedGrapheneAPIVersionError(
        "{} is not a supported API version for range read requests. Currently, only version 1.0 is supported: ".format(self.meta.api_version) \
      )

    layer_id = self.meta.decode_layer_id(chunk_id)
    if layer_id != 2:
      raise ValueError("This function currently only accepts Layer 2 chunk IDs. Got {}".format(self.meta.decode_label(chunk_id)))

    chunk_id = self.meta.decode_chunk_id(chunk_id)
    
    version = GrapheneApiVersion('v1')
    path = version.path(self.meta.server_path)
    url = posixpath.join(self.meta.base_path, path, "l2_chunk_children_binary", str(chunk_id))

    params = {'as_array': True}
    if timestamp is not None:
      params['timestamp'] = timestamp

    response = requests.get(url, params=params, headers=self.meta.auth_header)
    response.raise_for_status()

    chunk_array = np.frombuffer(response.content, dtype=np.uint64)
    chunk_mappings = defaultdict(list)

    for i in range(0, len(chunk_array), 2):
      chunk_mappings[chunk_array[i]].append(chunk_array[i+1])

    return chunk_mappings

  def _get_roots_v1(self, segids, timestamp, binary=False, stop_layer=None):
    if len(segids) == 0:
      return []

    args = {}

    headers = {}
    headers.update(self.meta.auth_header)

    gzip_condition = len(segids) * 8 > 1e6

    if gzip_condition:
      headers['Content-Encoding'] = 'gzip'
      headers['Accept-Encoding'] = 'gzip;q=1, identity;q=0.1'
    else:
      headers['Accept-Encoding'] = 'identity'

    version = GrapheneApiVersion('v1')
    path = version.path(self.meta.server_path)

    params = {}
    if stop_layer:
      params['stop_layer'] = int(stop_layer)
    
    if timestamp is not None:
      params['timestamp'] = timestamp

    result = []

    @retry
    def _request_root_ids(subset_segids):
      nonlocal result
      if binary:
        url = posixpath.join(self.meta.base_path, path, "roots_binary")
        data = np.array(subset_segids, dtype=np.uint64).tobytes()
      else:
        url = posixpath.join(self.meta.base_path, path, "roots")
        args['node_ids'] = subset_segids
        data = orjson.dumps(args).encode('utf8')

      if gzip_condition:
        data = compression.compress(data, method='gzip')
    
      response = requests.post(url, data=data, headers=headers, params=params)
      response.raise_for_status()

      if binary:
        result.append(np.frombuffer(response.content, dtype=np.uint64))
      else:
        result.extend(orjson.loads(response.content)['root_ids'])

    for subset_segids in sip(segids, int(250000)):
      _request_root_ids(subset_segids)

    if binary:
      return np.concatenate(result)

    return result

  def _get_roots_legacy(self, segids, timestamp):
    if len(segids) == 0:
      return []

    args = {}
    if timestamp is not None:
      args['timestamp'] = timestamp

    version = GrapheneApiVersion('1.0')
    path = version.path(self.meta.server_path)
    roots = []
    for segid in segids:
      url = posixpath.join(self.meta.base_path, path, "graph/{:d}/root".format(int(segid)))
      response = requests.get(url, json=args, headers=self.meta.auth_header)
      response.raise_for_status()
      root = np.frombuffer(response.content, dtype=np.uint64)[0]
      roots.append(root)
    return roots

  def get_leaves(self, root_id, bbox, mip, stop_layer=None):
    """
    Get the lower level ids for this root_id.

    root_id: uint64 root id to find supervoxels for
    bbox: cloudvolume.lib.Bbox 3d bounding box for segmentation
    mip: which mip the bbox is defined in terms of
    stop_layer: if provided, get leaves down to the specified layer
      otherwise watershed (layer 1) is assumed.

    Returns: uint64 numpy array of leaf ids
    """
    if stop_layer is not None and (stop_layer < 1 or stop_layer > self.meta.n_layers):
      raise ValueError(f"stop_layer must be 1 <= stop_layer < {self.meta.n_layers}. Got: {stop_layer}")

    if self.meta.supports_api('v1'):
      return self.get_leaves_v1(root_id, bbox, mip, stop_layer)

    if stop_layer is not None:
      raise UnsupportedGrapheneAPIVersionError("API 1.0 does not support stop_layer.")

    return self.get_leaves_legacy(root_id, bbox, mip)

  def get_leaves_v1(self, root_id, bbox, mip, stop_layer=None):
    root_id = int(root_id)    

    api = GrapheneApiVersion("v1")

    url = posixpath.join(
      self.meta.base_path, api.path(self.meta.server_path), 
      "node", str(root_id), "leaves"
    )
    bbox = Bbox.create(bbox, context=self.meta.bounds(mip), bounded=self.bounded)

    params = { "bounds": bbox.to_filename() }
    if stop_layer is not None:
      params["stop_layer"] = int(stop_layer)

    response = requests.get(url, params=params, headers=self.meta.auth_header)
    response.raise_for_status()

    content = response.json()
    if "leaf_ids" not in content:
      return np.array([], dtype=np.uint64)

    return np.array(content["leaf_ids"], dtype=np.uint64)

  def get_leaves_legacy(self, root_id, bbox, mip):
    root_id = int(root_id)

    api = GrapheneApiVersion("1.0")

    url = posixpath.join(
      self.meta.base_path, api.path(self.meta.server_path), 
      "segment", str(root_id), "leaves"
    )
    bbox = Bbox.create(bbox, context=self.meta.bounds(mip), bounded=self.bounded)
    response = requests.post(url, json=[ root_id ], params={
      'bounds': bbox.to_filename(),
    }, headers=self.meta.auth_header)
    response.raise_for_status()

    return np.frombuffer(response.content, dtype=np.uint64)
