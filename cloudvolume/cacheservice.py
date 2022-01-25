from functools import partial
import json
import os
import posixpath
import shutil

from . import scheduler

from .provenance import DataLayerProvenance
from cloudfiles import CloudFiles, compression
from cloudfiles.interfaces import COMPRESSION_EXTENSIONS

from .lib import (
  Bbox, colorize, jsonify, mkdir, 
  toabs, Vec, nvl
)

def warn(text):
  print(colorize('yellow', text))

class CacheService(object):
  def __init__(
    self, cloudpath,
    enabled, config,
    meta=None, compress=None
  ):
    """
    enabled: bool or path string
    config: SharedConfiguration 
    meta: PrecomputedMetadata
    compress: None = linked to dataset setting, bool = Force
    """
    self.path = cloudpath

    self.config = config
    self._enabled = enabled 
    self.compress = compress 

    # b/c there's a semi-circular dependency
    # meta is usually set afterwards
    self.meta = meta 

    self.initialize()

  def initialize(self):
    if not self.enabled:
      return 

    if not os.path.exists(self.path):
      mkdir(self.path)

    if not os.access(self.path, os.R_OK|os.W_OK):
      raise IOError('Cache directory needs read/write permission: ' + self.path)

  @property
  def enabled(self):
    return self._enabled

  @enabled.setter
  def enabled(self, val):
    self._enabled = val 
    self.initialize()
  
  def num_files(self, all_mips=False):
    def size(mip):
      path_at_mip = os.path.join(self.path, self.meta.key(mip))
      if not os.path.exists(path_at_mip):
        return 0
      return len(os.listdir(path_at_mip))

    if all_mips:
      sizes = [ 0 ] * (max(list(self.meta.available_mips)) + 1)
      for i in self.meta.available_mips:
        sizes[i] = size(i)
      return sizes
    else:
      return size(self.config.mip)

  def num_bytes(self, all_mips=False):
    def mip_size(mip):
      path_at_mip = os.path.join(self.path, self.meta.key(mip))
      if not os.path.exists(path_at_mip):
        return 0

      return sum( 
        ( os.path.getsize(os.path.join(path_at_mip, filename)) for filename in os.listdir(path_at_mip) ) 
      )

    if all_mips:
      sizes = [ 0 ] * (max(list(self.meta.available_mips)) + 1)
      for i in self.meta.available_mips:
        sizes[i] = mip_size(i)
      return sizes
    else:
      return mip_size(self.config.mip)

  def has(self, filename):
    return CloudFiles('file://' + self.path).exists(filename)
    
  def list(self, mip=None):
    mip = self.config.mip if mip is None else mip

    path = os.path.join(self.path, self.meta.key(mip))

    if not os.path.exists(path):
      return []

    return os.listdir(path)

  def list_skeletons(self):
    if self.meta.skeletons is None:
      return []

    path = os.path.join(self.path, self.meta.skeletons)
    if not os.path.exists(path):
      return []

    return os.listdir(path)

  def list_meshes(self):
    if self.meta.mesh is None:
      return []

    path = os.path.join(self.path, self.meta.mesh)
    if not os.path.exists(path):
      return []

    return os.listdir(path)

  def flush_info(self):
    path = os.path.join(self.path , 'info')
    if not os.path.exists(path):
      return
    os.remove(path)

  def flush_provenance(self):
    path = os.path.join(self.path , 'provenance')
    if not os.path.exists(path):
      return
    os.remove(path)

  def flush(self, preserve=None):
    """
    Delete the cache for this dataset. Optionally preserve
    a region. Helpful when working with overlaping volumes.

    Warning: the preserve option is not multi-process safe.
    You're liable to end up deleting the entire cache.

    Optional:
      preserve (Bbox: None): Preserve chunks located partially
        or entirely within this bounding box. 
    
    Return: void
    """
    if not os.path.exists(self.path):
      return

    if preserve is None:
      shutil.rmtree(self.path)
      return

    for mip in self.meta.available_mips:
      preserve_mip = self.meta.bbox_to_mip(preserve, 0, mip)
      mip_path = os.path.join(self.path, self.meta.key(mip))

      if not os.path.exists(mip_path):
        continue

      for filename in os.listdir(mip_path):
        bbox = Bbox.from_filename(filename)
        if not Bbox.intersects(preserve_mip, bbox):
          os.remove(os.path.join(mip_path, filename))

  # flush_cache_region seems like it could be tacked on
  # as a flag to delete, but there are reasons not
  # to do that. 
  # 1) reduces the risks of disasterous programming errors. 
  # 2) doesn't require chunk alignment
  # 3) processes potentially multiple mips at once

  def flush_region(self, region, mips=None):
    """
    Delete a cache region at one or more mip levels 
    bounded by a Bbox for this dataset. Bbox coordinates
    should be specified in mip 0 coordinates.

    Required:
      region (Bbox): Delete cached chunks located partially
        or entirely within this bounding box. 
    Optional:
      mip (int: None): Flush the cache from this mip. Region
        is in global coordinates.
    
    Return: void
    """
    if not os.path.exists(self.path):
      return
  
    cur_mip = self.config.mip

    region = Bbox.create(region, self.meta.bounds(cur_mip))
    mips = ( cur_mip, ) if mips == None else mips

    for mip in mips:
      mip_path = os.path.join(self.path, self.meta.key(mip))
      if not os.path.exists(mip_path):
        continue

      region_mip = self.meta.bbox_to_mip(region, mip=0, to_mip=mip)
      for filename in os.listdir(mip_path):
        bbox = Bbox.from_filename(filename)
        if not Bbox.intersects(region, bbox):
          os.remove(os.path.join(mip_path, filename))

  def check_info_validity(self):
    """
    ValueError if cache differs at all from source data layer with
    an excepton for volume_size which prints a warning.
    """
    cache_info = self.get_json('info')
    if not cache_info:
      return
    fresh_info = self.meta.fetch_info()

    mismatch_error = ValueError("""
      Data layer info file differs from cache. Please check whether this
      change invalidates your cache. 

      If VALID do one of:
      1) Manually delete the cache (see location below)
      2) Refresh your on-disk cache as follows:
        vol = CloudVolume(..., cache=False) # refreshes from source
        vol.cache = True
        vol.commit_info() # writes to disk
      If INVALID do one of: 
      1) Delete the cache manually (see cache location below) 
      2) Instantiate as follows: 
        vol = CloudVolume(..., cache=False) # refreshes info from source
        vol.flush_cache() # deletes cache
        vol.cache = True
        vol.commit_info() # writes info to disk

      CACHED: {cache}
      SOURCE: {source}
      CACHE LOCATION: {path}
      """.format(
      cache=cache_info, 
      source=fresh_info, 
      path=self.path
    ))

    try:
      fresh_sizes = [ scale['size'] for scale in fresh_info['scales'] ]
      cache_sizes = [ scale['size'] for scale in cache_info['scales'] ]
    except KeyError:
      raise mismatch_error

    for scale in fresh_info['scales']:
      del scale['size']

    for scale in cache_info['scales']:
      del scale['size']

    if fresh_info != cache_info:
      raise mismatch_error

    if fresh_sizes != cache_sizes:
      warn("WARNING: Data layer bounding box differs in cache.\nCACHED: {}\nSOURCE: {}\nCACHE LOCATION:{}".format(
      cache_sizes, fresh_sizes, self.path
      ))

  def check_provenance_validity(self):
    try:
      cached_prov = self.get_json('provenance')
    except json.decoder.JSONDecodeError:
      warn("Cached provenance file is not valid JSON.")
      return

    if not cached_prov:
      return

    cached_prov = self.meta._cast_provenance(cached_prov)
    fresh_prov = self.meta.fetch_provenance()
    if cached_prov != fresh_prov:
      warn("""
      WARNING: Cached provenance file does not match source.

      CACHED: {}
      SOURCE: {}
      """.format(cached_prov.serialize(), fresh_prov.serialize()))

  def get_json(self, filename):
    return CloudFiles('file://' + self.path).get_json(filename)

  def maybe_cache_info(self):
    if self.enabled:
      cf = CloudFiles('file://' + self.path)
      cf.put_json('info', self.meta.info)

  def maybe_cache_provenance(self):
    if self.enabled and self.meta.provenance:
      cf = CloudFiles('file://' + self.path)
      cf.put('provenance', self.meta.provenance.serialize().encode('utf8'))

  def upload_single(self, filename, content, *args, **kwargs):
    kwargs['progress'] = False
    return self.upload( [(filename, content)], *args, **kwargs )

  def upload(self, files, compress, cache_control, compress_level=None, content_type=None, progress=None):
    files = list(files)

    progress = progress if progress is not None else self.config.progress

    cf = CloudFiles(self.meta.cloudpath, progress=progress, secrets=self.config.secrets)
    files = list(compression.transcode(files, encoding=compress, level=compress_level))
    cf.puts( 
      files, 
      compress=compress, 
      compression_level=compress_level, 
      cache_control=cache_control, 
      content_type=content_type,
      raw=True,
    )

    if self.enabled:
      self.put(files, compress=compress)
      cf_cache = CloudFiles('file://' + self.path, progress=('to Cache' if progress else None))
      cf_cache.puts(
        files,
        compress=compress,
        raw=True
      )

  def download_json(self, path, compress=None):
    """
    Download a single path, but grab from 
    cache first if present and cache is enabled.

    Returns: content or None
    """
    res = self.download( [ path ], compress=compress, progress=False )
    res = res[path]
    if res is None:
      return None    
    return json.loads(res.decode('utf8'))

  def download_single(self, path, compress=None):
    files = self.download([ path ], compress=compress, progress=False)
    return files[path]

  def download_single_as(
    self, path, local_alias, 
    compress=None, start=None, end=None
  ):
    """
    Download a file or a byte range from a file 
    and save it locally as `local_alias`.
    """
    if self.enabled:
      locs = self.compute_data_locations([local_alias])
      if locs['local']:
        return self.get_single(local_alias)

    filedata = CloudFiles(self.meta.cloudpath, secrets=self.config.secrets)[path, start:end]

    if self.enabled:
      self.put([ (local_alias, filedata) ], compress=compress)

    return filedata

  def download_as(self, requests, compress=None, progress=None):
    """
    Works with byte ranges.

    requests: [{
      'path': ...,
      'local_alias': ...,
      'start': ...,
      'end': ...,
    }]
    """
    if len(requests) == 0:
      return {}

    progress = progress if progress is not None else self.config.progress

    aliases = [ req['local_alias'] for req in requests ]
    alias_tuples = { req['local_alias']: (req['path'], req['start'], req['end']) for req in requests }
    alias_to_path = { req['local_alias']: req['path'] for req in requests }
    path_to_alias = { v:k for k,v in alias_tuples.items() }

    if None in alias_to_path:
      del alias_to_path[None]
      del alias_tuples[None]

    locs = self.compute_data_locations(aliases)

    fragments = {}
    if self.enabled:
      fragments = self.get(locs['local'], progress=progress)
      keys = list(fragments.keys())
      for key in keys:
        return_key = (alias_to_path[key], alias_tuples[key][1], alias_tuples[key][2])
        fragments[return_key] = fragments[key]
        del fragments[key]
      for alias in locs['local']:
        del alias_tuples[alias]

    remote_path_tuples = list(alias_tuples.values())

    cf = CloudFiles(
      self.meta.cloudpath, 
      progress=progress, 
      secrets=self.config.secrets,
      parallel=self.config.parallel,
    )
    remote_fragments = cf.get(
      ( { 'path': p[0], 'start': p[1], 'end': p[2] } for p in remote_path_tuples ),
      total=len(remote_path_tuples),
    )

    for frag in remote_fragments:
      if frag['error'] is not None:
        raise frag['error']

    remote_fragments = { 
      (res['path'], res['byte_range'][0], res['byte_range'][1]): res['content'] \
      for res in remote_fragments 
    }

    if self.enabled:
      self.put(
        [ 
          (path_to_alias[file_bytes_tuple], content) \
          for file_bytes_tuple, content in remote_fragments.items() \
          if content is not None 
        ],
        compress=compress,
        progress=progress
      )

    fragments.update(remote_fragments)
    return fragments

  def download(self, paths, compress=None, progress=None):
    """
    Download the provided paths, but grab them from cache first
    if they are present and the cache is enabled. 

    Returns: { filename: content, ... }
    """
    if len(paths) == 0:
      return {}

    progress = nvl(progress, self.config.progress)
    compress = nvl(compress, self.compress, self.config.compress)

    locs = self.compute_data_locations(paths)
    locs['remote'] = [ str(x) for x in locs['remote'] ]

    fragments = {}
    if self.enabled:
      fragments = self.get(locs['local'], progress=progress)

    # fixes e.g. mesh\info -> mesh/info on Windows
    if self.meta.path.protocol != 'file' and os.path.sep == '\\':
      fragments = { 
        "/".join(key.split('\\')): val 
        for key,val in fragments.items() 
      }

    cf = CloudFiles(
      self.meta.cloudpath, 
      progress=progress, 
      secrets=self.config.secrets,
      parallel=self.config.parallel,
    )
    remote_fragments = cf.get(locs['remote'], raw=True)

    for frag in remote_fragments:
      if frag['error'] is not None:
        raise frag['error']

    if self.enabled:
      cf_cache = CloudFiles('file://' + self.path, progress=('to Cache' if progress else None))
      cf_cache.puts(
        compression.transcode(
          ( frag for frag in remote_fragments if frag['content'] is not None ),
          encoding=compress, progress=progress, in_place=False
        ),
        compress=compress,
        raw=True
      )

    remote_fragments_dict = {}
    while remote_fragments:
      res = remote_fragments.pop()
      remote_fragments_dict[res['path']] = compression.decompress(res['content'], res['compress'])

    fragments.update(remote_fragments_dict)
    return fragments

  def get_single(self, cloudpath, progress=None):
    res = self.get([ cloudpath ], progress=progress)
    return res[cloudpath]

  def get(self, cloudpaths, progress=None):
    progress = self.config.progress if progress is None else progress
    
    cf = CloudFiles('file://' + self.path, progress=progress)
    results = cf.get(list(cloudpaths))

    return { res['path']: res['content'] for res in results }

  def put_single(self, path, content, *args, **kwargs):
    kwargs['progress'] = False
    return self.put([ (path, content) ], *args, **kwargs)

  def put_json(self, path, content, *args, **kwargs):
    content = jsonify(content).encode("utf8")
    return self.put_single(path, content, *args, *kwargs)

  def put(self, files, progress=None, compress=None, compress_level=None):
    """files: [ (filename, content) ]"""
    if progress is None:
      progress = self.config.progress

    if compress is None:
      compress = self.compress

    if compress is None:
      compress = self.config.compress
    
    save_location = 'file://' + self.path
    progress = 'to Cache' if progress else None
    cf = CloudFiles(save_location, progress=progress)
    cf.puts(
      files, 
      compress=compress, 
      compression_level=compress_level,
      parallel=self.config.parallel,
    )

  def compute_data_locations(self, cloudpaths):
    if not self.enabled:
      return { 'local': [], 'remote': cloudpaths }

    pathmodule = posixpath if self.meta.path.protocol != 'file' else os.path

    def no_compression_ext(fnames):
      results = []
      for fname in fnames:
        (name, ext) = pathmodule.splitext(fname)
        if ext in COMPRESSION_EXTENSIONS:
          results.append(name)
        else:
          results.append(fname)
      return results

    list_dirs = set([ pathmodule.dirname(pth) for pth in cloudpaths ])
    filenames = []

    for list_dir in list_dirs:
      list_dir = os.path.join(self.path, list_dir)
      filenames += no_compression_ext(os.listdir(mkdir(list_dir)))

    basepathmap = { pathmodule.basename(path): pathmodule.dirname(path) for path in cloudpaths }

    # check which files are already cached, we only want to download ones not in cache
    requested = set([ pathmodule.basename(path) for path in cloudpaths ])
    already_have = requested.intersection(set(filenames))
    to_download = requested.difference(already_have)

    download_paths = [ pathmodule.join(basepathmap[fname], fname) for fname in to_download ]    
    already_have = [ os.path.join(basepathmap[fname], fname) for fname in already_have ]

    return { 'local': already_have, 'remote': download_paths }

  def __repr__(self):
    return "CacheService(enabled={}, compress={}, path='{}')".format(
      self.enabled, self.compress, self.path
    )
