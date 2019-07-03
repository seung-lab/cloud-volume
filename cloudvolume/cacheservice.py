import os
import posixpath
import shutil

from .provenance import DataLayerProvenance
from .storage import SimpleStorage, Storage, GreenStorage

from .lib import (
  Bbox, colorize, jsonify, mkdir, 
  toabs, Vec, extract_path
)
from .secrets import CLOUD_VOLUME_DIR

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
    self._path = extract_path(cloudpath)
    self.path = self.default_path()

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

  def default_path(self):
    return toabs(os.path.join(CLOUD_VOLUME_DIR, 'cache', 
      self._path.protocol, self._path.bucket.replace('/', ''), self._path.intermediate_path,
      self._path.dataset, self._path.layer
    ))
  
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
    cached_prov = self.get_json('provenance')
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
    with SimpleStorage('file://' + self.path) as storage:
      return storage.get_json(filename)

  def maybe_cache_info(self):
    if self.enabled:
      with SimpleStorage('file://' + self.path) as storage:
        storage.put_file('info', jsonify(self.meta.info), 'application/json')

  def maybe_cache_provenance(self):
    if self.enabled and self.meta.provenance:
      with SimpleStorage('file://' + self.path) as storage:
        storage.put_file('provenance', self.meta.provenance.serialize(), 'application/json')

  def upload(self, files, subdir, compress, cache_control):
    files = list(files)

    StorageClass = GreenStorage if self.config.green else Storage
    with StorageClass(self.meta.cloudpath, progress=self.config.progress) as stor:
      remote_fragments = stor.put_files(
        files=files,
        compress=compress,
        cache_control=cache_control,
      )

    if self.enabled:
      self.put(files, compress=compress)

  def download(self, paths, compress=None):
    """
    Download the provided paths, but grab them from cache first
    if they are present and the cache is enabled. 

    Returns: { filename: content, ... }
    """
    locs = self.compute_data_locations(paths)
    locs['remote'] = [ str(x) for x in locs['remote'] ]

    fragments = {}
    if self.enabled:
      fragments = self.get(locs['local'])

    StorageClass = GreenStorage if self.config.green else Storage
    with StorageClass(self.meta.cloudpath, progress=self.config.progress) as stor:
      remote_fragments = stor.get_files(locs['remote'])

    for frag in remote_fragments:
      if frag['error'] is not None:
        raise frag['error']

    remote_fragments = { 
      res['filename']: res['content'] \
      for res in remote_fragments 
    }

    if self.enabled:
      self.put(
        [ (filename, content) for filename, content in remote_fragments.items() ],
        compress=compress,
      )

    fragments.update(remote_fragments)
    return fragments

  def get(self, cloudpaths, progress=None):
    progress = self.config.progress if progress is None else progress
    StorageClass = GreenStorage if self.config.green else Storage

    with StorageClass('file://' + self.path, progress=progress) as stor:
      results = stor.get_files(
        [ filepath for filepath in cloudpaths ]
      )

    return { res['filename']: res['content'] for res in results }

  def put(self, files, progress=None, compress=None):
    """files is { filename: content }"""
    if progress is None:
      progress = self.config.progress

    if compress is None:
      compress = self.compress

    if compress is None:
      compress = self.config.compress
    
    StorageClass = GreenStorage if self.config.green else Storage

    save_location = 'file://' + self.path
    with StorageClass(save_location, progress=progress) as stor:
      stor.put_files(
        [ (name, content) for name, content in files ],
        compress=compress,
      )

  def compute_data_locations(self, cloudpaths):
    if not self.enabled:
      return { 'local': [], 'remote': cloudpaths }

    pathmodule = posixpath if self.meta.path.protocol != 'file' else os.path

    def noextensions(fnames):
      return [ pathmodule.splitext(fname)[0] for fname in fnames ]

    list_dirs = set([ pathmodule.dirname(pth) for pth in cloudpaths ])
    filenames = []

    for list_dir in list_dirs:
      list_dir = os.path.join(self.path, list_dir)
      filenames += noextensions(os.listdir(mkdir(list_dir)))

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


