import os
import shutil

from .provenance import DataLayerProvenance
from .storage import SimpleStorage, Storage

from .lib import Bbox, colorize, jsonify, mkdir, toabs, Vec, generate_slices
from .secrets import CLOUD_VOLUME_DIR

def warn(text):
  print(colorize('yellow', text))

class CacheService(object):
  
  def __init__(self, path_or_bool, vol):
    self.vol = vol
    self.enabled = path_or_bool # on/off or path (on)
    self.compress = None # None = linked, bool = force

    self.initialize()

  def initialize(self):
    if not self.enabled:
      return 

    if not os.path.exists(self.path):
        mkdir(self.path)

    if not os.access(self.path, os.R_OK|os.W_OK):
      raise IOError('Cache directory needs read/write permission: ' + self.path)

  @property
  def path(self):
    if type(self.enabled) is not str:
      path = self.vol.path
      return toabs(os.path.join(CLOUD_VOLUME_DIR, 'cache', 
        path.protocol, path.bucket.replace('/', ''), path.intermediate_path,
        path.dataset, path.layer
      ))
    else:
      return toabs(self.enabled)
  
  def num_files(self, all_mips=False):
    def size(mip):
      path_at_mip = os.path.join(self.path, self.vol.mip_key(mip))
      if not os.path.exists(path_at_mip):
        return 0
      return len(os.listdir(path_at_mip))

    if all_mips:
      sizes = [ 0 ] * (max(list(self.vol.available_mips)) + 1)
      for i in self.vol.available_mips:
        sizes[i] = size(i)
      return sizes
    else:
      return size(self.vol.mip)

  def num_bytes(self, all_mips=False):
    def mip_size(mip):
      path_at_mip = os.path.join(self.path, self.vol.mip_key(mip))
      if not os.path.exists(path_at_mip):
        return 0

      return sum( 
        ( os.path.getsize(os.path.join(path_at_mip, filename)) for filename in os.listdir(path_at_mip) ) 
      )

    if all_mips:
      sizes = [ 0 ] * (max(list(self.vol.available_mips)) + 1)
      for i in self.vol.available_mips:
        sizes[i] = mip_size(i)
      return sizes
    else:
      return mip_size(self.vol.mip)

  def list(self, mip=None):
    mip = self.vol.mip if mip is None else mip
    path = os.path.join(self.path, self.vol.mip_key(mip))

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

    for mip in self.vol.available_mips:
      preserve_mip = self.vol.slices_from_global_coords(preserve)
      preserve_mip = Bbox.from_slices(preserve_mip)

      mip_path = os.path.join(self.path, self.vol.mip_key(mip))
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
    
    if type(region) in (list, tuple):
      region = generate_slices(region, self.vol.bounds.minpt, self.vol.bounds.maxpt, bounded=False)
      region = Bbox.from_slices(region)

    mips = self.vol.mip if mips == None else mips
    if type(mips) == int:
      mips = (mips, )

    for mip in mips:
      mip_path = os.path.join(self.path, self.vol.mip_key(mip))
      if not os.path.exists(mip_path):
        continue

      region_mip = self.vol.slices_from_global_coords(region)
      region_mip = Bbox.from_slices(region_mip)

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

    fresh_info = self.vol._fetch_info()

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

    cached_prov = self.vol._cast_provenance(cached_prov)
    fresh_prov = self.vol._fetch_provenance()
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
        storage.put_file('info', jsonify(self.vol.info), 'application/json')

  def maybe_cache_provenance(self):
    if self.enabled and self.vol.provenance:
      with SimpleStorage('file://' + self.path) as storage:
        storage.put_file('provenance', self.vol.provenance.serialize(), 'application/json')

  def compute_data_locations(self, cloudpaths):
    if not self.enabled:
      return { 'local': [], 'remote': cloudpaths }

    def noextensions(fnames):
      return [ os.path.splitext(fname)[0] for fname in fnames ]

    list_dir = mkdir(os.path.join(self.vol.cache_path, self.vol.key))
    filenames = noextensions(os.listdir(list_dir))

    basepathmap = { os.path.basename(path): os.path.dirname(path) for path in cloudpaths }

    # check which files are already cached, we only want to download ones not in cache
    requested = set([ os.path.basename(path) for path in cloudpaths ])
    already_have = requested.intersection(set(filenames))
    to_download = requested.difference(already_have)

    download_paths = [ os.path.join(basepathmap[fname], fname) for fname in to_download ]    

    return { 'local': already_have, 'remote': download_paths }
    
  def __repr__(self):
    return "CacheService(enabled={}, compress={}, path='{}')".format(
      self.enabled, self.compress, self.path
    )


