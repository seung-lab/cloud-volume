from cloudvolume.storage import SimpleStorage
from cloudvolume.lib import jsonify, red

class PrecomputedMetadataService(object):
  def __init__(self, cloudpath, cache, info=None):
    self.cloudpath = cloudpath
    self.cache = cache
    self.info = None

    if info is None:
      self.refresh_info()
      if self.cache.enabled:
        self.cache.check_info_validity()
    else:
      self.info = info

    if provenance is None:
      self.provenance = None
      self.refresh_provenance()
      self.cache.check_provenance_validity()
    else:
      self.provenance = self._cast_provenance(provenance)

  def refresh_info(self):
    """Restore the current info from cache or storage."""
    if self.cache.enabled:
      info = self.cache.get_json('info')
      if info:
        self.info = info
        return self.info

    self.info = self.fetch_info()
    self.cache.maybe_cache_info()
    return self.info

  def fetch_info(self):    
    with SimpleStorage(self.cloudpath) as stor:
      info = stor.get_json('info')

    if info is None:
      raise exceptions.InfoUnavailableError(
        red('No info file was found: {}'.format(self.info_cloudpath))
      )
    return info

  def commit_info(self):
    for scale in self.scales:
      if scale['encoding'] == 'compressed_segmentation':
        if 'compressed_segmentation_block_size' not in scale.keys():
          raise KeyError("""
            'compressed_segmentation_block_size' must be set if 
            compressed_segmentation is set as the encoding.

            A typical value for compressed_segmentation_block_size is (8,8,8)

            Info file specification:
            https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/README.md#info-json-file-specification
          """)
        elif self.dtype not in ('uint32', 'uint64'):
          raise ValueError("compressed_segmentation can only be used with uint32 and uint64 data types.")

    infojson = jsonify(self.info, 
      sort_keys=True,
      indent=2, 
      separators=(',', ': ')
    )

    with SimpleStorage(self.cloudpath) as stor:
      stor.put_file('info', infojson, 
        content_type='application/json', 
        cache_control='no-cache'
      )
    self.cache.maybe_cache_info()

  def refresh_provenance(self):
    if self.cache.enabled:
      prov = self.cache.get_json('provenance')
      if prov:
        self.provenance = DataLayerProvenance(**prov)
        return self.provenance

    self.provenance = self._fetch_provenance()
    self.cache.maybe_cache_provenance()
    return self.provenance

  def _cast_provenance(self, prov):
    if isinstance(prov, DataLayerProvenance):
      return prov
    elif isinstance(prov, string_types):
      prov = json.loads(prov)

    provobj = DataLayerProvenance(**prov)
    provobj.sources = provobj.sources or []  
    provobj.owners = provobj.owners or []
    provobj.processing = provobj.processing or []
    provobj.description = provobj.description or ""
    provobj.validate()
    return provobj

  def _fetch_provenance(self):
    with SimpleStorage(self.cloudpath) as stor:
      provfile = stor.get_file('provenance')
      if provfile:
        provfile = provfile.decode('utf-8')

        # The json5 decoder is *very* slow
        # so use the stricter but much faster json 
        # decoder first, and try it only if it fails.
        try:
          provfile = json.loads(provfile)
        except json.decoder.JSONDecodeError:
          try:
            provfile = json5.loads(provfile)
          except ValueError:
            raise ValueError(red("""The provenance file could not be JSON decoded. 
              Please reformat the provenance file before continuing. 
              Contents: {}""".format(provfile)))
      else:
        provfile = {
          "sources": [],
          "owners": [],
          "processing": [],
          "description": "",
        }

    return self._cast_provenance(provfile)

  def commit_provenance(self):
    prov = self.provenance.serialize()

    # hack to pretty print provenance files
    prov = json.loads(prov)
    prov = jsonify(prov, 
      sort_keys=True,
      indent=2, 
      separators=(',', ': ')
    )

    with SimpleStorage(self.cloudpath) as stor:
      stor.put_file('provenance', prov, 
        content_type='application/json',
        cache_control='no-cache',
      )
    self.cache.maybe_cache_provenance()
