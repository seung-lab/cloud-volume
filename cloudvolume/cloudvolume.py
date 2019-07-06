from .frontends import CloudVolumeGraphene, CloudVolumePrecomputed
from .paths import strict_extract

def CloudVolume(cloudpath, *args, **kwargs):
  path = strict_extract(cloudpath)

  print(path)




