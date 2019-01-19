from cloudvolume import CloudVolume, CloudVolumeGraphene, CloudVolumeBoss
from .lib import extract_dataformat


def CloudFactory(cloudurl, *args, gs_replace=True, **kwargs):
    # split cloudurl into format and cloudpath
    dataformat = extract_dataformat(cloudurl)

    # switch on format to return proper MetaCloudVolume object
    if dataformat == "precomputed":
        return CloudVolume(dataformat.cloudpath, *args, **kwargs)
    elif dataformat == "graphene":
        return CloudVolumeGraphene(dataformat.cloudpath, *args, **kwargs)
    elif dataformat == "boss":
        return CloudVolumeBoss(dataformat.cloudpath, *args, **kwargs)
