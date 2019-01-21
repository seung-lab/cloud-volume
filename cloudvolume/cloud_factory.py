from cloudvolume import CloudVolumePrecomputed, CloudVolumeGraphene, CloudVolumeBoss
from .lib import extract_dataformat


def CloudVolume(cloudurl, *args, gs_replace=True, **kwargs):
    # split cloudurl into format and cloudpath
    dataformat = extract_dataformat(cloudurl)

    # switch on format to return proper MetaCloudVolume object
    if dataformat.dataformat == "precomputed":
        return CloudVolume(dataformat.cloudpath, *args, **kwargs)
    elif dataformat.dataformat == "graphene":
        return CloudVolumeGraphene(dataformat.cloudpath, *args, **kwargs)
    elif dataformat.dataformat == "boss":
        return CloudVolumeBoss(dataformat.cloudpath, *args, **kwargs)
