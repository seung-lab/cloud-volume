from cloudvolume import CloudVolume
from cloudvolume import CloudVolumeGraphene
from .lib import extract_dataformat


def CloudVolumeFactory(cloudurl, *args, map_gs_to_https=True, **kwargs):
    # split cloudurl into format and cloudpath
    dataformat = extract_dataformat(cloudurl)

    # switch on format to return proper MetaCloudVolume object
    if dataformat.dataformat == "precomputed":
        return CloudVolume(dataformat.cloudpath, *args, **kwargs)
    elif dataformat.dataformat == "graphene":
        return CloudVolumeGraphene(dataformat.cloudpath, *args, **kwargs, map_gs_to_https=map_gs_to_https)
    # elif dataformat.dataformat == "boss":
    #     return CloudVolumeBoss(dataformat.cloudpath, *args, **kwargs)
