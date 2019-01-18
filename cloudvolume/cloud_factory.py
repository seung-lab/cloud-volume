from cloudvolume import CloudVolume
from cloudvolume import CloudVolumeGraphene
from .lib import extract_dataformat


def CloudFactory(cloudurl, *args, **kwargs):
    # split protocol into protocol and path
    dataformat, cloudpath = extract_dataformat(cloudurl)
    # switch on protocol to return proper MetaCloudVolume object

    if dataformat == "precomputed":
        return CloudVolume(cloudpath, *args, **kwargs)
    elif dataformat == "graphene":
        return CloudVolumeGraphene(cloudpath, *args, **kwargs)
