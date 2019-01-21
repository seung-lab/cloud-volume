import weakref


from cloudvolume import CloudVolume

from .meshservice import GrapheneMeshService
from .cacheservice import CacheService
from .skeletonservice import PrecomputedSkeletonService


class CloudVolumeGraphene(CloudVolume):
    def __init__(self, *args, **kwargs):
        super(CloudVolumeGraphene, self).__init__(*args, **kwargs)

    def init_submodules(self, cache):
        """cache = path or bool"""
        self.cache = CacheService(cache, weakref.proxy(self))
        self.mesh = GrapheneMeshService(weakref.proxy(self))
        self.skeleton = PrecomputedSkeletonService(weakref.proxy(self))


