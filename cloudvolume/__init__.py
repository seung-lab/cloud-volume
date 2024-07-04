"""
A "serverless" Python client for reading and writing arbitrarily large 
Neuroglancer Precomputed volumes both locally and on cloud services. 
Precomputed volumes consist of chunked numpy arrays, meshes, and 
skeletons and can be visualized using Neuroglancer. Typically these 
volumes represent one or three channel 3D image stacks of microscopy 
data or labels annotating them, but they can also represent large 
tensors with compatible dimensions. 

  https://github.com/seung-lab/cloud-volume

Precomputed volumes can be stored on any service that provides a 
key-value mapping between a file path and file data. Typically, 
Precomputed volumes are located on cloud storage providers such 
as Amazon S3 or Google Cloud Storage. However, these volumes can 
be stored on any service, including the local file system or an 
ordinary webserver that can process these key value mappings.

Neuroglancer is a browser based WebGL 3D image viewer principally 
authored by Jeremy Maitin-Shepard at Google. CloudVolume is a 
third-party client for reading and writing Neuroglancer compatible 
formats: 

  https://github.com/google/neuroglancer

CloudVolume is often paired with Igneous, an image processing engine 
for visualizing and managing Precomputed volumes. Igneous can be
run locally or in the cloud using Kubernetes.

  https://github.com/seung-lab/igneous

The combination of Neuroglancer, Igneous, and CloudVolume comprises 
a system for visualizing, processing, and sharing (via browser viewable 
URLs) petascale datasets within and between laboratories.

CloudVolume Example: 

  from cloudvolume import CloudVolume

  vol = CloudVolume('gs://mylab/mouse/image', progress=True)
  image = vol[:,:,:] # Download an image stack as a numpy array
  vol[:,:,:] = image # Upload an image stack from a numpy array

  label = 1
  mesh = vol.mesh.get(label) 
  skel = vol.skeletons.get(label)
"""

from .cloudvolume import CloudVolume, register_plugin

from .connectionpools import ConnectionPool
from .lib import Bbox, Vec
from .mesh import Mesh
from .provenance import DataLayerProvenance
from .storage import Storage
from .threaded_queue import ThreadedQueue
from .exceptions import (
  EmptyVolumeException, EmptyRequestException, AlignmentError,
  SkeletonEncodeError, SkeletonDecodeError
)
from .volumecutout import VolumeCutout

from .skeleton import Skeleton, PrecomputedSkeleton

from . import exceptions
from . import secrets

__version__ = '9.2.0'

# Register plugins
from .datasource.precomputed import register as register_precomputed
from .datasource.graphene import register as register_graphene
from .datasource.n5 import register as register_n5
from .datasource.zarr import register as register_zarr

register_precomputed()
register_graphene()
register_n5()
register_zarr()

try:
  from .datasource.boss import register as register_boss
  register_boss()
except ImportError:
  pass

