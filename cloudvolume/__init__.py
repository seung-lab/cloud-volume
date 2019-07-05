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

from .connectionpools import ConnectionPool
from .lib import Bbox, Vec
from .provenance import DataLayerProvenance
from .datasource.precomputed.skeleton import (
  PrecomputedSkeleton, SkeletonEncodeError, SkeletonDecodeError
)
from .storage import Storage
from .threaded_queue import ThreadedQueue
from .exceptions import EmptyVolumeException, EmptyRequestException, AlignmentError
from .volumecutout import VolumeCutout

from .cloudvolume import CloudVolume

from . import exceptions
from . import secrets

from . import microviewer
from .microviewer import view, hyperview

__version__ = '0.52.3'
