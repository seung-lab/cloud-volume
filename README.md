[![Build Status](https://travis-ci.org/seung-lab/cloud-volume.svg?branch=master)](https://travis-ci.org/seung-lab/cloud-volume) [![PyPI version](https://badge.fury.io/py/cloud-volume.svg)](https://badge.fury.io/py/cloud-volume) [![SfN 2018 Poster](https://img.shields.io/badge/poster-SfN%202018-blue.svg)](https://drive.google.com/open?id=1RKtaAGV2f7F13opnkQfbp6YBqmoD3fZi)

# CloudVolume

```python3
from cloudvolume import CloudVolume

vol = CloudVolume('gs://mylab/mouse/image', parallel=True, progress=True)
image = vol[:,:,:] # Download a whole image stack into a numpy array from the cloud
vol[:,:,:] = image # Upload an entire image stack from a numpy array to the cloud
```


CloudVolume is a Python library for reading and writing chunked numpy arrays from [Neuroglancer](https://github.com/google/neuroglancer/) volumes in "[Precomputed](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed)" format, a simple hackable representation for arbitrarily large volumetric images. CloudVolume is typically paired with [Igneous](https://github.com/seung-lab/igneous), a Kubernetes based system for generating image hierarchies, meshes, and other dependency free tasks that might be applied to petavoxel scale images.

Precomputed volumes are typically stored on [AWS S3](https://aws.amazon.com/s3/) or on [Google Storage](https://cloud.google.com/storage/). CloudVolume can read and write to these object storage providers given a service account token with appropriate permissions. However, these volumes can be stored on any service, including an ordinary webserver or local filesystem, that supports hierarchical file system paths (or simulates them via path strings).

The combination of [Neuroglancer](https://github.com/google/neuroglancer/), [Igneous](https://github.com/seung-lab/igneous), and CloudVolume comprises a system for visualizing, processing, and sharing (via browser viewable URLs) petascale datasets within and between laboratories. A typical example usage would be to visualize raw electron microscope scans of mouse, fish, or fly brains up to a cubic millimeter in physical dimension. Neuroglancer and Igneous would enable you to visualize each step of the process of montaging the image, fine tuning alignment, creating segmentation layers, ROI masks, or performing other types of analysis. CloudVolume enables you to read from and write to each of these layers.

CloudVolume can be used in single or multi-process capacity and can be optimized to use no more than a little over a single cutout's worth of memory. It supports reading and writing the `compressed_segmentation` format via a C++ extension by Jeremy Maitin-Shepard, Stephen Plaza, and William Silversmith and a fallback to a pure python library provided by Yann Leprince.  

## Setup

Cloud-volume is regularly tested on Ubuntu with Python 2.7, 3.4, 3.5, and 3.6 (we've noticed it's faster on Python 3). Some people have used it with Python 3.7. We support Linux and OS X. Windows is currently unsupported. After installation, you'll also need to set up your cloud credentials.  

#### `pip` Binary Installation

```bash
pip install cloud-volume
```

CloudVolume depends on the PyPI packages [`fpzip`](https://github.com/seung-lab/fpzip) and [`compressed_segmentation`](https://github.com/seung-lab/compressedseg), which are Cython bindings for C++. We have provided compiled binaries for many platforms and python versions, however if you are on an unsupported system, pip will attempt to install from source. In that case, follow the instructions below.

#### `pip` Source Installation

*C++ compiler required.*

```bash
sudo apt-get install g++ python3-dev # python-dev if you're on python2
pip install numpy
pip install cloud-volume
```

Due to packaging problems endemic to Python, Cython packages that depend on numpy require numpy header files be installed before attempting to install the package you want. The numpy headers are not recognized unless numpy is installed in a seperate process that runs first. There are hacks for this issue, but I haven't gotten them to work. If you think binaries should be available for your platform, please let us know by opening an issue.

The libraries depending on numpy are:  
- compressed_segmentation: Smaller and faster segmentation files. A pure python fallback is present. When the accelerated version is present, IO is faster than with gzip alone.
- fpzip: A lossless compression library for 3D & 4D floating point data.

#### Manual Installation

This can be desirable if you want to hack on CloudVolume itself.  

```bash
git clone git@github.com:seung-lab/cloud-volume.git
cd cloud-volume

# With virtualenvwrapper
mkvirtualenv cv
workon cv
# With only virtualenv
virtualenv venv
source venv/bin/activate

sudo apt-get install g++ python3-dev # python-dev if you're on python2
pip install numpy # additional step needed for accelerated compressed_segmentation and fpzip
pip install -e .
```

### Credentials

You'll need credentials only for the services you'll use. If you plan to use the local filesystem, you won't need any. For Google Storage ([setup instructions here](https://github.com/seung-lab/cloud-volume/wiki/Setting-up-Google-Cloud-Storage)), default account credentials will be used if available and no service account is provided. 

If neither of those two conditions apply, you need a service account credential. `google-secret.json` is a service account credential for Google Storage, `aws-secret.json` is a service account for S3, etc. You can support multiple projects at once by prefixing the bucket you are planning to access to the credential filename. `google-secret.json` will be your defaut service account, but if you also want to also access bucket ABC, you can provide `ABC-google-secret.json` and you'll have simultaneous access to your ordinary buckets and ABC. The secondary credentials are accessed on the basis of the bucket name, not the project name.

```bash
mkdir -p ~/.cloudvolume/secrets/
mv aws-secret.json ~/.cloudvolume/secrets/ # needed for Amazon
mv google-secret.json ~/.cloudvolume/secrets/ # needed for Google
mv boss-secret.json ~/.cloudvolume/secrets/ # needed for the BOSS
mv matrix-secret.json ~/.cloudvolume/secrets/ # needed for Matrix
```

#### `aws-secret.json` and `matrix-secret.json`

Create an [IAM user service account](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users.html) that can read, write, and delete objects from at least one bucket.

```json
{
	"AWS_ACCESS_KEY_ID": "$MY_AWS_ACCESS_KEY_ID",
	"AWS_SECRET_ACCESS_KEY": "$MY_SECRET_ACCESS_TOKEN"
}
```

#### `google-secret.json`

You can create the `google-secret.json` file [here](https://console.cloud.google.com/iam-admin/serviceaccounts). You don't need to manually fill in JSON by hand, the below example is provided to show you what the end result should look like. You should be able to read, write, and delete objects from at least one bucket.

```json
{
  "type": "service_account",
  "project_id": "$YOUR_GOOGLE_PROJECT_ID",
  "private_key_id": "...",
  "private_key": "...",
  "client_email": "...",
  "client_id": "...",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://accounts.google.com/o/oauth2/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": ""
}
```

## Usage

CloudVolume supports reading and writing to Neuroglancer data layers on Amazon S3, Google Storage, The BOSS, and the local file system.

Supported URLs are of the forms:

`$PROTOCOL://$BUCKET/$DATASET/$LAYER`

### Supported Protocols 
* gs:   Google Storage
* s3:   Amazon S3
* boss: The BOSS (https://docs.theboss.io/docs)
* http(s): (read-only) Ordinary Web Servers 
* file: Local File System (absolute path)
* matrix: Princeton Internal System

### `info` Files - New Dataset

Neuroglancer relies on an [`info`](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed#info-json-file-specification) file located at the root of a dataset layer to tell it how to compute file locations and interpret the data in each file. CloudVolume piggy-backs on this functionality.

In the below example, assume you are creating a new segmentation volume from a 3d numpy array "rawdata". Note Precomputed stores data in Fortran (column major, aka CZYX) order. You should do a small test to see if the image is written transposed. You can fix this by uploading `rawdata.T`.

```python3
from cloudvolume import CloudVolume

info = CloudVolume.create_new_info(
    num_channels    = 1,
    layer_type      = 'segmentation',
    data_type       = 'uint64', # Channel images might be 'uint8'
    encoding        = 'raw', # raw, jpeg, compressed_segmentation, fpzip, kempressed
    resolution      = [4, 4, 40], # Voxel scaling, units are in nanometers
    voxel_offset    = [0, 0, 0], # x,y,z offset in voxels from the origin
    mesh            = 'mesh',
    # Pick a convenient size for your underlying chunk representation
    # Powers of two are recommended, doesn't need to cover image exactly
    chunk_size      = [ 512, 512, 16 ], # units are voxels
    volume_size     = [ 250000, 250000, 25000 ], # e.g. a cubic millimeter dataset
)
vol = CloudVolume(cfg.path, info=info)
vol.commit_info()
vol[cfg.x: cfg.x + cfg.length, cfg.y:cfg.y + cfg.length, cfg.z: cfg.z + cfg.length] = rawdata[:,:,:] 
```
| Encoding                | Image Type                 | Lossless | Neuroglancer Viewable | Description                                                                              | 
|-------------------------|----------------------------|----------|-------------|------------------------------------------------------------------------------------------| 
| raw                     | Any                        | Y        | Y           | Serialized numpy arrays.                                                                 | 
| jpeg                    | Image                      | N        | Y           | Multiple slices stiched into a single JPEG.                                              | 
| compressed_segmentation | Segmentation               | Y        | Y           | Renumbered numpy arrays to reduce data width. Also used by Neuroglancer internally.      | 
| fpzip                   | Floating Point             | Y        | N*           | Takes advantage of IEEE 754 structure + L1 Lorenzo predictor to get higher compression.  | 
| kempressed              | Anisotropic Z Floating Point | N        | N*           | Adds manipulations on top of fpzip to achieve higher compression.                        | 

\* Coming soon.


### Examples

```python3
# Basic Examples
vol = CloudVolume('gs://mybucket/retina/image') 
vol = CloudVolume('gs://bucket/dataset/channel', mip=0, bounded=True, fill_missing=False) 
vol = CloudVolume('gs://bucket/dataset/channel', mip=[ 8, 8, 40 ], bounded=True, fill_missing=False) # set mip at this resolution
vol = CloudVolume('gs://bucket/datasset/channel', info=info) # New info file from scratch
image = vol[:,:,:] # Download the entire image stack into a numpy array
listing = vol.exists( np.s_[0:64, 0:128, 0:64] ) # get a report on which chunks actually exist
listing = vol.delete( np.s_[0:64, 0:128, 0:64] ) # delete this region (bbox must be chunk aligned)
vol[64:128, 64:128, 64:128] = image # Write a 64^3 image to the volume

# Microviewer
img = vol[64:1028, 64:1028, 64:128]
img.view() # launches web viewer on http://localhost:8080

# Meshes
vol.mesh.save(12345) # save 12345 as ./12345.ply on disk
vol.mesh.save([12345, 12346, 12347]) # merge three segments into one file
vol.mesh.save(12345, file_format='obj') # 'ply' and 'obj' are both supported
vol.mesh.get(12345) # return the mesh as vertices and faces instead of writing to disk
vol.mesh.get([ 12345, 12346 ]) # return these two segids fused into a single mesh 
vol.mesh.get([ 12345, 12346 ], fuse=False) # return { 12345: mesh, 12346: mesh }

# Skeletons
skel = vol.skeleton.get(12345)
vol.skeleton.upload_raw(segid, skel.vertices, skel.edges, skel.radii, skel.vertex_types) 
vol.skeleton.upload(skel) 

skel.empty() # boolean

bytes = skel.encode() # encode to Precomputed format (bytes)
skel = PrecomputedSkeleton.decode(bytes) # decode from PrecomputedFormat

skel = skel.crop(slices or bbox) # eliminate vertices and edges outside bbox
skel = skel.consolidate() # eliminate duplicate vertices and edges
skel3 = skel.merge(skel2) # merge two skeletons into one
skel = skel.clone() # create copy
skel = PrecomputedSkeleton.from_swc(swcstr) # decode an SWC file
skel_str = skel.to_swc() # convert to SWC file in string representation

skel.cable_length() # sum of all edge lengths
skel = skel.downsample(2, preserve_endpoints=True) # reduce size of skeleton by factor of 2 while ensuring the endpoints aren't stripped

skel1 == skel2 # check if contents of internal arrays match
PrecomputedSkeleton.equivalent(skel1, skel2) # ...even if there are differences like differently numbered edges

# Parallel Operation
vol = CloudVolume('gs://mybucket/retina/image', parallel=True) # Use all cores
vol.parallel = 4 # e.g. any number > 1, use this many cores
data = vol[:] # uses shared memory to coordinate processes under the hood

# Shared Memory Output (can be used by other processes)
vol = CloudVolume(...)
# data backed by a shared memory buffer
# location is optional (defaults to vol.shared_memory_id)
data = vol.download_to_shared_memory(np.s_[:], location='some-example') 
vol.unlink_shared_memory() # delete the shared memory associated with this cloudvolume
vol.shared_memory_id # get/set the default shared memory location for this instance

# Shared Memory Upload
vol = CloudVolume(...)
vol.upload_from_shared_memory('my-shared-memory-id', # do not prefix with /dev/shm
    bbox=Bbox( (0,0,0), (10000, 7500, 64) )) 

# Transfer w/o Excess Memory Allocation
vol = CloudVolume(...)
# single core, send all of vol to destination, no painting memory
vol.transfer_to('gs://bucket/dataset/layer', vol.bounds) 

# Caching, located at $HOME/.cloudvolume/cache/$PROTOCOL/$BUCKET/$DATASET/$LAYER/$RESOLUTION
vol = CloudVolume('gs://mybucket/retina/image', cache=True) # Basic Example
image = vol[0:10,0:10,0:10] # Download partial image and cache
vol[0:10,0:10,0:10] = image # Upload partial image and cache

# Evaluating the Cache
vol.cache.list() # list files in cache at this mip level  
vol.cache.list(mip=1) # list files in cache at mip 1  
vol.cache.num_files() # number of files at this mip level  
vol.cache.num_bytes(all_mips=True) # Return num files for each mip level in a list  
vol.cache.num_bytes() # number of bytes taken up by files, size on disk can be bigger  
vol.cache.num_bytes(all_mips=True) # Return num bytes for each mip level in a list  

vol.cache.enabled = True/False/Path # Turn the cache on/off 
vol.cache.compress = None/True/False # None: Link to cloud setting, Boolean: Force cache to compressed (True) or uncompressed (False)

# Deleting Cache
vol.cache.flush() # Delete local cache for this layer at this mip level  
vol.cache.flush(preserve=Bbox(...)) # Same, but preserve cache in a region of space  
vol.cache.flush_region(region=Bbox(...), mips=[...]) # Delete the cached files in this region at these mip levels (default all mips)  
vol.cache.flush_info()
vol.cache.flush_provenance()
```

### CloudVolume Constructor

```python3
CloudVolume(cloudpath, 
     mip=0, bounded=True, fill_missing=False, autocrop=False, 
     cache=False, compress_cache=None, cdn_cache=False, progress=INTERACTIVE, info=None, 
     provenance=None, compress=None, non_aligned_writes=False, parallel=1)
```

* mip - Which mip level to access
* bounded - Whether access is allowed outside the bounds defined in the info file
* fill_missing - If a chunk is missing, should it be zero filled or throw an EmptyVolumeException? Note that under conditions of high load, it's possible for fill_missing to be activated for existing files. Set to false for extra safety.
* cache - Save uploads/downloads to disk. You can also provide a string path instead of a boolean to specify a custom cache location.
* compress_cache - Override default cache compression behavior if set to a boolean.
* autocrop - If bounded is False, automatically crop requested uploads and downloads to the volume boundary.
* cdn_cache - Set the HTTP Cache-Control header on uploaded image chunks.
* progress - Show progress bars. Defaults to True if in python interactive mode else default False.
* info - Use this info object rather than pulling from the cloud (useful for creating new layers).
* provenance - Use this object as the provenance file.
* compress - None or 'gzip', force this compression algorithm to be used for upload
* non_aligned_writes - True/False. If False, non-chunk-aligned writes will trigger an error with a helpful message. If True,
    Non-aligned writes will proceed. Be careful, non-aligned writes are wasteful in memory and bandwidth, and in a mulitprocessing environment, are subject to an ugly race condition. (c.f. https://github.com/seung-lab/cloud-volume/wiki/Advanced-Topic:-Non-Aligned-Writes)
* parallel - True/False/(int > 0), If False or 1, use a single process. If > 1, use that number of processes for downloading 
   that coordinate over shared memory. If True, use a number of processes equal to the number of available cores.

### CloudVolume Methods

Better documentation coming later, but for now, here's a summary of the most useful method calls. Use help(cloudvolume.CloudVolume.$method) for more info.

* create_new_info (class method) - Helper function for creating info files for creating new data layers.
* refresh_info - Repull the info file.
* refresh_provenance - Repull the provenance file.
* bbox_to_mip - Covert a bounding box or slice from one mip level to another.
* slices_from_global_coords - *deprecated, why not use bbox_to_mip?* Find the CloudVolume slice from MIP 0 coordinates if you're on a different MIP. Often used in combination with neuroglancer. 
* reset_scales - Delete mips other than 0 in the info file. Does not autocommit.
* add_scale - Generate a new mip level in the info property. Does not autocommit.
* commit_info - Push the current info property into the cloud as a JSON file.
* commit_provenance - Push the current provenance property into the cloud as a JSON file.
* mesh - Access mesh operations
	* get - Download an object. Can merge multiple segmentids
	* save - Download an object and save it in `.obj` format. You can combine equivialences into a single object too.
* skeleton - Access Skeletons
  * get - Download an object.
  * upload - Save a skeleton object to the cloud.
  * get_point_cloud - Download the point cloud, a skeleton precursor, for an object. 
* cache - Access cache operations
	* enabled - Boolean switch to enable/disable cache. If true, on reading, check local disk cache before downloading, and save downloaded chunks to cache. When writing, write to the cloud then save the chunks you wrote to cache. If false, bypass cache completely. The cache is located at `$HOME/.cloudvolume/cache`.
	* path - Property that shows the current filesystem path to the cache
	* list - List files in cache 
	* num_files - Number of files in cache at this mip level , use all_mips=True to get them all
	* num_bytes - Return the number of bytes in cache at this mip level, all_mips=True to get them all
	* flush - Delete the cache at this mip level, preserve=Bbox/slice to save a spatial region
	* flush_region - Delete a spatial region at this mip level
* exists - Generate a report on which chunks within a bounding box exist.
* delete - Delete the chunks within this bounding box.
* transfer_to - Transfer data from a bounding box to another data storage location. Does not allocate memory and transfers in blocks, so can transfer large volumes of data. May be less efficient than a dedicated tool like `gsutil` or `aws s3`.
* unlink_shared_memory - Delete shared memory associated with this instance (`vol.shared_memory_id`)
* generate_shared_memory_location - Create a new unique shared memory identifier string. No side effects.
* download_to_shared_memory - Instead of using ordinary numpy memory allocations, download to shared memory.
    Be careful, shared memory is like a file and doesn't disappear unless explicitly unlinked. (`vol.unlink_shared_memory()`)
* upload_from_shared_memory - Upload from a given shared memory block without making a copy.

### CloudVolume Properties

Accessed as `vol.$PROPERTY` like `vol.mip`. Parens next to each property mean (data type:default, writability). (r) means read only, (w) means write only, (rw) means read/write.

* mip (uint:0, rw) - Read from and write to this mip level (0 is highest res). Each additional increment in the number is typically a 2x reduction in resolution.
* bounded (bool:True, rw) - If a region outside of volume bounds is accessed throw an error if True or Fill the region with black (useful for e.g. marching cubes's 1px boundary) if False.
* autocrop (bool:False, rw) - If bounded is False and this option is True, automatically crop requested uploads and downloads to the volume boundary.
* fill_missing (bool:False, rw) - If a file inside volume bounds is unable to be fetched use a block of zeros if True, else throw an error.
* info (dict, rw) - Python dict representation of Neuroglancer info JSON file. You must call `vol.commit_info()` to save your changes to storage.
* provenance (dict-like, rw) - Data layer provenance file representation. You must call `vol.commit_provenance()` to save your changes to storage.
* available_mips (list of ints, r) - Query which mip levels are defined for reading and writing.
* dataset_name (str, rw) - Which dataset (e.g. test_v0, snemi3d_v0) on S3, GS, or FS you're reading and writing to. Known as an "experiment" in BOSS terminology. Writing to this property triggers an info refresh.
* layer (str, rw) - Which data layer (e.g. image, segmentation) on S3, GS, or FS you're reading and writing to. Known as a "channel" in BOSS terminology. Writing to this property triggers an info refresh.
* base_cloudpath (str, r) - The cloud path to the dataset e.g. s3://bucket/dataset/
* layer_cloudpath (str, r) - The cloud path to the data layer e.g. gs://bucket/dataset/image
* info_cloudpath (str, r) - Generate the cloud path to this data layer's info file.
* scales (dict, r) - Shortcut to the 'scales' property of the info object
* scale (dict, rw)* - Shortcut to the working scale of the current mip level
* shape (Vec4, r)* - Like numpy.ndarray.shape for the entire data layer. 
* volume_size (Vec3, r)* - Like shape, but omits channel (x,y,z only). 
* num_channels (int, r) - The number of channels, the last element of shape. 
* layer_type (str, r) - The neuroglancer info type, 'image' or 'segmentation'.
* dtype (str, r) - The info data_type of the volume, e.g. uint8, uint32, etc. Similar to numpy.ndarray.dtype.
* encoding (str, r) - The neuroglancer info encoding. e.g. 'raw', 'jpeg', 'npz'
* resolution (Vec3, r)* - The 3D physical resolution of a voxel in nanometers at the working mip level.
* downsample_ratio (Vec3, r) - Ratio of the current resolution to the highest resolution mip available.
* underlying (Vec3, r)* - Size of the underlying chunks that constitute the volume in storage. e.g. Vec(64, 64, 64)
* key (str, r)* - The 'directory' we're accessing the current working mip level from within the data layer. e.g. '6_6_30'
* bounds (Bbox, r)* - A Bbox object that represents the bounds of the entire volume.
* shared_memory_id (str, rw) - Shared memory location used for parallel operation or for output.

\* These properties can also be accessed with a function named like `vol.mip_$PROPERTY($MIP)`. By default they return the current mip level assigned to the CloudVolume, but any mip level can be accessed via the corresponding `mip_` function. Example: `vol.mip_resolution(2)` would return the resolution of mip 2.

### VolumeCutout Functions

When you download an image using CloudVolume it gives you a `VolumeCutout`. These are `numpy.ndarray` subclasses that support a few extra properties to help make book keeping easier. The major advantage is `save_images()` which can help you view your dataset as PNG slices.

* `dataset_name` - The dataset this image came from.
* `layer` - Which layer it came from.
* `mip` - Which mip it came from
* `layer_type` - "image" or "segmentation"
* `bounds` - The bounding box of the cutout
* `num_channels` - Alias for `vol.shape[3]`
* `save_images()` - Save Z slice PNGs of the current image to `./saved_images` for manual inspection
* `view()` - Start a local web server (http://localhost:8080) that can view small volumes interactively.

### Viewing a Precomputed Volume on Disk

If you have serialized a Precomputed volume onto local disk and would like to point neuroglancer to it, this solution works nicely for experimenting:

```bash
npm install http-server -g
cd $LOCATION_ABOVE_DATA
http-server -p 3000 --cors
```

You can then point any hosted version of neuroglancer at it using `precomputed://http://localhost:3000/NAME/OF/LAYER`.

### Microviewer

CloudVolume includes a built-in dependency free viewer for 3D volumetric datasets smaller than about 2GB uncompressed. It supports uint8, uint16, uint32, float32, and float64 data types for both images and segmentation and can render a composite overlay of image and segmentation.  

You can launch a viewer using the `.view()` method of a VolumeCutout object or by using the `view(...)` or `hyperview(...)` functions that come with the cloudvolume module. This launches a web server on `http://localhost:8080`. You can read more [on the wiki](https://github.com/seung-lab/cloud-volume/wiki/%CE%BCViewer).

```python3
from cloudvolume import CloudVolume, view, hyperview

channel_vol = CloudVolume(...)
seg_vol = CloudVolume(...)
img = vol[...]
seg = vol[...]

img.view() # works on VolumeCutouts
seg.view()
view(img) # alternative for arbitrary numpy arrays
view(seg) 
hyperview(img, seg) # img and seg shape must match

>>> Viewer server listening to http://localhost:8080
```

## Spinoff Projects

CloudVolume in Julia - https://github.com/seung-lab/CloudVolume.jl   
fpzip Python Package - https://github.com/seung-lab/fpzip  
compressed_segmentation Python Package - https://github.com/seung-lab/compressedseg  
Igneous - https://github.com/seung-lab/igneous   

## Acknowledgments

Thank you to Jeremy Maitin-Shepard for creating [Neuroglancer](https://github.com/google/neuroglancer) and defining the Precomputed format.  
Thanks to Yann Leprince for providing a [pure Python codec](https://github.com/HumanBrainProject/neuroglancer-scripts) for the compressed_segmentation format.  
Thanks to Jeremy Maitin-Shepard and Stephen Plaza for their C++ code defining the compression and decompression (respectively) protocol for [compressed_segmentation](https://github.com/janelia-flyem/compressedseg).  
Thanks to Peter Lindstrom et al. for [their work](https://computation.llnl.gov/projects/floating-point-compression) on fpzip, the C++ code, and assistance.  
Thanks to Nico Kemnitz for his work on the "Kempression" protocol that builds on fpzip (we named it, not him).   
  
