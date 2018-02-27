[![Build Status](https://travis-ci.org/seung-lab/cloud-volume.svg?branch=master)](https://travis-ci.org/seung-lab/cloud-volume)

# cloud-volume

Python client for reading and writing to Neuroglancer Precomputed volumes on cloud services. (https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed)

When working with a particular dataset, say an EM scan of a mouse, fish, or fly brain, you'll typically store that as a grayscale data layer accessible to neuroglanger. You may store additional labellings and processing results as other layers.

## Setup

You'll need to set up your cloud credentials as well as the main install. On linux, installation requires python3-dev and g++.

### Credentials

```
mkdir -p ~/.cloudvolume/secrets/
mv aws-secret.json ~/.cloudvolume/secrets/ # needed for Amazon
mv google-secret.json ~/.cloudvolume/secrets/ # needed for Google
mv boss-secret.json ~/.cloudvolume/secrets/ # needed for the BOSS
```

### pip

```
pip install cloud-volume
```

### Manual
```
git clone git@github.com:seung-lab/cloud-volume.git
cd cloud-volume
mkvirtualenv cloud-volume
workon cloud-volume
pip install -e .
```

## Other Languages

Julia - https://github.com/seung-lab/CloudVolume.jl

## Usage

Supports reading and writing to neuroglancer data layers on Amazon S3, Google Storage, and the local file system.

Supported URLs are of the forms:

$PROTOCOL://$BUCKET/$DATASET/$LAYER  

### Supported Protocols 
* gs:   Google Storage
* s3:   Amazon S3
* boss: The BOSS (https://docs.theboss.io/docs)
* file: Local File System (absolute path)

### Examples

```
vol = CloudVolume('gs://mybucket/retina/image') # Basic Example
vol = CloudVolume('gs://buck/ds/chan', mip=0, bounded=True, fill_missing=False) # Using multiple initialization options
vol = CloudVolume('gs://buck/ds/chan', info=info) # Creating a new volume's info file from scratch
image = vol[:,:,:] # Download the entire image stack into a numpy array
listing = vol.exists( np.s_[0:64, 0:128, 0:64] ) # get a report on which chunks actually exist
listing = vol.delete( np.s_[0:64, 0:128, 0:64] ) # delete this region (bbox must be chunk aligned)
vol[64:128, 64:128, 64:128] = image # Write a 64^3 image to the volume
vol.mesh.save(12345) # save 12345 as ./12345.obj
vol.mesh.save([12345, 12346, 12347]) # merge three segments into one obj
vol.mesh.get(12345) # return the mesh as vertices and faces instead of writing to disk

# Caching, located at $HOME/.cloudvolume/cache/$PROTOCOL/$BUCKET/$DATASET/$LAYER/$RESOLUTION
vol = CloudVolume('gs://mybucket/retina/image', cache=True) # Basic Example
image = vol[0:10,0:10,0:10] # Download partial image and cache
vol[0:10,0:10,0:10] = image # Upload partial image and cache
vol.flush_cache() # Delete local cache for this layer at this mip level
```

### CloudVolume Constructor

`CloudVolume(cloudpath, mip=0, bounded=True, fill_missing=False, cache=False, cdn_cache=False, progress=INTERACTIVE, info=None, provenance=None)`  

* mip - Which mip level to access
* bounded - Whether access is allowed outside the bounds defined in the info file
* fill_missing - If a chunk is missing, should it be zero filled or throw an EmptyVolumeException?
* cache - Save uploads/downloads to disk. You can also provide a string path instead of a boolean to specify a custom cache location.
* cdn_cache - Set the HTTP Cache-Control header on uploaded image chunks.
* progress - Show progress bars. Defaults to True if in python interactive mode else default False.
* info - Use this info object rather than pulling from the cloud (useful for creating new layers).
* provenance - Use this object as the provenance file.

### CloudVolume Methods

Better documentation coming later, but for now, here's a summary of the most useful method calls. Use help(cloudvolume.CloudVolume.$method) for more info.

* create_new_info (class method) - Helper function for creating info files for creating new data layers.
* refresh_info - Repull the info file.
* refresh_provenance - Repull the provenance file.
* slices_from_global_coords - Find the CloudVolume slice from MIP 0 coordinates if you're on a different MIP. Often used in combination with neuroglancer.
* reset_scales - Delete mips other than 0 in the info file. Does not autocommit.
* add_scale - Generate a new mip level in the info property. Does not autocommit.
* commit_info - Push the current info property into the cloud as a JSON file.
* commit_provenance - Push the current provenance property into the cloud as a JSON file.
* mesh - Access mesh operations
	* get - Download an object. Can merge multiple segmentids
	* save - Download an object and save it in `.obj` format. You can combine equivialences into a single object too.
* exists - Generate a report on which chunks within a bounding box exist.
* delete - Delete the chunks within this bounding box.


### CloudVolume Properties

Accessed as `vol.$PROPERTY` like `vol.mip`. Parens next to each property mean (data type:default, writability). (r) means read only, (w) means write only, (rw) means read/write.

* mip (uint:0, rw) - Read from and write to this mip level (0 is highest res). Each additional increment in the number is typically a 2x reduction in resolution.
* bounded (bool:True, rw) - If a region outside of volume bounds is accessed throw an error if True or Fill the region with black (useful for e.g. marching cubes's 1px boundary) if False.
* fill_missing (bool:False, rw) - If a file inside volume bounds is unable to be fetched use a block of zeros if True, else throw an error.
* cache (bool:False, rw) - If true, on reading, check local disk cache before downloading, and save downloaded chunks to cache. When writing, write to the cloud then save the chunks you wrote to cache. If false, bypass cache completely. The cache is located at `$HOME/.cloudvolume/cache`.
* info (dict, rw) - Python dict representation of Neuroglancer info JSON file. You must call `vol.commit_info()` to save your changes to storage.
* provenance (dict-like, rw) - Data layer provenance file representation. You must call `vol.commit_provenance()` to save your changes to storage.
* available_mips (list of ints, r) - Query which mip levels are defined for reading and writing.
* dataset_name (str, rw) - Which dataset (e.g. test_v0, snemi3d_v0) on S3, GS, or FS you're reading and writing to. Known as an "experiment" in BOSS terminology. Writing to this property triggers an info refresh.
* layer (str, rw) - Which data layer (e.g. image, segmentation) on S3, GS, or FS you're reading and writing to. Known as a "channel" in BOSS terminology. Writing to this property triggers an info refresh.
* base_cloudpath (str, r) - The cloud path to the dataset e.g. s3://bucket/dataset/
* layer_cloudpath (str, r) - The cloud path to the data layer e.g. gs://bucket/dataset/image
* info_cloudpath (str, r) - Generate the cloud path to this data layer's info file.
* scales (dict, r) - Shortcut to the 'scales' property of the info object
* scale (dict, r)† - Shortcut to the working scale of the current mip level
* shape (Vec4, r)† - Like numpy.ndarray.shape for the entire data layer. 
* volume_size (Vec3, r)† - Like shape, but omits channel (x,y,z only). 
* num_channels (int, r) - The number of channels, the last element of shape. 
* layer_type (str, r) - The neuroglancer info type, 'image' or 'segmentation'.
* dtype (str, r) - The info data_type of the volume, e.g. uint8, uint32, etc. Similar to numpy.ndarray.dtype.
* encoding (str, r) - The neuroglancer info encoding. e.g. 'raw', 'jpeg', 'npz'
* resolution (Vec3, r)† - The 3D physical resolution of a voxel in nanometers at the working mip level.
* downsample_ratio (Vec3, r) - Ratio of the current resolution to the highest resolution mip available.
* underlying (Vec3, r)† - Size of the underlying chunks that constitute the volume in storage. e.g. Vec(64, 64, 64)
* key (str, r)† - The 'directory' we're accessing the current working mip level from within the data layer. e.g. '6_6_30'
* bounds (Bbox, r)† - A Bbox object that represents the bounds of the entire volume.

† These properties can also be accessed with a function named like `vol.mip_$PROPERTY($MIP)`. By default they return the current mip level assigned to the CloudVolume, but any mip level can be accessed via the corresponding `mip_` function. Example: `vol.mip_resolution(2)` would return the resolution of mip 2.

### VolumeCutout Functions

When you download an image using CloudVolume it gives you a `VolumeCutout`. These are `numpy.ndarray` subclasses that support a few extra properties to help make book keeping easier. The major advantage is `save_images()` which can help you debug your dataset.

* `dataset_name` - The dataset this image came from.
* `layer` - Which layer it came from.
* `mip` - Which mip it came from
* `layer_type` - "image" or "segmentation"
* `bounds` - The bounding box of the cutout
* `num_channels` - Alias for `vol.shape[3]`
* `save_images()` - Save Z slice PNGs of the current image to `./saved_images` for manual inspection



