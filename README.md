[![Build Status](https://travis-ci.org/seung-lab/cloud-volume.svg?branch=master)](https://travis-ci.org/seung-lab/cloud-volume)

# cloud-volume

Cloud-Volume is a library for writing services that work with the "precomputed" volume type for Neuroglancer. These volumes are typically stored in AWS S3 or Google GS (and the library can work with those services given appropriate credentials), but they can also be stored on a regular webserver, and the library can generate the appropriate file hierarchies for that as well. (https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed)

A typical dataset for Neuroglancer might be an EM scan of a mouse, fish, or fly brain. It is normally stored that as a grayscale data layer accessible to neuroglancer. You may store additional labelings and processing results (such as segmentation) as other layers.

## Setup

Cloud-volume is compatible with Python 2.6+ and 3.4+ (we've noticed it's faster on Python 3). On linux it requires g++ and python3-dev. After installation, you'll also need to set up your cloud credentials. 

### Credentials

```
mkdir -p ~/.cloudvolume/secrets/
mv aws-secret.json ~/.cloudvolume/secrets/ # needed for Amazon
mv google-secret.json ~/.cloudvolume/secrets/ # needed for Google
mv boss-secret.json ~/.cloudvolume/secrets/ # needed for the BOSS
```

The format for the aws-secret.json file is as follows:
```
{
	"AWS_ACCESS_KEY_ID": "",
	"AWS_SECRET_ACCESS_KEY_ID": ""
}
```
(adjust key name for your service, and fill in the values with your credentials)
### pip

```
pip install cloud-volume
```

### Manual
```
git clone git@github.com:seung-lab/cloud-volume.git
cd cloud-volume

# With virtualenvwrapper
mkvirtualenv cv
workon cv
# With only virtualenv
virtualenv venv
source venv/bin/activate

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

# Parallel Operation
vol.parallel = 4 # e.g. any number > 1
data = vol[:] # uses shared memory to coordinate processes
del data # closes mmap file handle
vol.unlink_shared_memory() # delete the shared memory associated with this cloudvolume
vol.shared_memory_id # get/set the shared memory location for this instance

# Shared Memory Output
vol = CloudVolume(..., output_to_shared_memory=True)
vol = CloudVolume(..., output_to_shared_memory='my-shared-memory-location')
data = vol[:] # data now is a shared memory buffer
vol.unlink_shared_memory() # delete the shared memory associated with this cloudvolume
vol.shared_memory_id # get/set the shared memory location for this instance
vol.output_to_shared_memory = True/False # Turn this feature on/off

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

# Deleting Cache
vol.cache.flush() # Delete local cache for this layer at this mip level  
vol.cache.flush(preserve=Bbox(...)) # Same, but presere cache in a region of space  
vol.cache.flush_region(region=Bbox(...), mips=[...]) # Delete the cached files in this region at these mip levels (default all mips)  

```

Loading a segmentation as a new volume,
assume you have a 3d numpy array as "rawdata" and various cfg values:
```
metadata = cloudvolume.CloudVolume.create_new_info(
        num_channels    = 1,
        layer_type      = 'segmentation',
        data_type       = 'uint64',
        encoding        = 'raw',
        resolution      = [8, 8, 8], # Pick scaling for your data!
        voxel_offset    = [cfg.x, cfg.y, cfg.z],
        mesh            = 'mesh',
        chunk_size      = [cfg.chunksize, cfg.chunksize, cfg.chunksize], # This must divide evenly into image length or you won't cover the whole cube
        volume_size     = [cfg.length, cfg.length, cfg.length]
        )
vol = cloudvolume.CloudVolume(cfg.path, mip=cfg.compression, info=metadata)
vol.commit_info()
vol[cfg.x: cfg.x + cfg.length, cfg.y:cfg.y + cfg.length, cfg.z: cfg.z + cfg.length] = rawdata[:,:,:] # Be wary of XYZ-versus-ZYX issues

```

### CloudVolume Constructor

`CloudVolume(cloudpath, mip=0, bounded=True, fill_missing=False, autocrop=False, cache=False, cdn_cache=False, progress=INTERACTIVE, info=None, provenance=None, compress=None, non_aligned_writes=False, parallel=1, output_to_shared_memory=False)`  

* mip - Which mip level to access
* bounded - Whether access is allowed outside the bounds defined in the info file
* fill_missing - If a chunk is missing, should it be zero filled or throw an EmptyVolumeException?
* cache - Save uploads/downloads to disk. You can also provide a string path instead of a boolean to specify a custom cache location.
* autocrop - If bounded is False, automatically crop requested uploads and downloads to the volume boundary.
* cdn_cache - Set the HTTP Cache-Control header on uploaded image chunks.
* progress - Show progress bars. Defaults to True if in python interactive mode else default False.
* info - Use this info object rather than pulling from the cloud (useful for creating new layers).
* provenance - Use this object as the provenance file.
* compress - None or 'gzip', force this compression algorithm to be used for upload
* non_aligned_writes - True/False. If False, non-chunk-aligned writes will trigger an error with a helpful message. If True,
    Non-aligned writes will proceed. Be careful, non-aligned writes are wasteful in memory and bandwidth, and in a mulitprocessing environment, are subject to an ugly race condition. (c.f. https://github.com/seung-lab/cloud-volume/issues/87)
* parallel - True/False/(int > 0), If False or 1, use a single process. If > 1, use that number of processes for downloading 
   that coordinate over shared memory. If True, use a number of processes equal to the number of available cores.
* otuput_to_shared_memory - True/False/string. Instead of using ordinary numpy memory allocations, download to shared memory.
    Be careful, shared memory is like a file and doesn't disappear unless explicitly unlinked. (`vol.unlink_shared_memory()`)
    A string input specifies a possibly preexisting shared memory location.

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
* unlink_shared_memory - Delete shared memory associated with this instance (`vol.shared_memory_id`)
* generate_shared_memory_location - Create a new unique shared memory identifier string. No side effects.

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
* shared_memory_id (str, rw) - Shared memory location used for parallel operation or for output.
* output_to_shared_memory (bool, rw) - Turn on/off outputing to shared memory.

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

