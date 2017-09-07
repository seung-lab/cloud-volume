[![Build Status](https://travis-ci.org/seung-lab/cloud-volume.svg?branch=master)](https://travis-ci.org/seung-lab/cloud-volume)

# cloud-volume

Python client for reading and writing to Neuroglancer Precomputed volumes on cloud services. (https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed)

When working with a particular dataset, say an EM scan of a mouse, fish, or fly brain, you'll typically store that as a grayscale data layer accessible to neuroglanger. You may store additional labellings and processing results as other layers.

## Setup

You'll need to set up your cloud credentials as well as the main install.

### Credentials

```
mkdir -p ~/.neuroglancer/secrets/
echo $GOOGLE_STORAGE_PROJECT > ~/.neuroglancer/project_name # needed for Google
mv aws-secret.json ~/.neuroglancer/secrets/ # needed for Google
mv google-secret.json ~/.neuroglancer/secrets/ # needed for Amazon
mv boss-secret.json ~/.neuroglancer/secrets/ # needed for the BOSS
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
vol[64:128, 64:128, 64:128] = image # Write a 64^3 image to the volume
vol.save_mesh(12345) # save 12345 as ./12345.obj
vol.save_mesh([12345, 12346, 12347]) # merge three segments into one obj
```

### Property List

Accessed as `vol.$PROPERTY` like `vol.mip`. Parens next to each property mean (data type:default, writability). (r) means read only, (w) means write only, (rw) means read/write.

* mip (uint:0, rw) - Read from and write to this mip level (0 is highest res). Each additional increment in the number is typically a 2x reduction in resolution.
* bounded (bool:True, rw) - If a region outside of volume bounds is accessed throw an error if True or Fill the region with black (useful for e.g. marching cubes's 1px boundary) if False.
* fill_missing (bool:False, rw) - If a file inside volume bounds is unable to be fetched use a block of zeros if True, else throw an error
* info (dict, rw) - Python dict representation of Neuroglancer info JSON file. You must call `vol.commit_info()` to save your changes to storage.
* provenance (dict-like, rw) - Data layer provenance file representation. You must call `vol.commit_provenance()` to save your changes to storage.
* available_mips (list of ints, r) - Query which mip levels are defined for reading and writing.
* dataset_name (str, rw) - Which dataset (e.g. test_v0, snemi3d_v0) on S3, GS, or FS you're reading and writing to. Known as an "experiment" in BOSS terminology. Writing to this property triggers an info refresh.
* layer (str, rw) - Which data layer (e.g. image, segmentation) on S3, GS, or FS you're reading and writing to. Known as a "channel" in BOSS terminology. Writing to this property triggers an info refresh.
* base_cloudpath (str, r) - The cloud path to the dataset e.g. s3://bucket/dataset/
* layer_cloudpath (str, r) - The cloud path to the data layer e.g. gs://bucket/dataset/image
* info_cloudpath (str, r) - Generate the cloud path to this data layer's info file.
* scales (dict, r) - Shortcut to the 'scales' property of the info object
* scale (dict, r) - Shortcut to the working scale of the current mip level
* shape (Vec4, r) - Like numpy.ndarray.shape for the entire data layer. 
* volume_size (Vec3, r) - Like shape, but omits channel (x,y,z only). 
* num_channels (int, r) - The number of channels, the last element of shape. 
* layer_type (str, r) - The neuroglancer info type, 'image' or 'segmentation'.
* dtype (str, r) - The info data_type of the volume, e.g. uint8, uint32, etc. Similar to numpy.ndarray.dtype.
* encoding (str, r) - The neuroglancer info encoding. e.g. 'raw', 'jpeg', 'npz'
* resolution (Vec3, r) - The 3D physical resolution of a voxel in nanometers at the working mip level.
* downsample_ratio (Vec3, r) - Ratio of the current resolution to the highest resolution mip available.
* underlying (Vec3, r) - Size of the underlying chunks that constitute the volume in storage. e.g. Vec(64, 64, 64)
* key (str, r) - The 'directory' we're accessing the current working mip level from within the data layer. e.g. '6_6_30'
* bounds (Bbox, r) - A Bbox object that represents the bounds of the entire volume.



