[![Build Status](https://travis-ci.org/seung-lab/cloud-volume.svg?branch=master)](https://travis-ci.org/seung-lab/cloud-volume)

# cloud-volume

Python client for reading and writing to Neuroglancer Precomputed volumes on cloud services. (https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed)

When working with a particular dataset, say an EM scan of a mouse, fish, or fly brain, you'll typically store that as a grayscale data layer accessible to neuroglanger. You may store additional labellings and processing results as other layers.


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
image = vol[:,:,:] # Download the entire image stack into a numpy array
vol[64:128, 64:128, 64:128] = image # Write a 64^3 image to the volume
vol.save_mesh(12345) # save 12345 as ./12345.obj
vol.save_mesh([12345, 12346, 12347]) # merge three segments into one obj
```

## Setup

You'll need to set up your cloud credentials as well as the main install.

### Credentials

```
mkdir -p ~/.neuroglancer/secrets/
echo $GOOGLE_STORAGE_PROJECT > ~/.neuroglancer/project_name # needed for Google
mv aws-secret.json ~/.neuroglancer/secrets/ # needed for Google
mv google-secret.json ~/.neuroglancer/secrets/ # needed for Amazon
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





