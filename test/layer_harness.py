import pytest

import shutil
import numpy as np

import os

from cloudvolume import Storage, CloudVolume
from cloudvolume.lib import Bbox, Vec, min2, max2, find_closest_divisor

TEST_NUMBER = np.random.randint(0, 999999)
CLOUD_BUCKET = 'seunglab-test'

layer_path = '/tmp/removeme/'

def create_storage(layer_name='layer'):
    stor_path = os.path.join(layer_path, layer_name)
    return Storage('file://' + stor_path, n_threads=0)

def create_image(size, layer_type="image", dtype=None):
    default = lambda dt: dtype or dt

    if layer_type == "image":
        return np.random.randint(255, size=size, dtype=default(np.uint8))
    elif layer_type == 'affinities':
        return np.random.uniform(low=0, high=1, size=size).astype(default(np.float32))
    elif layer_type == "segmentation":
        return np.random.randint(0xFFFFFF, size=size, dtype=np.uint32)
    else:
        high = np.array([0], dtype=default(np.uint32)) - 1
        return np.random.randint(high[0], size=size, dtype=default(np.uint32))

def create_layer(size, offset, layer_type="image", layer_name='layer', dtype=None, order='F'):
    random_data = create_image(size, layer_type, dtype)
    vol = upload_image(random_data, offset, layer_type, layer_name, order=order)
    return vol, random_data

def upload_image(image, offset, layer_type, layer_name, order='F'):
    lpath = 'file://{}'.format(os.path.join(layer_path, layer_name))
    
    # Jpeg encoding is lossy so it won't work
    vol = create_volume_from_image(
        image, 
        offset=offset,
        layer_path=lpath,
        layer_type=layer_type, 
        encoding="raw", 
        resolution=[1,1,1],
        order=order
    )
    
    return vol

def delete_layer(path=layer_path):
    if os.path.exists(path):
        shutil.rmtree(path)  

# Helper Functions

def create_volume_from_image(image, offset, layer_path, layer_type, resolution, 
                                encoding, order='F'):
    assert layer_type in ('image', 'segmentation', 'affinities')

    offset = Vec(*offset)
    if order=='F':
        volsize = Vec(*image.shape[:3])
    else:
        volsize = Vec(*image.shape[-3:])

    data_type = str(image.dtype)
    bounds = Bbox(offset, offset + volsize)

    import pdb; pdb.set_trace()
    neuroglancer_chunk_size = find_closest_divisor(volsize, closest_to=[64,64,64])

    info = CloudVolume.create_new_info(
        num_channels=1, # Increase this number when we add more tests for RGB
        layer_type=layer_type, 
        data_type=data_type, 
        encoding=encoding,
        resolution=resolution, 
        voxel_offset=bounds.minpt, 
        volume_size=bounds.size3(),
        mesh=(layer_type == 'segmentation'), 
        chunk_size=neuroglancer_chunk_size,
    )

    vol = CloudVolume(layer_path, mip=0, info=info, order=order)
    vol.commit_info()
    vol[:,:,:] = image
    return vol
