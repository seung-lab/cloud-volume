from __future__ import print_function
from six.moves import range

import pytest
import re
import time

from cloudvolume.storage import Storage
from layer_harness import delete_layer

def test_path_extraction():
    assert (Storage.extract_path('s3://bucket_name/dataset_name/layer_name') 
        == Storage.ExtractedPath('s3', "bucket_name", None, 'dataset_name', 'layer_name'))

    assert Storage.extract_path('s4://dataset_name/layer_name') is None

    assert Storage.extract_path('dataset_name/layer_name') is None

    assert Storage.extract_path('s3://dataset_name') is None

    assert (Storage.extract_path('s3://neuroglancer/intermediate/path/dataset_name/layer_name') 
        == Storage.ExtractedPath('s3', 'neuroglancer', 'intermediate/path/','dataset_name', 'layer_name'))

    assert (Storage.extract_path('file:///tmp/dataset_name/layer_name') 
        == Storage.ExtractedPath('file', "/tmp",  None, 'dataset_name', 'layer_name'))

    assert (Storage.extract_path('file://neuroglancer/intermediate/path/dataset_name/layer_name') 
        == Storage.ExtractedPath('file', 'neuroglancer','intermediate/path/','dataset_name', 'layer_name'))

    assert (Storage.extract_path('gs://neuroglancer/intermediate/path/dataset_name/layer_name') 
        == Storage.ExtractedPath('gs', 'neuroglancer', 'intermediate/path/','dataset_name', 'layer_name'))

    assert Storage.extract_path('s3://dataset_name/layer_name/') is None

#TODO delete files created by tests
def test_read_write():
    urls = [
        "file:///tmp/removeme/read_write",
        "gs://neuroglancer/removeme/read_write",
        "s3://neuroglancer/removeme/read_write"
    ]

    for num_threads in range(0,11,5):
        for url in urls:
            with Storage(url, n_threads=num_threads) as s:
                content = b'some_string'
                s.put_file('info', content, compress=False)
                s.wait()
                assert s.get_file('info') == content
                assert s.get_file('nonexistentfile') is None

                num_infos = max(num_threads, 1)

                results = s.get_files([ 'info' for i in range(num_infos) ])

                assert len(results) == num_infos
                assert results[0]['filename'] == 'info'
                assert results[0]['content'] == content
                assert all(map(lambda x: x['error'] is None, results))
                assert s.get_files([ 'nonexistentfile' ])[0]['content'] is None

    delete_layer("/tmp/removeme/read_write")

def test_delete():
    urls = [
        "file:///tmp/removeme/delete",
        "gs://neuroglancer/removeme/delete",
        "s3://neuroglancer/removeme/delete"
    ]

    for url in urls:
        with Storage(url, n_threads=1) as s:
            content = b'some_string'
            s.put_file('delete-test', content, compress=False).wait()
            s.put_file('delete-test-compressed', content, compress=True).wait()
            assert s.get_file('delete-test') == content
            s.delete_file('delete-test').wait()
            assert s.get_file('delete-test') is None

            assert s.get_file('delete-test-compressed') == content
            s.delete_file('delete-test-compressed').wait()
            assert s.get_file('delete-test-compressed') is None

def test_compression():
    urls = [
        "file:///tmp/removeme/compression",
        "gs://neuroglancer/removeme/compression",
        "s3://neuroglancer/removeme/compression"
    ]

    for url in urls:
        with Storage(url, n_threads=5) as s:
            content = b'some_string'
            s.put_file('info', content, compress=True)
            s.wait()
            assert s.get_file('info') == content
            assert s.get_file('nonexistentfile') is None
            s.delete_file('info')

    delete_layer("/tmp/removeme/compression")

def test_list():  
    urls = [
        "file:///tmp/removeme/list",
        "gs://neuroglancer/removeme/list",
        "s3://neuroglancer/removeme/list"
    ]

    for url in urls:
        with Storage(url, n_threads=5) as s:
            print('testing service:', url)
            content = b'some_string'
            s.put_file('info1', content, compress=False)
            s.put_file('info2', content, compress=False)
            s.put_file('build/info3', content, compress=False)
            s.put_file('level1/level2/info4', content, compress=False)
            s.put_file('info5', content, compress=True)
            s.put_file('info.txt', content, compress=False)
            s.wait()
            time.sleep(1) # sometimes it takes a moment for google to update the list
            assert set(s.list_files(prefix='')) == set(['build/info3','info1', 'info2', 'level1/level2/info4', 'info5', 'info.txt'])
            
            assert set(s.list_files(prefix='inf')) == set(['info1','info2','info5','info.txt'])
            assert set(s.list_files(prefix='info1')) == set(['info1'])
            assert set(s.list_files(prefix='build')) == set(['build/info3'])
            assert set(s.list_files(prefix='build/')) == set(['build/info3'])
            assert set(s.list_files(prefix='level1/')) == set(['level1/level2/info4'])
            assert set(s.list_files(prefix='nofolder/')) == set([])

            # Tests (1)
            assert set(s.list_files(prefix='', flat=True)) == set(['info1','info2','info5','info.txt'])
            assert set(s.list_files(prefix='inf', flat=True)) == set(['info1','info2','info5','info.txt'])
            # Tests (2)
            assert set(s.list_files(prefix='build', flat=True)) == set([])
            # Tests (3)
            assert set(s.list_files(prefix='level1/', flat=True)) == set([])
            assert set(s.list_files(prefix='build/', flat=True)) == set(['build/info3'])
            # Tests (4)
            assert set(s.list_files(prefix='build/inf', flat=True)) == set(['build/info3'])

            for file_path in ('info1', 'info2', 'build/info3', 'level1/level2/info4', 'info5', 'info.txt'):
                s.delete_file(file_path)
    
    delete_layer("/tmp/removeme/list")


def test_exists():
    urls = [
        # "file:///tmp/removeme/exists",
        # "gs://neuroglancer/removeme/exists",
        "s3://neuroglancer/removeme/exists"
    ]

    for url in urls:
        with Storage(url, n_threads=5) as s:
            content = b'some_string'
            s.put_file('info', content, compress=False)
            s.wait()
            time.sleep(1) # sometimes it takes a moment for google to update the list
            
            assert s.exists('info')
            assert not s.exists('doesntexist')
            s.delete_file('info')
