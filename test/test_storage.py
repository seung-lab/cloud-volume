# from __future__ import print_function
# from six.moves import range

# import pytest
# import re
# import time

# from cloudvolume.storage import Storage
# from cloudvolume import exceptions, Bbox, chunks
# from layer_harness import delete_layer, TEST_NUMBER


# #TODO delete files created by tests
# def test_read_write():
#   urls = [
#     "file:///tmp/removeme/read_write",
#     "gs://seunglab-test/cloudvolume/read_write",
#     "s3://seunglab-test/cloudvolume/read_write"
#   ]

#   for num_threads in range(0,11,5):
#     for url in urls:
#       url = url + '-' + str(TEST_NUMBER)
#       with Storage(url, n_threads=num_threads) as s:
#         content = b'some_string'
#         s.put_file('info', content, compress=None, cache_control='no-cache')
#         s.wait()
#         assert s.get_file('info') == content
#         assert s.get_file('nonexistentfile') is None

#         num_infos = max(num_threads, 1)

#         results = s.get_files([ 'info' for i in range(num_infos) ])

#         assert len(results) == num_infos
#         assert results[0]['filename'] == 'info'
#         assert results[0]['content'] == content
#         assert all(map(lambda x: x['error'] is None, results))
#         assert s.get_files([ 'nonexistentfile' ])[0]['content'] is None

#         s.delete_file('info')
#         s.wait()

#         s.put_json('info', { 'omg': 'wow' }, cache_control='no-cache')
#         s.wait()
#         results = s.get_json('info')
#         assert results == { 'omg': 'wow' }

#   delete_layer("/tmp/removeme/read_write")

# def test_http_read():
#   with Storage("https://storage.googleapis.com/seunglab-test/test_v0/black/") as stor:
#     info = stor.get_json('info')

#   assert info == {
#     "data_type": "uint8",
#     "num_channels": 1,
#     "scales": [
#       {
#         "chunk_sizes": [
#           [
#             64,
#             64,
#             50
#           ]
#         ],
#         "encoding": "raw",
#         "key": "6_6_30",
#         "resolution": [
#           6,
#           6,
#           30
#         ],
#         "size": [
#           1024,
#           1024,
#           100
#         ],
#         "voxel_offset": [
#           0,
#           0,
#           0
#         ]
#       }
#     ],
#     "type": "image"
#   }


# def test_http_read_brotli_image():
#   fn = "2_2_50/4096-4608_4096-4608_112-128"
#   bbox = Bbox.from_filename(fn) # possible off by one error w/ exclusive bounds

#   with Storage("https://open-neurodata.s3.amazonaws.com/kharris15/apical/em") as stor:
#     img_bytes = stor.get_file(fn)

#   img = chunks.decode(img_bytes, 'raw', shape=bbox.size3(), dtype="uint8")

#   assert img.shape == (512, 512, 16)


# def test_delete():
#   urls = [
#     "file:///tmp/removeme/delete",
#     "gs://seunglab-test/cloudvolume/delete",
#     "s3://seunglab-test/cloudvolume/delete"
#   ]

#   for url in urls:
#     url = url + '-' + str(TEST_NUMBER)
#     with Storage(url, n_threads=1) as s:
#       content = b'some_string'
#       s.put_file('delete-test', content, compress=None, cache_control='no-cache').wait()
#       s.put_file('delete-test-compressed', content, compress='gzip', cache_control='no-cache').wait()
#       assert s.get_file('delete-test') == content
#       s.delete_file('delete-test').wait()
#       assert s.get_file('delete-test') is None

#       assert s.get_file('delete-test-compressed') == content
#       s.delete_file('delete-test-compressed').wait()
#       assert s.get_file('delete-test-compressed') is None

#       # Reset for batch delete
#       s.put_file('delete-test', content, compress=None, cache_control='no-cache').wait()
#       s.put_file('delete-test-compressed', content, compress='gzip', cache_control='no-cache').wait()
#       assert s.get_file('delete-test') == content
#       assert s.get_file('delete-test-compressed') == content

#       s.delete_files(['delete-test', 'delete-nonexistent',
#                       'delete-test-compressed']).wait()
#       assert s.get_file('delete-test') is None
#       assert s.get_file('delete-test-compressed') is None

# def test_compression():
#   urls = [
#     "file:///tmp/removeme/compress",
#     "gs://seunglab-test/cloudvolume/compress",
#     "s3://seunglab-test/cloudvolume/compress"
#   ]

#   compression_tests = [
#     '',
#     None,
#     True,
#     False,
#     'gzip',
#     'br',
#   ]

#   for url in urls:
#     url = url + '-' + str(TEST_NUMBER)
#     for method in compression_tests:
#       with Storage(url, n_threads=5) as s:
#         content = b'some_string'

#         # remove when GCS enables "br"
#         if method == "br" and "gs://" in url:
#           with pytest.raises(
#               exceptions.UnsupportedCompressionType, 
#               match="Brotli unsupported on google cloud storage"
#           ):
#             s.put_file('info', content, compress=method)
#             s.wait()
#             retrieved = s.get_file('info')
#         else:
#           s.put_file('info', content, compress=method)
#           s.wait()
#           retrieved = s.get_file('info')
#           assert content == retrieved

#         assert s.get_file('nonexistentfile') is None

#     with Storage(url, n_threads=5) as s:
#       content = b'some_string'
#       try:
#         s.put_file('info', content, compress='nonexistent').wait()
#         assert False
#       except NotImplementedError:
#         pass

#     if "file://" in url:
#       delete_layer("/tmp/removeme/compress-" + str(TEST_NUMBER))

# @pytest.mark.parametrize("compression_method", ("gzip", "br"))
# def test_compress_level(compression_method):
#   filepath = "/tmp/removeme/compress_level-" + str(TEST_NUMBER)
#   url = "file://" + filepath

#   content = b'some_string' * 1000

#   compress_levels = range(1, 9, 2)
#   for compress_level in compress_levels:
#     with Storage(url, n_threads=5) as s:
#       s.put_file('info', content, compress=compression_method, compress_level=compress_level)
#       s.wait()

#       retrieved = s.get_file('info')
#       assert content == retrieved

#       _, e = s._interface.get_file("info")
#       assert e == compression_method

#       assert s.get_file('nonexistentfile') is None

#     delete_layer(filepath)


# def test_list():  
#   urls = [
#     "file:///tmp/removeme/list",
#     "gs://seunglab-test/cloudvolume/list",
#     "s3://seunglab-test/cloudvolume/list"
#   ]

#   for url in urls:
#     url = url + '-' + str(TEST_NUMBER)
#     with Storage(url, n_threads=5) as s:
#       print('testing service:', url)
#       content = b'some_string'
#       s.put_file('info1', content, compress=None)
#       s.put_file('info2', content, compress=None)
#       s.put_file('build/info3', content, compress=None)
#       s.put_file('level1/level2/info4', content, compress=None)
#       s.put_file('info5', content, compress='gzip')
#       s.put_file('info.txt', content, compress=None)
#       s.wait()
#       time.sleep(1) # sometimes it takes a moment for google to update the list
#       assert set(s.list_files(prefix='')) == set(['build/info3','info1', 'info2', 'level1/level2/info4', 'info5', 'info.txt'])
      
#       assert set(s.list_files(prefix='inf')) == set(['info1','info2','info5','info.txt'])
#       assert set(s.list_files(prefix='info1')) == set(['info1'])
#       assert set(s.list_files(prefix='build')) == set(['build/info3'])
#       assert set(s.list_files(prefix='build/')) == set(['build/info3'])
#       assert set(s.list_files(prefix='level1/')) == set(['level1/level2/info4'])
#       assert set(s.list_files(prefix='nofolder/')) == set([])

#       # Tests (1)
#       assert set(s.list_files(prefix='', flat=True)) == set(['info1','info2','info5','info.txt'])
#       assert set(s.list_files(prefix='inf', flat=True)) == set(['info1','info2','info5','info.txt'])
#       # Tests (2)
#       assert set(s.list_files(prefix='build', flat=True)) == set([])
#       # Tests (3)
#       assert set(s.list_files(prefix='level1/', flat=True)) == set([])
#       assert set(s.list_files(prefix='build/', flat=True)) == set(['build/info3'])
#       # Tests (4)
#       assert set(s.list_files(prefix='build/inf', flat=True)) == set(['build/info3'])

#       for file_path in ('info1', 'info2', 'build/info3', 'level1/level2/info4', 'info5', 'info.txt'):
#         s.delete_file(file_path)
  
#   delete_layer("/tmp/removeme/list")


# def test_exists():
#   urls = [
#     "file:///tmp/removeme/exists",
#     "gs://seunglab-test/cloudvolume/exists",
#     "s3://seunglab-test/cloudvolume/exists"
#   ]

#   for url in urls:
#     url = url + '-' + str(TEST_NUMBER)
#     with Storage(url, n_threads=5) as s:
#       content = b'some_string'
#       s.put_file('info', content, compress=None)
#       s.wait()
#       time.sleep(1) # sometimes it takes a moment for google to update the list
      
#       assert s.exists('info')
#       assert not s.exists('doesntexist')
#       s.delete_file('info')

# def test_access_non_cannonical_paths():
#   urls = [
#     "file:///tmp/noncanon",
#     "gs://seunglab-test/noncanon",
#     "s3://seunglab-test/noncanon"
#   ]

#   for url in urls:
#     url = url + '-' + str(TEST_NUMBER)
#     with Storage(url, n_threads=5) as s:
#       content = b'some_string'
#       s.put_file('info', content, compress=None)
#       s.wait()
#       time.sleep(0.5) # sometimes it takes a moment for google to update the list
      
#       assert s.get_file('info') == content
#       assert s.get_file('nonexistentfile') is None
#       s.delete_file('info')
#       s.wait()