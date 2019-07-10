import os
import re

from cloudvolume import paths
from cloudvolume.paths import strict_extract, extract, ExtractedPath

from cloudvolume.exceptions import UnsupportedProtocolError
from cloudvolume import lib

def test_path_extraction():
  extract = paths.extract(r'file://C:\wow\this\is\a\cool\path', windows=True, disable_toabs=True)
  print(extract)
  assert extract.protocol == 'file'
  assert extract.bucket == 'C:\\wow\\'

  # on linux the toabs prepends the current path because it
  # doesn't understand C:\... so we can't test that here.
  # assert extract.path == 'this\\is\\a\\cool\\path' 

  try:
    extract = paths.strict_extract(r'file://C:\wow\this\is\a\cool\path', windows=False, disable_toabs=True)
    assert False 
  except UnsupportedProtocolError:
    pass

  def shoulderror(url):
    try:
        paths.strict_extract(url)
        assert False, url
    except:
        pass

  def okgoogle(url):
    path = paths.strict_extract(url)
    assert path.protocol == 'gs', url
    assert path.bucket == 'bucket', url
    assert path.intermediate_path == '', url
    assert path.dataset == 'dataset', url
    assert path.layer == 'layer', url

  okgoogle('gs://bucket/dataset/layer') 
  shoulderror('s4://dataset/layer')
  shoulderror('dataset/layer')
  shoulderror('s3://dataset')

  firstdir = lambda x: '/' + x.split('/')[1]

  homepath = lib.toabs('~')
  homerintermediate = homepath.replace(firstdir(homepath), '')[1:]

  curpath = lib.toabs('.')
  curintermediate = curpath.replace(firstdir(curpath), '')[1:]

  print(curintermediate)

  assert (paths.extract('s3://seunglab-test/intermediate/path/dataset/layer') 
      == ExtractedPath(
        'precomputed', 's3', 'seunglab-test', 
        'intermediate/path/dataset/layer', 'intermediate/path/', 
        'dataset', 'layer'
      ))

  assert (paths.extract('file:///tmp/dataset/layer') 
      == ExtractedPath('precomputed', 'file', "/tmp", 'dataset/layer', '', 'dataset', 'layer'))

  assert (paths.extract('file://seunglab-test/intermediate/path/dataset/layer') 
      == ExtractedPath(
        'precomputed', 'file', firstdir(curpath), 
        os.path.join(curintermediate, 'seunglab-test/intermediate/path/dataset/layer'),   
        os.path.join(curintermediate, 'seunglab-test', 'intermediate/path/'), 
       'dataset', 'layer'))

  assert (paths.extract('gs://seunglab-test/intermediate/path/dataset/layer') 
      == ExtractedPath(
        'precomputed', 'gs', 
        'seunglab-test', 'intermediate/path/dataset/layer', 
        'intermediate/path/', 'dataset', 'layer'
      ))

  assert (paths.extract('file://~/seunglab-test/intermediate/path/dataset/layer') 
      == ExtractedPath(
        'precomputed', 'file', 
        firstdir(homepath), 
        os.path.join(homerintermediate, 'seunglab-test', 'intermediate/path/dataset/layer'),
        os.path.join(homerintermediate, 'seunglab-test', 'intermediate/path/'),  
        'dataset', 
        'layer'
      )
  )

  assert (paths.extract('file:///User/me/.cloudvolume/cache/gs/bucket/dataset/layer') 
      == ExtractedPath(
        'precomputed', 'file', 
        '/User', 'me/.cloudvolume/cache/gs/bucket/dataset/layer', 
        'me/.cloudvolume/cache/gs/bucket/', 'dataset', 'layer'
      ))

  shoulderror('s3://dataset/layer/')

  shoulderror('ou3bouqjsa fkj aojsf oaojf ojsaf')

  okgoogle('gs://bucket/dataset/layer/')
  shoulderror('gs://bucket/dataset/layer/info')

  path = paths.extract('s3://bucketxxxxxx/datasetzzzzz91h8__3/layer1br9bobasjf/')
  assert path.format == 'precomputed'
  assert path.protocol == 's3'
  assert path.bucket == 'bucketxxxxxx'
  assert path.dataset == 'datasetzzzzz91h8__3'
  assert path.layer == 'layer1br9bobasjf'

  path = paths.extract('file:///bucket/dataset/layer/')
  assert path.format == 'precomputed'
  assert path.protocol == 'file'
  assert path.bucket == '/bucket'
  assert path.dataset == 'dataset'
  assert path.layer == 'layer'

  shoulderror('lucifer://bucket/dataset/layer/')
  shoulderror('gs://///')
  shoulderror('gs://seunglab-test//segmentation')

  path = paths.extract('file:///tmp/removeme/layer/')
  assert path.format == 'precomputed'
  assert path.protocol == 'file'
  assert path.bucket == '/tmp'
  assert path.dataset == 'removeme'
  assert path.layer == 'layer'
