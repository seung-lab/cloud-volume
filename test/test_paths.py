import os
import re

from cloudvolume import paths
from cloudvolume.paths import strict_extract, extract, ExtractedPath

from cloudvolume.exceptions import UnsupportedProtocolError
from cloudvolume import lib

def test_path_extraction():

  def shoulderror(url):
    try:
        paths.strict_extract(url)
        assert False, url
    except:
        pass

  def okgoogle(url):
    path = paths.extract(url)
    assert path.protocol == 'gs', url
    assert path.bucket == 'bucket', url
    assert path.basepath == 'bucket/dataset', url
    assert path.no_bucket_basepath == 'dataset', url
    assert path.dataset == 'dataset', url
    assert path.layer == 'layer', url

  okgoogle('gs://bucket/dataset/layer') 
  shoulderror('s4://dataset/layer')
  shoulderror('dataset/layer')
  shoulderror('s3://dataset')

  # don't error
  assert (strict_extract('graphene://http://localhost:8080/segmentation/1.0/testvol')
    == ExtractedPath(
      'graphene', 'http', 'localhost:8080', 
      'localhost:8080/segmentation/1.0', 'segmentation/1.0', '1.0', 'testvol'))

  assert (strict_extract('precomputed://gs://fafb-ffn1-1234567/segmentation')
    == ExtractedPath(
      'precomputed', 'gs', 'fafb-ffn1-1234567', 
      'fafb-ffn1-1234567', '', 'fafb-ffn1-1234567', 'segmentation'))

  firstdir = lambda x: '/' + x.split('/')[1]

  homepath = lib.toabs('~')
  homerintermediate = homepath.replace(firstdir(homepath), '')[1:]

  curpath = lib.toabs('.')
  curintermediate = curpath.replace(firstdir(curpath), '')[1:]
  
  match = re.match(r'((?:(?:\w:\\\\)|/).+?)\b', lib.toabs('.'))
  bucket, = match.groups()
  
  print(bucket, curintermediate)

  assert (paths.extract('s3://seunglab-test/intermediate/path/dataset/layer') 
      == ExtractedPath(
        'precomputed', 's3', 'seunglab-test', 
        'seunglab-test/intermediate/path/dataset', 'intermediate/path/dataset', 
        'dataset', 'layer'
      ))

  assert (paths.extract('file:///tmp/dataset/layer') 
      == ExtractedPath(
        'precomputed', 'file', "/tmp", '/tmp/dataset', 'dataset', 'dataset', 'layer'
      ))

  assert (paths.extract('file://seunglab-test/intermediate/path/dataset/layer') 
      == ExtractedPath(
        'precomputed', 'file', firstdir(curpath), 
        os.path.join(bucket, curintermediate, 'seunglab-test/intermediate/path/dataset'),   
        os.path.join(curintermediate, 'seunglab-test', 'intermediate/path/dataset'), 
       'dataset', 'layer'))

  assert (paths.extract('gs://seunglab-test/intermediate/path/dataset/layer') 
      == ExtractedPath(
        'precomputed', 'gs', 'seunglab-test',
        'seunglab-test/intermediate/path/dataset', 'intermediate/path/dataset', 
        'dataset', 'layer'
      ))

  assert (paths.extract('file://~/seunglab-test/intermediate/path/dataset/layer') 
      == ExtractedPath(
        'precomputed', 'file', firstdir(homepath), 
        os.path.join(bucket, homerintermediate, 'seunglab-test', 'intermediate/path/dataset'),
        os.path.join(homerintermediate, 'seunglab-test', 'intermediate/path/dataset'),  
        'dataset', 
        'layer'
      )
  )

  assert (paths.extract('file:///User/me/.cloudvolume/cache/gs/bucket/dataset/layer') 
      == ExtractedPath(
        'precomputed', 'file', '/User', 
        '/User/me/.cloudvolume/cache/gs/bucket/dataset', 
        'me/.cloudvolume/cache/gs/bucket/dataset', 'dataset', 'layer'
      ))

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


def test_windows_path_extraction():
  extract = paths.extract(r'file://C:\wow\this\is\a\cool\path', windows=True, disable_toabs=True)
  assert extract.format == 'precomputed'
  assert extract.protocol == 'file'
  assert extract.bucket == 'C:\\wow'
  assert extract.basepath == 'C:\\wow\\this\\is\\a\\cool'
  assert extract.no_bucket_basepath == 'this\\is\\a\\cool'  
  assert extract.dataset == 'cool'  
  assert extract.layer == 'path'  

  extract = paths.extract('file://C:\\wow\\this\\is\\a\\cool\\path\\', windows=True, disable_toabs=True)
  assert extract.format == 'precomputed'
  assert extract.protocol == 'file'
  assert extract.bucket == 'C:\\wow'
  assert extract.basepath == 'C:\\wow\\this\\is\\a\\cool'
  assert extract.no_bucket_basepath == 'this\\is\\a\\cool'  
  assert extract.dataset == 'cool'  
  assert extract.layer == 'path'  

  extract = paths.extract('precomputed://https://storage.googleapis.com/neuroglancer-public-data/kasthuri2011/ground_truth', windows=True)
  assert extract.format == 'precomputed'
  assert extract.protocol == 'https'
  assert extract.bucket == 'storage.googleapis.com'
  assert extract.basepath == 'storage.googleapis.com/neuroglancer-public-data/kasthuri2011'
  assert extract.no_bucket_basepath == 'neuroglancer-public-data/kasthuri2011'  
  assert extract.dataset == 'kasthuri2011'  
  assert extract.layer == 'ground_truth'  
