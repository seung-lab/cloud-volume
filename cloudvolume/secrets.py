from __future__ import print_function

from collections import defaultdict
import os
import json

from google.oauth2 import service_account

from .lib import mkdir, colorize

CLOUD_VOLUME_DIR = mkdir(os.path.join(os.environ['HOME'], '.cloudvolume'))

def secretpath(filepath):
  preferred = os.path.join(CLOUD_VOLUME_DIR, filepath)
  
  if os.path.exists(preferred):
    return preferred

  backcompat = [
    os.path.join(os.environ['HOME'], '.neuroglancer'), # older
    '/' # original
  ]

  backcompat = [ os.path.join(path, filepath) for path in backcompat ] 

  for path in backcompat:
    if os.path.exists(path):
      print(colorize('yellow', 'Deprecation Warning: {} is now preferred to {}.'.format(preferred, path)))  
      return path

  return preferred

project_name_paths = [ 
  secretpath('secrets/project_name'),
  secretpath('project_name')
]

def default_google_project_name():
  if 'GOOGLE_PROJECT_NAME' in os.environ: 
    return os.environ['GOOGLE_PROJECT_NAME']
  else: 
    for path in project_name_paths:
      if os.path.exists(path):
        with open(path, 'r') as f:
          return f.read().strip()

  default_credentials_path = secretpath('secrets/google-secret.json')
  if os.path.exists(default_credentials_path):
    with open(default_credentials_path, 'rt') as f:
      return json.loads(f.read())['project_id']

  return None

PROJECT_NAME = default_google_project_name()
GOOGLE_CREDENTIALS_CACHE = {}
google_credentials_path = secretpath('secrets/google-secret.json')

def google_credentials(bucket = ''):
  global PROJECT_NAME
  global GOOGLE_CREDENTIALS_CACHE

  if bucket in GOOGLE_CREDENTIALS_CACHE.keys():
    return GOOGLE_CREDENTIALS_CACHE[bucket]

  paths = [
    secretpath('secrets/google-secret.json')
  ]

  if bucket:
    paths = [ secretpath('secrets/{}-google-secret.json'.format(bucket)) ] + paths

  google_credentials = None
  project_name = PROJECT_NAME
  for google_credentials_path in paths:
    if os.path.exists(google_credentials_path):
      google_credentials = service_account.Credentials \
        .from_service_account_file(google_credentials_path)
      
      with open(google_credentials_path, 'rt') as f:
        project_name = json.loads(f.read())['project_id']
      break

  GOOGLE_CREDENTIALS_CACHE[bucket] = (project_name, google_credentials)

  if google_credentials == None:
    print(colorize('yellow', 'Using default Google credentials. There is no ~/.cloudvolume/secrets/google-secret.json set.'))  
  return project_name, google_credentials

AWS_CREDENTIALS_CACHE = defaultdict(dict)
aws_credentials_path = secretpath('secrets/aws-secret.json')
def aws_credentials(bucket = '', service = 'aws'):
  global AWS_CREDENTIALS_CACHE

  if service == 's3':
    service = 'aws'

  if bucket in AWS_CREDENTIALS_CACHE.keys():
    return AWS_CREDENTIALS_CACHE[bucket]

  default_file_path = 'secrets/{}-secret.json'.format(service)

  paths = [
    secretpath(default_file_path)
  ]

  if bucket:
    paths = [ secretpath('secrets/{}-{}-secret.json'.format(bucket, service)) ] + paths

  aws_credentials = ''
  aws_credentials_path = secretpath(default_file_path)
  for aws_credentials_path in paths:
    if os.path.exists(aws_credentials_path):
      with open(aws_credentials_path, 'r') as f:
        aws_credentials = json.loads(f.read())
      break

  AWS_CREDENTIALS_CACHE[service][bucket] = aws_credentials
  return aws_credentials
    

boss_credentials_path = secretpath('secrets/boss-secret.json')
if os.path.exists(boss_credentials_path):
  with open(boss_credentials_path, 'r') as f:
    boss_credentials = json.loads(f.read())
else:
  boss_credentials = ''
