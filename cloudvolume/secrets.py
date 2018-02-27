from __future__ import print_function

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

PROJECT_NAME = None 
project_name_paths = [ 
  secretpath('secrets/project_name'),
  secretpath('project_name')
]

google_credentials_path = secretpath('secrets/google-secret.json')
if os.path.exists(google_credentials_path):
  google_credentials = service_account.Credentials \
    .from_service_account_file(google_credentials_path)
else:
  google_credentials = ''


if 'GOOGLE_PROJECT_NAME' in os.environ: 
  PROJECT_NAME = os.environ['GOOGLE_PROJECT_NAME']
else: 
  for path in project_name_paths:
    if os.path.exists(path):
      with open(path, 'r') as f:
        PROJECT_NAME = f.read().strip()
      break

if not PROJECT_NAME and google_credentials_path:
  if os.path.exists(google_credentials_path):
    with open(google_credentials_path, 'rt') as f:
      PROJECT_NAME = json.loads(f.read())['project_id']

aws_credentials_path = secretpath('secrets/aws-secret.json')
if os.path.exists(aws_credentials_path):
  with open(aws_credentials_path, 'r') as f:
    aws_credentials = json.loads(f.read())
else:
  aws_credentials = ''

boss_credentials_path = secretpath('secrets/boss-secret.json')
if os.path.exists(boss_credentials_path):
  with open(boss_credentials_path, 'r') as f:
    boss_credentials = json.loads(f.read())
else:
  boss_credentials = ''
