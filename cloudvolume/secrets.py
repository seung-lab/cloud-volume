from __future__ import print_function

import os
import json

from google.oauth2 import service_account

from .lib import mkdir, colorize

backwards_compatible_path = os.path.join(os.environ['HOME'], '.neuroglancer/')
new_path = os.path.join(os.environ['HOME'], '.cloudvolume/')

if os.path.exists(new_path):
  CLOUD_VOLUME_DIR = new_path
elif os.path.exists(backwards_compatible_path):
  CLOUD_VOLUME_DIR = backwards_compatible_path
  print(colorize('yellow', 'Deprecation Warning: Directory ~/.cloudvolume is now preferred to ~/.neuroglancer.\nConsider running: mv ~/.neuroglancer ~/.cloudvolume'))
else:
  CLOUD_VOLUME_DIR = mkdir(new_path)

secret_path = mkdir(os.path.join(CLOUD_VOLUME_DIR, 'secrets/'))

PROJECT_NAME = None
project_name_path = os.path.join(CLOUD_VOLUME_DIR, 'project_name')
if os.path.exists(project_name_path):
  with open(project_name_path, 'r') as f:
    PROJECT_NAME = f.read()

google_credentials_path = os.path.join(secret_path, 'google-secret.json')
if os.path.exists(google_credentials_path):
  google_credentials = service_account.Credentials \
    .from_service_account_file(google_credentials_path)
else:
  google_credentials = ''

aws_credentials_path = os.path.join(secret_path, 'aws-secret.json')
if os.path.exists(aws_credentials_path):
  with open(aws_credentials_path, 'r') as f:
    aws_credentials = json.loads(f.read())
else:
  aws_credentials = ''

boss_credentials_path = os.path.join(secret_path, 'boss-secret.json')
if os.path.exists(boss_credentials_path):
  with open(boss_credentials_path, 'r') as f:
    boss_credentials = json.loads(f.read())
else:
  boss_credentials = ''
