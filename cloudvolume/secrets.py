import os
import json

from google.oauth2 import service_account

from lib import mkdir

ngpath = mkdir(os.path.join(os.environ['HOME'], '.neuroglancer/'))
secret_path = mkdir(os.path.join(ngpath, 'secrets/'))

PROJECT_NAME = None
project_name_path = os.path.join(ngpath, 'project_name')
if os.path.exists(project_name_path):
  with open(project_name_path) as f:
    PROJECT_NAME = f.read()

google_credentials_path = os.path.join(secret_path, 'google-secret.json')
if os.path.exists(google_credentials_path):
  google_credentials = service_account.Credentials \
    .from_service_account_file(google_credentials_path)
else:
  google_credentials = None

aws_credentials_path = os.path.join(secret_path, 'aws-secret.json')
if os.path.exists(aws_credentials_path):
  with open(aws_credentials_path, 'rb') as f:
    aws_credentials = json.loads(f.read())
else:
  aws_credentials = None

