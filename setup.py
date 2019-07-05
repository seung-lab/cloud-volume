import os
import setuptools
import sys

def read(fname):
  with open(os.path.join(os.path.dirname(__file__), fname), 'rt') as f:
    return f.read()

def requirements():
  with open(os.path.join(os.path.dirname(__file__), 'requirements.txt'), 'rt') as f:
    return f.readlines()

setuptools.setup(
  name="cloud-volume",
  version="0.52.3",
  setup_requires=['numpy'],
  install_requires=requirements(),
  extras_require={
    "boss": "intern>=0.9.11",
    ':sys_platform!="win32"': [
      "posix_ipc==1.0.4",
      "psutil==5.4.3",
    ],
  },
  author="William Silversmith, Nico Kemnitz, Ignacio Tartavull, and others",
  author_email="ws9@princeton.edu",
  packages=[ 
    'cloudvolume', 
    'cloudvolume.datasource',
    'cloudvolume.datasource.boss',
    'cloudvolume.datasource.precomputed',
    'cloudvolume.storage', 
  ],
  package_data={
    'cloudvolume': [
      './microviewer/*',
      'LICENSE',
    ],
  },
  description="A serverless client for reading and writing Neuroglancer Precomputed volumes both locally and on cloud services.",
  long_description=read('README.md'),
  long_description_content_type="text/markdown",
  license = "BSD 3-Clause",
  keywords = "neuroglancer volumetric-data numpy connectomics microscopy image-processing biomedical-image-processing s3 gcs mesh meshes skeleton skeletons",
  url = "https://github.com/seung-lab/cloud-volume/",
  classifiers=[
    "Intended Audience :: Developers",
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: BSD License",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Topic :: Scientific/Engineering",
    "Intended Audience :: Science/Research",
    "Operating System :: POSIX",
    "Operating System :: MacOS",
    "Topic :: Utilities",
  ],
)
