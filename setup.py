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
  version="4.1.0",
  setup_requires=[
    'numpy<1.17; python_version<"3.5"',
    'numpy; python_version>="3.5"',
  ],
  python_requires="~=3.6", # >= 3.6 < 4.0
  install_requires=requirements(),
  # Environment Marker Examples:
  # https://www.python.org/dev/peps/pep-0496/
  extras_require={
    "boss": [
      "intern>=0.9.11",
      "blosc==1.8.3",
    ],
    ':sys_platform!="win32"': [
      "posix_ipc>=1.0.4",
      "psutil==5.4.3",
    ],
    "mesh_viewer": [ 'vtk' ],
    "skeleton_viewer": [ 'matplotlib' ],
    "all_viewers": [ 'vtk', 'matplotlib' ],
    "dask": [ 'dask[array]' ],
    "test": [ "pytest", "pytest-cov", "codecov", "requests_mock", "scipy"]
  },
  author="William Silversmith, Nico Kemnitz, Ignacio Tartavull, and others",
  author_email="ws9@princeton.edu",
  packages=setuptools.find_packages(),
  package_data={
    'cloudvolume': [
      './microviewer/*',
      'LICENSE',
      'requirements.txt',
    ],
  },
  description="A serverless client for reading and writing Neuroglancer Precomputed volumes both locally and on cloud services.",
  long_description=read('README.md'),
  long_description_content_type="text/markdown",
  license = "License :: OSI Approved :: BSD License",
  keywords = "neuroglancer volumetric-data numpy connectomics microscopy image-processing biomedical-image-processing s3 gcs mesh meshes skeleton skeletons",
  url = "https://github.com/seung-lab/cloud-volume/",
  classifiers=[
    "Intended Audience :: Developers",
    "Development Status :: 5 - Production/Stable",
    "License :: OSI Approved :: BSD License",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Topic :: Scientific/Engineering",
    "Intended Audience :: Science/Research",
    "Operating System :: POSIX",
    "Operating System :: MacOS",
    "Operating System :: Microsoft :: Windows :: Windows 10",
    "Topic :: Utilities",
  ],
)
