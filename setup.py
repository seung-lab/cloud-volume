import os
import setuptools

setuptools.setup(
  setup_requires=['pbr', 'numpy'],
  extras_require={
    ':python_version == "2.7"': ['futures'],
    ':python_version == "2.6"': ['futures'],
  },
  package_data={
    'cloudvolume': [
      './microviewer/*',
    ],
  },
  long_description_content_type="text/markdown",
  pbr=True)

