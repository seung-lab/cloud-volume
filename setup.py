import os
import setuptools

import platform

requires = ['pbr', 'numpy']

if platform.system() in ('Linux', 'Darwin'):
  requires.append('posix_ipc')

setuptools.setup(
  setup_requires=requires,
  extras_require={
    ':python_version == "2.7"': ['futures'],
    ':python_version == "2.6"': ['futures'],
  },
  package_data={
    'cloudvolume': [
      '../ext/microviewer/*',
    ],
  },
  long_description_content_type="text/markdown",
  pbr=True)

