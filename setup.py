import os
import setuptools

join = os.path.join
third_party_dir = './ext/third_party'
fpzipdir = join(third_party_dir, 'fpzip-1.2.0')

# NOTE: You must run 
# cython -3 --fast-fail -v --cplus ./ext/src/third_party/fpzip-1.2.0/src/fpzip.pyx
# if fpzip.cpp does not exist.

try:
  import numpy as np
except ImportError:
  np = None

setup_requires = ['pbr']

extensions = []
if np:
  setup_requires.append('numpy')
  extensions = [
    setuptools.Extension(
      'fpzip',
      optional=True,
      sources=[ join(fpzipdir, 'src', x) for x in ( 
        'error.cpp', 'rcdecoder.cpp', 'rcencoder.cpp', 
        'rcqsmodel.cpp', 'write.cpp', 'read.cpp', 'fpzip.cpp'
      )],
      language='c++',
      include_dirs=[ join(fpzipdir, 'inc'), np.get_include() ],
      extra_compile_args=[
        '-std=c++11', 
        '-DFPZIP_FP=FPZIP_FP_FAST', '-DFPZIP_BLOCK_SIZE=0x1000', '-DWITH_UNION',
      ]
    )
  ]

setuptools.setup(
  setup_requires=setup_requires,
  extras_require={
    ':python_version == "2.7"': ['futures'],
    ':python_version == "2.6"': ['futures'],
  },
  ext_modules=extensions,
  pbr=True)


if not np:
  yellow = "\033[1;93m"
  reset = "\033[m"
  print(yellow \
    + "WARNING: the fpzip extension has not been compiled. " \
    + "Please reinstall after running \"pip install numpy\"." \
    + reset)



