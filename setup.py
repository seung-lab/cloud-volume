import os
import setuptools

join = os.path.join
third_party_dir = './ext/third_party'
compressedsegdir = join(third_party_dir, 'compressed_segmentation')

# NOTE: Run if _compressed_segmentation.cpp does not exist:
# cython -3 --fast-fail -v --cplus \
#    -I./ext/third_party/compressed_segmentation/include \
#    ./ext/third_party/compressed_segmentation/src/_compressed_segmentation.pyx

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
        '_compressed_segmentation',
        optional=True,
        sources=[ join(compressedsegdir, 'src', x) for x in ( 
            'compress_segmentation.cc', 'decompress_segmentation.cc',
            '_compressed_segmentation.cpp'
        )],
        language='c++',
        include_dirs=[ join(compressedsegdir, 'include'), np.get_include() ],
        extra_compile_args=[
          '-O3', '-std=c++11'
        ],
    ),
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
    + "WARNING: Several C/C++ extensions to cloud-volume require numpy C++ headers to install. " \
    + "cloud-volume installs numpy automatically. Please rerun cloud-volume installation to access the following packages:")

  print("Accelerated compressed_segmentation Compression (a pure python implementation is available)")

  try:
    import fpzip
  except ImportError:
    print("fpzip Floating Point Compression (you can also run `pip install fpzip`)")
  print(reset)





