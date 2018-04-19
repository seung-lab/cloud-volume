import os
import setuptools

import numpy as np

compresso_cpp = './ext/third_party/compresso/c++/'
compresso_py = './ext/third_party/compresso/python/'

setuptools.setup(
    setup_requires=['pbr', 'numpy'],
    pbr=True,
    ext_modules=[
      setuptools.Extension(
          'compresso',
          include_dirs=[np.get_include(), compresso_cpp ],
          sources=[ os.path.join(compresso_py, 'compresso.pyx') ],
          extra_compile_args=['-O3', '-std=c++11'],
          language='c++'
      )
  ]
)
