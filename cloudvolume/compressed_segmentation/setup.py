#!/usr/bin/env python
from __future__ import print_function
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
  ext_modules=cythonize([
    Extension("compress_segmentation", 
      sources=[ "compress_segmentation.pyx" ],
      include_dirs=[ np.get_include() ],
      language='c++',
      extra_compile_args=['-std=c++11','-O3', '-Wall']
    )
  ])
)


