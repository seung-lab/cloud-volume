## example:
## http://stackoverflow.com/questions/16792792/project-organization-with-cython-and-c

from distutils.core import setup, Extension
import numpy as np

setup(
    ext_modules=[
      Extension(
          'compresso',
          include_dirs=[np.get_include(), '../c++/'],
          sources=['compresso.pyx'],
          extra_compile_args=['-O3', '-std=c++11'],
          language='c++'
      )
  ]
)
