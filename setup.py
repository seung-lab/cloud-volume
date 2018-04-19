import os
import setuptools
from setuptools.command.build_ext import build_ext as _build_ext

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

compresso_cpp = './ext/third_party/compresso/c++/'
compresso_py = './ext/third_party/compresso/python/'

setuptools.setup(
    cmdclass={'build_ext':build_ext},
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
