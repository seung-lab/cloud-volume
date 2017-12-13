import setuptools
from distutils.command.install_data import install_data

def CustomBuildExt(build_ext):
    """build_ext command for use when numpy headers are needed."""
    def run(self):
        # Import numpy here, only when headers are needed
        import numpy as np

        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())

        # Call original build_ext command
        build_ext.run(self)


setuptools.setup(
  setup_requires=[ 'pbr', 'numpy' ],
  pbr=True,
  cmdclass = { 'build_ext': CustomBuildExt },
  ext_modules=[
    setuptools.Extension("compress_segmentation", 
      sources=[ "./cloudvolume/compressed_segmentation/compress_segmentation.cpp" ],

      # need to wait for numpy to be installed first hence CustomBuildExt
      # include_dirs=[ np.get_include() ], 

      language='c++',
      extra_compile_args=[ '-std=c++11','-O3' ],
    )
  ],
)