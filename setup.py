import os
import setuptools

join = os.path.join
third_party_dir = './ext/third_party'
fpzipdir = join(third_party_dir, 'fpzip-1.2.0')
ofiles = 'build/temp.linux-x86_64-3.4/ext/third_party/fpzip-1.2.0/src/'

# NOTE: You must run 
# cython -3 --fast-fail -v --cplus ./ext/src/third_party/fpzip-1.2.0/src/fpzip.pyx
# if fpzip.cpp does not exist.

os.system("pip install numpy")
# os.system("cd ./ext/src/third_party/fpzip-1.2.0/src; make")

# g++ -ansi -Wall -g -O3 -DFPZIP_BLOCK_SIZE=0x1000 -DFPZIP_FP=FPZIP_FP_FAST -DWITH_UNION -I../inc -c error.cpp
# g++ -ansi -Wall -g -O3 -DFPZIP_BLOCK_SIZE=0x1000 -DFPZIP_FP=FPZIP_FP_FAST -DWITH_UNION -I../inc -c rcdecoder.cpp
# g++ -ansi -Wall -g -O3 -DFPZIP_BLOCK_SIZE=0x1000 -DFPZIP_FP=FPZIP_FP_FAST -DWITH_UNION -I../inc -c rcencoder.cpp
# g++ -ansi -Wall -g -O3 -DFPZIP_BLOCK_SIZE=0x1000 -DFPZIP_FP=FPZIP_FP_FAST -DWITH_UNION -I../inc -c rcqsmodel.cpp
# g++ -ansi -Wall -g -O3 -DFPZIP_BLOCK_SIZE=0x1000 -DFPZIP_FP=FPZIP_FP_FAST -DWITH_UNION -I../inc -c read.cpp
# g++ -ansi -Wall -g -O3 -DFPZIP_BLOCK_SIZE=0x1000 -DFPZIP_FP=FPZIP_FP_FAST -DWITH_UNION -I../inc -c write.cpp
# ar rc ../lib/libfpzip.a error.o rcdecoder.o rcencoder.o rcqsmodel.o read.o write.o

# g++ -std=c++98 -O3 -Wall -pedantic -I../inc -L../lib fpzip.cpp -lfpzip -o fpzip
# g++ -std=c++98 -O3 -Wall -pedantic -L../lib testfpzip.o -lfpzip -o testfpzip


setuptools.setup(
    setup_requires=['pbr'],
    extras_require={
    	':python_version == "2.7"': ['futures'],
    	':python_version == "2.6"': ['futures'],
    },
    ext_modules=[
        setuptools.Extension(
            'cloudvolume.fpzip',
            optional=True,
            sources=[ join(fpzipdir, 'src', x) for x in ( 
            	'error.cpp', 'rcdecoder.cpp', 'rcencoder.cpp', 
            	'rcqsmodel.cpp', 'write.cpp', 'read.cpp', 'fpzip.cpp'
            )],
            # depends=[ join(fpzipdir, x) for x in (
            	
            # ) ],
            language='c++',
            include_dirs=[ join(fpzipdir, 'inc') ],
            extra_compile_args=[
              '-std=c++11', 
              '-DFPZIP_FP=FPZIP_FP_FAST', '-DFPZIP_BLOCK_SIZE=0x1000', '-DWITH_UNION',
              # '-lfpzip', '-Lext/third_party/fpzip-1.2.0/lib/'
            ]) #don't use  '-fvisibility=hidden', python can't see init module
    ],
    pbr=True)

# 'error.cpp', 'rcdecoder.cpp', 'rcencoder.cpp', 'rcqsmodel.cpp', 'read.cpp', 'write.cpp' 

# g++ -std=c++11 -fPIC -shared -DFPZIP_BLOCK_SIZE=0x1000 -DFPZIP_FP=FPZIP_FP_FAST -DWITH_UNION -I/usr/include/python3.4m -I/usr/people/ws9/.virtualenvs/cv/include/python3.4m -I../inc ../src/error.cpp ../src/read.cpp ../src/write.cpp ../src/rcdecoder.cpp ../src/rcencoder.cpp ../src/rcqsmodel.cpp ../src/fpzip.cpp -o _fpzip.cpython-34m.so