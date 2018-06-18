/* fpzip_emscripten.cpp
 *
 * This is a C++ fpzip interface that can be compiled 
 * into Javascript asm for use with web apps. It supports
 * decompressing fpzip files and modified "kempressed" fpzip 
 * files that store only 0-1 values, have 2.0f added to them,
 * and have their axes permuted from XYZC to XYCZ.
 *
 * fpzip was originally authored by Peter Lindstrom et al at LLNL.
 *
 * Author: William Silversmith
 * Affiliation: Seung Lab, Princeton Neuroscience Institute
 * Date: June 2018
 */

/*
Debugging compilation:

emcc --bind -g4 -std=c++11 -DFPZIP_FP=FPZIP_FP_FAST -DFPZIP_BLOCK_SIZE=0x1000 -DWITH_UNION \
  -fPIC -Iinc src/read.cpp src/fpzip_emscripten.cpp src/rcdecoder.cpp src/rcqsmodel.cpp \
  -s EXPORTED_FUNCTIONS="[ '_decompress', '_dekempress' ]" -s SIMD=0 -o fpzip.js

Production compilation:

emcc -O2 -s SIMD=1 --bind -std=c++11 -DFPZIP_FP=FPZIP_FP_FAST -DFPZIP_BLOCK_SIZE=0x1000 -DWITH_UNION \
  -fPIC -Iinc src/read.cpp src/fpzip_emscripten.cpp src/rcdecoder.cpp src/rcqsmodel.cpp \
  -s EXPORTED_FUNCTIONS="[ '_decompress', '_dekempress' ]" -o fpzip.js
*/

#include <stdio.h>
#include <stdlib.h>
#include "fpzip.h"
#include "read.h"

#include <emscripten/bind.h>

using namespace emscripten;

/* PROTOTYPES */

struct DecodedImage {
  int type;
  int prec;
  int nx;
  int ny;
  int nz;
  int nf;
  int nbytes; // nx * ny * nz * nf * (type + 1) * sizeof(float)
  void* data;
};

DecodedImage* decompress(void* buffer);
DecodedImage* dekempress(void* buffer);

/* IMPLEMENTATION */

template <typename T>
DecodedImage* dekempress_algo(DecodedImage *di) {
  int nx = di->nx;
  int ny = di->ny;
  int nz = di->nz;
  int nf = di->nf; 

  T *data = (T*)di->data;
  const int nbytes = di->nbytes;
  for (int i = 0; i < nbytes; i++) {
    data[i] -= 2.0;
  }

  T *dekempressed = (T*)calloc(nx * ny * nz * nf, sizeof(T));
  T *src;
  T *dest;

  const int xysize = nx * ny;
  int offset = 0;
  
  for (int channel = 0; channel < nf; channel++) {
    offset = nx * ny * nz * channel;

    for (int z = 0; z < nz; z++) {
      src = &data[ z * xysize * (nf + channel) ];
      dest = &dekempressed[ z * xysize + offset ];
      memcpy(dest, src, xysize * sizeof(T)); 
    }  
  }

  free(di->data);
  di->data = (void*)dekempressed;

  return di;
}

/* fpzip decompression + dekempression.
 *  
 * 1) fpzip decompress
 * 2) Subtract 2.0 from all elements.  
 * 3) XYCZ -> XYZC
 * 
 * Example:
 * DecodedImage *di = dekempress(buffer);
 * float* img = (float*)di->data;
 */
DecodedImage* dekempress(void *buffer) {
  DecodedImage* di = decompress(buffer);

  if (di->type == FPZIP_TYPE_FLOAT) {
    return dekempress_algo<float>(di);
  }

  return dekempress_algo<double>(di);
}

/* Standard fpzip decompression. 
 * 
 * Example:
 * DecodedImage *di = decompress(buffer);
 * float* img = (float*)di->data;
 */
DecodedImage* decompress(void *buffer) {
  int type = FPZIP_TYPE_FLOAT;
  int prec = 0;
  int nx = 1;
  int ny = 1;
  int nz = 1;
  int nf = 1;

  char *errorstr;
  
  FPZ* fpz = fpzip_read_from_buffer(buffer);
  // read header
  if (!fpzip_read_header(fpz)) {
    sprintf(errorstr, "cannot read header: %s\n", fpzip_errstr[fpzip_errno]);
    throw errorstr;
  }
  type = fpz->type;
  prec = fpz->prec;
  nx = fpz->nx;
  ny = fpz->ny;
  nz = fpz->nz;
  nf = fpz->nf;

  size_t count = (size_t)nx * ny * nz * nf;
  size_t size = (type == FPZIP_TYPE_FLOAT ? sizeof(float) : sizeof(double));
  void *data = (type == FPZIP_TYPE_FLOAT 
    ? static_cast<void*>(new float[count]) 
    : static_cast<void*>(new double[count])
  );
  
  // perform actual decompression
  if (!fpzip_read(fpz, data)) {
    sprintf(errorstr, "decompression failed: %s\n", fpzip_errstr[fpzip_errno]);
    throw errorstr;
  }
  fpzip_read_close(fpz);

  free(buffer);

  DecodedImage *res = new DecodedImage();
  res->type = type;
  res->prec = prec;
  res->nx = nx;
  res->ny = ny;
  res->nz = nz;
  res->nf = nf;
  res->nbytes = (int)count * (type + 1) * (int)sizeof(float);
  res->data = data;

  return res;
}

EMSCRIPTEN_BINDINGS(my_module) {
    function("decompress", &decompress, allow_raw_pointers());
    function("dekempress", &dekempress, allow_raw_pointers());
}

