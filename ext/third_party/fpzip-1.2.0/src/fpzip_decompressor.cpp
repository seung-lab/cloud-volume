#include <stdio.h>
#include <stdlib.h>
#include "fpzip.h"
#include "read.h"

extern "C" {

void* decompress(void *buffer) {
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

  return (void*)data;
}

int main() {
  void *data = static_cast<void*>(new float[100]);
  decompress(data);
  return 0;
}

}

/*
emcc -O2 -g4 -std=c++11 -DFPZIP_FP=FPZIP_FP_FAST -DFPZIP_BLOCK_SIZE=0x1000 -DWITH_UNION -fPIC -Iinc src/read.cpp src/fpzip_decompressor.cpp src/rcdecoder.cpp src/rcqsmodel.cpp -s EXPORTED_FUNCTIONS="[ '_decompress', '_main' ]" -s EXTRA_EXPORTED_RUNTIME_METHODS="[ 'cwrap', 'getValue' ]" -o decompress.js
*/