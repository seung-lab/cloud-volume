#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "fpzip.h"

static int
usage()
{
  fprintf(stderr, "fpzip version %d.%d.%d\n", FPZIP_VERSION_MAJOR, FPZIP_VERSION_MINOR, FPZIP_VERSION_PATCH);
  fprintf(stderr, "Usage: fpzip [options] [<infile] [>outfile]\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -d : decompress\n");
  fprintf(stderr, "  -q : quiet mode\n");
  fprintf(stderr, "  -i <path> : input file (default=stdin)\n");
  fprintf(stderr, "  -o <path> : output file (default=stdout)\n");
  fprintf(stderr, "  -t <float|double> : scalar type (default=float)\n");
  fprintf(stderr, "  -p <precision> : number of bits of precision (default=full)\n");
  fprintf(stderr, "  -1 <nx> : dimensions of 1D array a[nx]\n");
  fprintf(stderr, "  -2 <nx> <ny> : dimensions of 2D array a[ny][nx]\n");
  fprintf(stderr, "  -3 <nx> <ny> <nz> : dimensions of 3D array a[nz][ny][nx]\n");
  fprintf(stderr, "  -4 <nx> <ny> <nz> <nf> : dimensions of multi-field 3D array a[nf][nz][ny][nx]\n");
  return EXIT_FAILURE;
}

int main(int argc, char* argv[])
{
  int type = FPZIP_TYPE_FLOAT;
  int prec = 0;
  int nx = 1;
  int ny = 1;
  int nz = 1;
  int nf = 1;
  char* inpath= 0;
  char* outpath = 0;
  bool zip = true;
  bool quiet = false;

  if (argc == 1)
    return usage();

  for (int i = 1; i < argc; i++)
    if (!strcmp(argv[i], "-h"))
      return usage();
    else if (!strcmp(argv[i], "-d"))
      zip = false;
    else if (!strcmp(argv[i], "-q"))
      quiet = true;
    else if (!strcmp(argv[i], "-i")) {
      if (++i == argc)
        return usage();
      inpath = argv[i];
    }
    else if (!strcmp(argv[i], "-o")) {
      if (++i == argc)
        return usage();
      outpath = argv[i];
    }
    else if (!strcmp(argv[i], "-t")) {
      if (++i == argc)
        return usage();
      if (!strcmp(argv[i], "float"))
        type = FPZIP_TYPE_FLOAT;
      else if (!strcmp(argv[i], "double"))
        type = FPZIP_TYPE_DOUBLE;
      else
        return usage();
    }
    else if (!strcmp(argv[i], "-p")) {
      if (++i == argc || sscanf(argv[i], "%d", &prec) != 1)
        return usage();
    }
    else if (!strcmp(argv[i], "-1")) {
      if (++i == argc || sscanf(argv[i], "%d", &nx) != 1)
        return usage();
      ny = nz = nf = 1;
    }
    else if (!strcmp(argv[i], "-2")) {
      if (++i == argc || sscanf(argv[i], "%d", &nx) != 1 ||
          ++i == argc || sscanf(argv[i], "%d", &ny) != 1)
        return usage();
      nz = nf = 1;
    }
    else if (!strcmp(argv[i], "-3")) {
      if (++i == argc || sscanf(argv[i], "%d", &nx) != 1 ||
          ++i == argc || sscanf(argv[i], "%d", &ny) != 1 ||
          ++i == argc || sscanf(argv[i], "%d", &nz) != 1)
        return usage();
      nf = 1;
    }
    else if (!strcmp(argv[i], "-4")) {
      if (++i == argc || sscanf(argv[i], "%d", &nx) != 1 ||
          ++i == argc || sscanf(argv[i], "%d", &ny) != 1 ||
          ++i == argc || sscanf(argv[i], "%d", &nz) != 1 ||
          ++i == argc || sscanf(argv[i], "%d", &nf) != 1)
        return usage();
    }

  // initialize

  // allocate buffers
  void* data = 0;

  if (zip) {
    // allocate memory for uncompressed data
    size_t count = (size_t)nx * ny * nz * nf;
    size_t size = (type == FPZIP_TYPE_FLOAT ? sizeof(float) : sizeof(double));
    data = (type == FPZIP_TYPE_FLOAT ? static_cast<void*>(new float[count]) : static_cast<void*>(new double[count]));
    if (prec == 0)
      prec = CHAR_BIT * size;
    else if (prec < 0 || (size_t)prec > CHAR_BIT * size) {
      fprintf(stderr, "precision out of range\n");
      return EXIT_FAILURE;
    }

    // read raw data
    FILE* file = inpath ? fopen(inpath, "rb") : stdin;
    if (!file) {
      fprintf(stderr, "cannot open input file\n");
      return EXIT_FAILURE;
    }
    if (fread(data, size, count, file) != count) {
      fprintf(stderr, "cannot read input file\n");
      return EXIT_FAILURE;
    }
    fclose(file);

    // compress to file
    file = outpath ? fopen(outpath, "wb") : stdout;
    if (!file) {
      fprintf(stderr, "cannot create output file\n");
      return EXIT_FAILURE;
    }
    FPZ* fpz = fpzip_write_to_file(file);
    fpz->type = type;
    fpz->prec = prec;
    fpz->nx = nx;
    fpz->ny = ny;
    fpz->nz = nz;
    fpz->nf = nf;
    // write header
    if (!fpzip_write_header(fpz)) {
      fprintf(stderr, "cannot write header: %s\n", fpzip_errstr[fpzip_errno]);
      return EXIT_FAILURE;
    }
    // perform actual compression
    size_t outbytes = fpzip_write(fpz, data);
    if (!outbytes) {
      fprintf(stderr, "compression failed: %s\n", fpzip_errstr[fpzip_errno]);
      return EXIT_FAILURE;
    }
    fpzip_write_close(fpz);
    fclose(file);
    if (!quiet)
      fprintf(stderr, "outbytes=%lu ratio=%.2f\n", (unsigned long)outbytes, double(nx) * ny * nz * nf * size / outbytes);
  }
  else {
    // decompress from file
    FILE* file = inpath ? fopen(inpath, "rb") : stdin;
    if (!file) {
      fprintf(stderr, "cannot open input file\n");
      return EXIT_FAILURE;
    }
    FPZ* fpz = fpzip_read_from_file(file);
    // read header
    if (!fpzip_read_header(fpz)) {
      fprintf(stderr, "cannot read header: %s\n", fpzip_errstr[fpzip_errno]);
      return EXIT_FAILURE;
    }
    type = fpz->type;
    prec = fpz->prec;
    nx = fpz->nx;
    ny = fpz->ny;
    nz = fpz->nz;
    nf = fpz->nf;
    if (!quiet)
      fprintf(stderr, "type=%s nx=%d ny=%d nz=%d nf=%d prec=%d\n", type == FPZIP_TYPE_FLOAT ? "float" : "double", nx, ny, nz, nf, prec);

    size_t count = (size_t)nx * ny * nz * nf;
    size_t size = (type == FPZIP_TYPE_FLOAT ? sizeof(float) : sizeof(double));
    data = (type == FPZIP_TYPE_FLOAT ? static_cast<void*>(new float[count]) : static_cast<void*>(new double[count]));
    // perform actual decompression
    if (!fpzip_read(fpz, data)) {
      fprintf(stderr, "decompression failed: %s\n", fpzip_errstr[fpzip_errno]);
      return EXIT_FAILURE;
    }
    fpzip_read_close(fpz);
    fclose(file);

    // write decompressed data to file
    file = outpath ? fopen(outpath, "wb") : stdout;
    if (!file) {
      fprintf(stderr, "cannot create output file\n");
      return EXIT_FAILURE;
    }
    if (fwrite(data, size, count, file) != count) {
      fprintf(stderr, "cannot write output file\n");
      return EXIT_FAILURE;
    }
    fclose(file);
  }

  // deallocate buffers
  if (type == FPZIP_TYPE_FLOAT)
    delete[] static_cast<float*>(data);
  else
    delete[] static_cast<double*>(data);

  return 0;
}
