# Benchmarks

On an x86_64 Intel Core i7-4820K CPU @ 3.70GHz with DDR3 1600 MHz RAM and a 1 Gbps connection, I compared the performance of gzip compressed and uncompressed downloads and uploads to different hosts using three versions of a 1024x1024x100 voxel test volume:

- 8-bit Zeroed Out (Black)
- 8-bit Grayscale EM Image 
- 16-bit Segmentation

## Google Cloud Storage

<p style="font-style: italics;" align="center">
<img height="512" src="https://raw.githubusercontent.com/seung-lab/cloud-volume/master/benchmarks/gcloud.png" alt="Fig. 1: Four plots of transfer rate versus chunk size for black, image, and segmentation data types for CloudVolume 1.0.0. (top left) Download rate of gzipped data (top right) Download rate of uncompressed data (bottom left) Upload rate of gzip compressed data (bottom right) Upload rate of uncompressed data." /><br>
Fig. 1: Four plots of transfer rate versus chunk size for black, image, and segmentation data types for CloudVolume 1.0.0. (top left) Download rate of gzipped data (top right) Download rate of uncompressed data (bottom left) Upload rate of gzip compressed data (bottom right) Upload rate of uncompressed data.
</p>

