from collections import defaultdict
from copy import deepcopy
import time
import re
import json
import socket
from operator import mul
from functools import reduce

import numpy as np
from cloudvolume import CloudVolume

N = 2

CHUNK_SIZES = ( 
	(128,128,1),
	(128,128,2),
	(128,128,4),
	(128,128,8),
	(128,128,16),
	(128,128,32),
	(128,128,64),
	(128,128,128),
	(1024,1024,1),
	(1024,1024,2),
	(1024,1024,4),
	(1024,1024,8),
	(1024,1024,16),
	(1024,1024,32),
	(1024,1024,64),
	(1024,1024,100), # max size of SNEMI3D
)

logfile = open('./benchmark.tsv', 'wt')
logfile.write("hostname\tdirection\tcompression\timage_type\tchunk_size\tdtype\tMB\tMean MB/sec\tN\tmean (sec)\tfastest (sec)\tslowest (sec)\tcloudpath\n")
logfile.flush()

def stopwatch(fn):
	start = time.time()
	fn()
	return time.time() - start

def benchmark(fn, N):
	lst = [ stopwatch(fn) for _ in range(N) ]
	return {
		'mean': (sum(lst) / N), 
		'fastest': min(lst), 
		'slowest': max(lst), 
		'N': N,
	}

def MBs(vol, sec):
	bits, = re.search(r'(\d+)$', vol.dtype).groups()
	dtype_bytes = int(bits) // 8
	total_bytes = vol.bounds.volume() * dtype_bytes
	return total_bytes / 1e6 / sec

def disp(desc, vol, stats):
	mbs = lambda sec: MBs(vol, sec)
	return "%s -- mean=%.2f MB/sec, range=%.2f to %.2f MB/sec; N=%d" % (
		desc, mbs(stats['mean']), mbs(stats['slowest']), mbs(stats['fastest']), stats['N']
	)

def log(row):
	global logfile
	bits, = re.search(r'(\d+)$', row['dtype']).groups()
	dtype_bytes = int(bits) // 8
	row['MB'] = reduce(mul, row['chunk_size']) * dtype_bytes / 1024**2
	row['MBs'] = row['MB'] / row['mean']

	for k,v in row.items():
		if type(v) is float:
			row[k] = '%.3f' % v

	entry = "{hostname}\t{direction}\t{compression}\t{image_type}\t{chunk_size}\t{dtype}\t{MB}\t{MBs}\t{N}\t{mean}\t{fastest}\t{slowest}\t{cloudpath}\n".format(**row)
	logfile.write(entry)
	logfile.flush()

def benchmark_upload(voltype):
	global CHUNK_SIZES
	global N

	originvol = CloudVolume('gs://seunglab-test/test_v0/{}'.format(voltype))
	originvol.reset_scales()
	info = deepcopy(originvol.info)

	img = originvol[:]

	for chunk_size in CHUNK_SIZES[::-1]:
		cloudpath = 'gs://seunglab-test/test_v0/{}_upload_{}_{}_{}'.format(voltype, *chunk_size)
		info['scales'][0]['chunk_sizes'] = [ list(chunk_size) ]
		vol = CloudVolume(cloudpath, progress=True, info=info, compress='gzip')

		def upload():
			vol[:] = img

		stats = benchmark(upload, N)

		log({
			"direction": "upload",
			"compression": "gzip",
			"image_type": voltype,
			"N": N,
			"mean": stats['mean'],
			"fastest": stats['fastest'],
			"slowest": stats['slowest'],
			"cloudpath": cloudpath,
			"chunk_size": chunk_size,
			"dtype": vol.dtype,
			"hostname": socket.gethostname(),
		})

		vol.delete(vol.bounds)

	for chunk_size in CHUNK_SIZES[::-1]:
		cloudpath = 'gs://seunglab-test/test_v0/{}_upload_{}_{}_{}'.format(voltype, *chunk_size)
		info['scales'][0]['chunk_sizes'] = [ list(chunk_size) ]
		vol = CloudVolume(cloudpath, progress=True, info=info, compress='')

		def upload():
			vol[:] = img
		
		stats = benchmark(upload, N)

		log({
			"direction": "upload",
			"compression": "none",
			"image_type": voltype,
			"N": N,
			"mean": stats['mean'],
			"fastest": stats['fastest'],
			"slowest": stats['slowest'],
			"cloudpath": cloudpath,
			"chunk_size": chunk_size,
			"dtype": vol.dtype,
			"hostname": socket.gethostname(),
		})

		vol.delete(vol.bounds)


def benchmark_download(voltype):
	global CHUNK_SIZES
	global N

	for chunk_size in CHUNK_SIZES[::-1]:
		cloudpath = 'gs://seunglab-test/test_v0/{}_{}_{}_{}'.format(voltype, *chunk_size)
		vol = CloudVolume(cloudpath, progress=True)
		stats = benchmark(lambda: vol[:], N)

		log({
			"direction": "download",
			"compression": "gzip",
			"image_type": voltype,
			"N": N,
			"mean": stats['mean'],
			"fastest": stats['fastest'],
			"slowest": stats['slowest'],
			"cloudpath": cloudpath,
			"chunk_size": chunk_size,
			"dtype": vol.dtype,
			"hostname": socket.gethostname(),
		})

	for chunk_size in CHUNK_SIZES[::-1]:
		cloudpath = 'gs://seunglab-test/test_v0/{}_uncompressed_{}_{}_{}'.format(voltype, *chunk_size)
		vol = CloudVolume(cloudpath, progress=True)
		stats = benchmark(lambda: vol[:], N)

		log({
			"direction": "download",
			"compression": "none",
			"image_type": voltype,
			"N": N,
			"mean": stats['mean'],
			"fastest": stats['fastest'],
			"slowest": stats['slowest'],
			"cloudpath": cloudpath,
			"chunk_size": chunk_size,
			"dtype": vol.dtype,
			"hostname": socket.gethostname(),
		})

benchmark_download('black')
benchmark_download('image')
benchmark_download('segmentation')

benchmark_upload('black')
benchmark_upload('image')
benchmark_upload('segmentation')


logfile.close()
