from cloudvolume import CloudVolume
import numpy as np
from copy import deepcopy

black = CloudVolume('gs://seunglab-test/test_v0/black')
info = deepcopy(black.info)

sizes = ( 
	# [128,128,1],
	# [128,128,2],
	# [128,128,4],
	# [128,128,4],
	# [128,128,8],
	# [128,128,16],
	# [128,128,32],
	# [128,128,64],
	[128,128,128],
	[1024,1024,1],
	[1024,1024,2],
	[1024,1024,4],
	[1024,1024,8],
	[1024,1024,16],
	[1024,1024,32],
	[1024,1024,64],
	[1024,1024,100], # max size of SNEMI3D
)

imgvol = CloudVolume('gs://seunglab-test/test_v0/black', progress=True)
img = imgvol[:]

for chunksize in sizes:
	info['scales'][0]['chunk_sizes'] = [ chunksize ]

	layer = 'black_{}_{}_{}'.format(*chunksize)

	vol = CloudVolume('gs://seunglab-test/test_v0/' + layer, info=info, compress='gzip', progress=True)
	vol.reset_scales()
	vol.commit_info()

	# vol[:] = np.zeros(shape=vol.shape, dtype=vol.dtype)
	vol[:] = img

	layer = 'black_uncompressed_{}_{}_{}'.format(*chunksize)

	vol = CloudVolume('gs://seunglab-test/test_v0/' + layer, info=info, compress='', progress=True)
	vol.reset_scales()
	vol.commit_info()

	vol[:] = img

