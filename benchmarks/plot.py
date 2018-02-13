import matplotlib.pyplot as plt
import numpy as np
import csv

import sys
from collections import defaultdict

rows = []

filename = sys.argv[1] 

with open(filename, 'r') as csvfile:
  votereader = csv.reader(csvfile, delimiter='\t', quotechar=None)
  for row in votereader:
        rows.append(row)

headers = rows.pop(0)

stats = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

for row in rows:
	direction, imgtype, compression, MB, meanMBs = row[1], row[3], row[2], row[6], row[7]
	stats[direction][compression][imgtype][float(MB)] = float(meanMBs)


styles = {
	'black': 'k',
	'image': (.7, .7, .7, 1),
	'segmentation': 'r',
}


index = 0
for direction in sorted(stats.keys()):
	direxpr = stats[direction]

	for compress in sorted(direxpr.keys()):
		imageexper = direxpr[compress]
		index += 1

		plt.subplot(2, 2, index)
		plt.title(filename + ': CloudVolume ' + direction + ' Speed w/ ' + compress + ' Compression')
		plt.xlabel('Chunk Size (MB)')
		plt.ylabel('MB/sec')

		lines = []
		legends = []

		for imgtype in imageexper.keys():
			style = styles[imgtype]
			experiment = imageexper[imgtype]
			xdata = sorted([ float(mb) for mb in experiment.keys() ])
			ydata = [ float(experiment[mb]) for mb in xdata ]
			line = plt.plot(xdata, ydata, color=style, linestyle='-', linewidth=2)
			lines.append(line)
			legends.append(imgtype)

		plt.legend(legends)
plt.show()






