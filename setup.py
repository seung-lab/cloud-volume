import setuptools
import subprocess
import re
from functools import reduce

def get_version():
	tags = str(subprocess.check_output('git tag', shell=True))
	tags = re.sub("^b'", '', tags)
	tags = re.sub("'$", '', tags)
	tags = re.split(r'\\n', tags)
	tags = [ tuple(map(int, re.split('\.', tag))) for tag in tags if tag ]
	tags = sorted(tags, reverse=True)
	version = [ str(_) for _ in tags[0] ]
	return '.'.join(version)

def materialize_version():
	version = get_version()
	with open('cloudvolume/__init__.py', 'rt') as f:
		code = f.read()

	code = re.sub("__version__ = '.*?'", "__version__ = '{}'".format(version), code)
	with open('cloudvolume/__init__.py', 'wt') as f:
		f.write(code)

materialize_version()

setuptools.setup(
    setup_requires=['pbr'],
    pbr=True)