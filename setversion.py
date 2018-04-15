import re
import subprocess
import sys

def get_latest_version():
	tags = str(subprocess.check_output('git tag', shell=True))
	tags = re.sub("^b'", '', tags)
	tags = re.sub("'$", '', tags)
	tags = re.split(r'\\n', tags)
	tags = [ tuple(map(int, re.split('\.', tag))) for tag in tags if tag ]
	tags = sorted(tags, reverse=True)
	version = [ str(_) for _ in tags[0] ]
	return '.'.join(version)

def materialize_version(version):
	# version = get_version()
	with open('cloudvolume/__init__.py', 'rt') as f:
		code = f.read()

	code = re.sub("__version__ = '.*?'", "__version__ = '{}'".format(version), code)
	with open('cloudvolume/__init__.py', 'wt') as f:
		f.write(code)

version = sys.argv[1]
message = sys.argv[2]

materialize_version(version)
print("Updated __init__.py with version " + version)

assert len(version) < 9
subprocess.check_output("git reset && git add cloudvolume/__init__.py && git commit -m 'Version {}'".format(version), shell=True)

try:
	subprocess.check_output(['git', 'tag', '-a',  version, '-m', message])
	print('Created tag ' + version)
except:
	pass

