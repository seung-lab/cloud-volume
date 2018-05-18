import setuptools

setuptools.setup(
    setup_requires=['pbr'],
    extras_require={
    	':python_version == "2.7"': ['futures'],
    	':python_version == "2.6"': ['futures'],
    },
    pbr=True)