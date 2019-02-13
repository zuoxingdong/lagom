import codecs

from setuptools import setup
from setuptools import find_packages


# read version directly as variable, not import lagom, because it might not be installed yet
with open('lagom/version.py') as f:
    exec(f.read())

# Read long description of README markdown, shows in Python Package Index
with codecs.open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# Minimal requried dependencies (full dependencies in requirements.txt)
install_requires = ['numpy', 
                    'scipy', 
                    'gym', 
                    'cloudpickle', 
                    'pyyaml', 
                    'colored']
tests_require = ['pytest', 
                 'flake8', 
                 'sphinx', 
                 'sphinx_rtd_theme']

setup(name='lagom',
      version=__version__,
      author='Xingdong Zuo',
      author_email='zuoxingdong@hotmail.com',
      description='lagom: A light PyTorch infrastructure to quickly prototype reinforcement learning algorithms.',
      long_description=long_description, 
      long_description_content_type='text/markdown',
      url='https://github.com/zuoxingdong/lagom',      
      install_requires=install_requires,
      tests_require=tests_require,
      python_requires='>=3',
      # List all lagom packages (folder with __init__.py), useful to distribute a release
      packages=find_packages(), 
      # tell pip some metadata (e.g. Python version, OS etc.)
      classifiers=['Programming Language :: Python :: 3', 
                   'License :: OSI Approved :: MIT License', 
                   'Operating System :: OS Independent', 
                   'Natural Language :: English', 
                   'Topic :: Scientific/Engineering :: Artificial Intelligence']
)


# ensure PyTorch is installed
import pkg_resources
pkg = None
for name in ['torch', 'torch-nightly']:
    try:
        pkg = pkg_resources.get_distribution(name)
    except pkg_resources.DistributionNotFound:
        pass
assert pkg is not None, 'PyTorch is not correctly installed.'

from distutils.version import LooseVersion
import re
version_msg = 'PyTorch of version above 1.0.0 expected'
assert LooseVersion(re.search(r'\d+[.]\d+[.]\d+', pkg.version)[0]) >= LooseVersion('1.0.0'), version_msg
