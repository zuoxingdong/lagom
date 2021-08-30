from importlib.machinery import SourceFileLoader
import pkg_resources
from distutils.version import LooseVersion
import re
import codecs
from setuptools import setup
from setuptools import find_packages


# Read long description of README markdown, shows in Python Package Index
with codecs.open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# Minimal requried dependencies (full dependencies in requirements.txt)
install_requires = ['numpy', 
                    'scipy', 
                    'gym>=0.15.3', 
                    'cloudpickle', 
                    'pyyaml', 
                    'colorama']
tests_require = ['pytest', 
                 'flake8', 
                 'sphinx', 
                 'sphinx_rtd_theme']

setup(name='lagom',
      version=SourceFileLoader('version', 'lagom/version.py').load_module().__version__,
      # List all lagom packages (folder with __init__.py), useful to distribute a release
      packages=find_packages(), 
      
      install_requires=install_requires,
      tests_require=tests_require,
      python_requires='>=3.7',
      
      author='Xingdong Zuo',
      author_email='zuoxingdong@hotmail.com',
      description='lagom: A light PyTorch infrastructure to quickly prototype reinforcement learning algorithms.',
      long_description=long_description, 
      long_description_content_type='text/markdown',
      url='https://github.com/zuoxingdong/lagom',      
      # tell pip some metadata (e.g. Python version, OS etc.)
      classifiers=['Intended Audience :: Science/Research',
                   'Programming Language :: Python :: 3.7',
                   'Programming Language :: Python :: 3.8',
                   'License :: OSI Approved :: MIT License', 
                   'Operating System :: OS Independent', 
                   'Natural Language :: English', 
                   'Topic :: Scientific/Engineering :: Artificial Intelligence']
)


# check PyTorch installation
pkg = None
for name in ['torch', 'torch-nightly']:
    try:
        pkg = pkg_resources.get_distribution(name)
    except pkg_resources.DistributionNotFound:
        pass
assert pkg is not None, 'PyTorch is not correctly installed.'
version_msg = 'PyTorch of version above 1.2.0 expected'
assert LooseVersion(re.search(r'\d+[.]\d+[.]\d+', pkg.version)[0]) >= LooseVersion('1.2.0'), version_msg
