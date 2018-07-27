from setuptools import setup
from setuptools import find_packages

from lagom.version import __version__


# Read content of README.md
with open('README.md', 'r') as f:
    long_description = f.read()

setup(name='lagom',
      version=__version__,
      author='Xingdong Zuo',
      author_email='zuoxingdong@hotmail.com',
      description='lagom: A light PyTorch infrastructure to quickly prototype reinforcement learning algorithms.',
      # Long description of README markdown, shows in Python Package Index
      long_description=long_description, 
      long_description_content_type='text/markdown',
      url='https://github.com/zuoxingdong/lagom',
      # Install dependencies
      install_requires=['numpy', 
                        'scipy', 
                        'matplotlib', 
                        'scikit-image', 
                        'imageio',
                        'pandas',
                        'seaborn',
                        'jupyterlab', 
                        'gym', 
                        'cma', 
                        'flake8', 
                        'sphinx'],
      tests_require=['pytest'],
      # Only Python 3+
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
