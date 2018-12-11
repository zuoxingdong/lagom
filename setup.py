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
install_requires = ['cloudpickle', 
                    'pyyaml',
                    'opencv-python',
                    'colored',
                    'mpi4py',
                    'numpy', 
                    'scipy', 
                    'matplotlib', 
                    'seaborn',
                    'scikit-image', 
                    'scikit-learn', 
                    'imageio',
                    'pandas',
                    'gym', 
                    'cma']
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
