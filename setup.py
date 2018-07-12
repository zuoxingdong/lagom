from setuptools import setup

from lagom.__init__ import __version__


setup(name='lagom',
      install_requires=['pytest', 
                        'numpy', 
                        'matplotlib', 
                        'scikit-image', 
                        'jupyterlab', 
                        'gym', 
                        'cma'],
      description='lagom: A light PyTorch infrastructure to quickly prototype reinforcement learning algorithms.',
      author='Xingdong Zuo',
      url='https://github.com/zuoxingdong/lagom',
      version=__version__
)