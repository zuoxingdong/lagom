#!/bin/bash

#############################
# Install dependencies      #
# Date: 2018-07-05  	    #
# Author: Xingdong Zuo      #
#############################

# Stop when some intermediate command is failed
set -e

# Create list of variables
export ENV_NAME=RL  # name of Anaconda environment
export NVIDIA_DRIVER_VER=396  # major version of Nvidia driver

# Alias of conda command
export CONDA=$HOME/anaconda3/bin/conda

# Update all installed packages
$CONDA update --all

# Append more useful things to bashrc
echo "# PATH" >> ~/.bashrc
echo "export PATH=/usr/lib/nvidia-$NVIDIA_DRIVER_VER/bin/:$PATH" >> ~/.bashrc

echo "# LIBRARY" >> ~/.bashrc
echo "export LIBRARY_PATH=$HOME/anaconda3/envs/$ENV_NAME/lib:$LIBRARY_PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$HOME/anaconda3/envs/$ENV_NAME/lib:$LD_LIBRARY_PATH" >> ~/.bashrc

echo "# CUDA" >> ~/.bashrc
echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc

echo "# PKG_CONFIG" >> ~/.bashrc
echo "export PKG_CONFIG_PATH=$HOME/anaconda3/envs/$ENV_NAME/lib/pkgconfig/" >> ~/.bashrc

# Install IPython to enforce all later command in Python 3 context
$CONDA install -y ipython

# Upgrade pip and install some dependencies
pip install --upgrade pip
pip install cmake cython msgpack

# Install some conda packages
$CONDA install -y numpy matplotlib scikit-image

# Install some pip packages
pip install jupyterlab gym[all] pytest

# PyTorch
$CONDA install pytorch torchvision -c pytorch