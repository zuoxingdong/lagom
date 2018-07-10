#!/bin/bash


#############################
# Build PyTorch wheel file  #
# Date: 2018-07-10          #
# Author: Xingdong Zuo      #
#############################

# Stop when some intermediate command is failed
set -e

#####################################################
# Make sure the anaconda environment is activated.  #
#####################################################

# Alias of conda command
export CONDA=$HOME/anaconda3/bin/conda

# Create list of variables
export CMAKE_PREFIX_PATH=/home/zuo/anaconda3
export CUDA_VERSION=90

# Install dependencies
$CONDA install -y numpy pyyaml mkl mkl-include setuptools cffi typing
$CONDA install -y -c mingfeima mkldnn
$CONDA install -y -c pytorch magma-cuda$CUDA_VERSION

# Git clone PyTorch source file
git clone --recursive https://github.com/pytorch/pytorch

# Build wheel file
cd pytorch
python setup.py bdist_wheel

# Move wheel file out
cd ..  # move back to parent directory of pytorch
mv pytorch/dist/torch* ./

# Clean PyTorch source folder
rm -rf pytorch
