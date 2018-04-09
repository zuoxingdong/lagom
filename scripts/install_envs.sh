#!/bin/bash

#############################
# Install dependencies      #
# Date: April 7, 2018  	    #
# Author: Xingdong Zuo      #
#############################

# Stop when some intermediate command is failed
set -e

# Create list of variables
export ENV_NAME=RL_server  # name of Anaconda environment
export PYTORCH=http://download.pytorch.org/whl/cu80/torch-0.3.1-cp36-cp36m-linux_x86_64.whl  # Check cuda version

# Alias of conda command
export CONDA=$HOME/anaconda3/bin/conda

# Update all installed packages
$CONDA update --all

# Append more useful things to bashrc
echo "# LIBRARY" >> ~/.bashrc
echo "export LIBRARY_PATH=$HOME/anaconda3/envs/$ENV_NAME/lib:$LIBRARY_PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$HOME/anaconda3/envs/$ENV_NAME/lib:$LD_LIBRARY_PATH" >> ~/.bashrc

echo "# CUDA" >> ~/.bashrc
echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc

echo "# PKG_CONFIG" >> ~/.bashrc
echo "export PKG_CONFIG_PATH=$HOME/anaconda3/envs/$ENV_NAME/lib/pkgconfig/" >> ~/.bashrc

# Install packages and dependencies
# IPython, Numpy, Matplotlib, Scikit-image
$CONDA install -y ipython numpy matplotlib scikit-image
# Update pip
pip install --upgrade pip
# Jupyterlab
pip install jupyterlab
# PyTorch
pip install $PYTORCH
pip install torchvision
# OpenAI Gym
pip install gym
