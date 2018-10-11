#!/bin/bash

#############################
# Install dependencies      #
# Date: 2018-10-11  	    #
# Author: Xingdong Zuo      #
#############################

# Stop when some intermediate command is failed
set -e

# Alias of conda command
export CONDA=$HOME/anaconda3/bin/conda

# Set up .vimrc for tab as 4 spaces
echo "set tabstop=4" >> ~/.vimrc
echo "set shiftwidth=4" >> ~/.vimrc
echo "set expandtab" >> ~/.vimrc

# Install dependencies
$CONDA update --all
$CONDA install -y ipython patchelf

pip install --upgrade pip
pip install -r ../requirements.txt
