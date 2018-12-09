#!/bin/bash

##################################
# Install Miniconda & create env #
# Date: 2018-12-09  	         #
# Author: Xingdong Zuo           #
##################################

# Stop when some intermediate command is failed
set -e

# Create list of variables
export MINICONDA=Miniconda3-latest-Linux-x86_64.sh  # Check new version
export ENV_NAME=lagom  # name of conda environment
export PYTHON_VERSION=3.7  # Python version

# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/$MINICONDA
bash $MINICONDA -b  # batch mode, auto agree license and not touch .bashrc
hash -r
rm $MINICONDA

# Temporarily enable conda command
. $HOME/miniconda3/etc/profile.d/conda.sh

# Update to latest version
conda update -q -y conda
conda update -y --all
conda info -a

# Create and activate an environment with latest Python
conda create -q -y -n $ENV_NAME python=$PYTHON_VERSION cython patchelf ipython numpy
conda activate $ENV_NAME

# Append environment variables to .bashrc
echo "# Appended by Miniconda installer" >> ~/.bashrc
echo "export LIBRARY_PATH=\$LIBRARY_PATH:$HOME/miniconda3/envs/$ENV_NAME/lib" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$HOME/miniconda3/envs/$ENV_NAME/lib" >> ~/.bashrc
echo "export PKG_CONFIG_PATH=\$PKG_CONFIG_PATH:$HOME/miniconda3/envs/$ENV_NAME/lib/pkgconfig/" >> ~/.bashrc
echo ". $HOME/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc  # Load conda command
echo "conda activate $ENV_NAME" >> ~/.bashrc  # auto-load environment
echo ""  >> ~/.bashrc

echo "#######################"
echo "# Verify installation #"
echo "#######################"
conda list
echo "##########################"
echo "# Installation finished  #"
echo "# Restart a new terminal #"
echo "##########################"
