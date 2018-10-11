#!/bin/bash

#############################
# Install Anaconda with env #
# Date: 2018-10-11  	    #
# Author: Xingdong Zuo      #
#############################

# Stop when some intermediate command is failed
set -e

# Create list of variables
export ANACONDA=Anaconda3-5.3.0-Linux-x86_64.sh  # Check new version
export ENV_NAME=lagom  # name of Anaconda environment
export PYTHON_VERSION=3.7  # Python version

# Download and install Anaconda
# Select NO to append the installation path to /.bashrc
wget https://repo.continuum.io/archive/$ANACONDA
chmod +x ./$ANACONDA  # make it executable
./$ANACONDA  # New version: don't choose yes to append PATH variable
rm ./$ANACONDA  # remove the installation file

# Alias for conda command
export CONDA=$HOME/anaconda3/bin/conda

# Update Anaconda to latest version
$CONDA update -n base conda

# Create an Anaconda environment with latest Python
$CONDA create -n $ENV_NAME python=$PYTHON_VERSION

# Append environment variables to .bashrc
echo "# Appended by Anaconda installer" >> ~/.bashrc
echo ". $HOME/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc  # Load conda command
echo "export LIBRARY_PATH=\$LIBRARY_PATH:$HOME/anaconda3/envs/$ENV_NAME/lib" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$HOME/anaconda3/envs/$ENV_NAME/lib" >> ~/.bashrc
echo "export PKG_CONFIG_PATH=\$PKG_CONFIG_PATH:$HOME/anaconda3/envs/$ENV_NAME/lib/pkgconfig/" >> ~/.bashrc
echo "conda activate $ENV_NAME  # automatically activate virtual envs" >> ~/.bashrc  # auto-load environment
echo ""  >> ~/.bashrc
. ~/.bashrc  # refresh bashrc

# Message to restart the shell
echo "#####################################################################"
echo "Please restart the shell to automatically load conda and environment."
echo "Then run the second bash file to set up dependencies."
