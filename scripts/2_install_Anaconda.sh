#!/bin/bash

#############################
# Install Anaconda with env #
# Date: 2018-07-05  	    #
# Author: Xingdong Zuo      #
#############################

# Stop when some intermediate command is failed
set -e

# Create list of variables
export ANACONDA=Anaconda3-5.2.0-Linux-x86_64.sh  # Check new version
export ENV_NAME=RL  # name of Anaconda environment

# Download and install Anaconda
# Select NO to append the installation path to /.bashrc
wget https://repo.continuum.io/archive/$ANACONDA
chmod +x ./$ANACONDA  # make it executable
./$ANACONDA  # New version: don't choose yes to append PATH variable
rm ./$ANACONDA  # remove the installation file

# Append to load conda command from next run
# Remove PATH part in bashrc
echo ". $HOME/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc

# Alias for conda command
export CONDA=$HOME/anaconda3/bin/conda

# Update Anaconda to latest version
$CONDA update -n base conda

# Create an Anaconda environment
$CONDA create -n $ENV_NAME

# Append to bashrc to automatically load virtual environment for each start
echo "conda activate $ENV_NAME  # automatically activate virtual envs" >> ~/.bashrc  # double quotation works for variable

# Message to restart the shell
echo "############################"
echo "Please restart the shell to automatically load conda and environment."
echo "Then run the second bash file to set up dependencies."

