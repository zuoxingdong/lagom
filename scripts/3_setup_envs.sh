#!/bin/bash

################################################
# Configure environment & install dependencies #
# Date: 2018-12-09  	                       #
# Author: Xingdong Zuo                         #
################################################

# Stop when some intermediate command is failed
set -e

# Set up .vimrc for tab as 4 spaces
echo "set tabstop=4" >> ~/.vimrc
echo "set shiftwidth=4" >> ~/.vimrc
echo "set expandtab" >> ~/.vimrc

# Install Mujoco
wget https://www.roboti.us/download/mujoco200_linux.zip
mkdir -p ~/.mujoco
unzip mujoco200_linux.zip -d ~/.mujoco
ln -s ~/.mujoco/mujoco200_linux ~/.mujoco/mujoco200
echo "# Mujoco" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin:$HOME/.mujoco/mujoco200_linux/bin" >> ~/.bashrc
echo ""  >> ~/.bashrc
rm mujoco200_linux.zip

# Install dependencies
pip install --upgrade pip
pip install -q -r --upgrade ../requirements.txt
