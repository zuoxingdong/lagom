#!/bin/bash

################################################
# Configure environment & install dependencies #
# Date: 2018-12-09  	                       #
# Author: Xingdong Zuo                         #
################################################

# Stop when some intermediate command is failed
set -e

# Set up .vimrc for tab as 4 spaces
echo "# Appended for Vim" >> ~/.vimrc
echo "set tabstop=4" >> ~/.vimrc
echo "set shiftwidth=4" >> ~/.vimrc
echo "set expandtab" >> ~/.vimrc
echo "" >> ~/.vimrc

# Install dependencies
pip install --upgrade pip
pip install -q -r ../requirements.txt