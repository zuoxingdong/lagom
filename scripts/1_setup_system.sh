#!/bin/bash

################################
# Install system-wide packages #
# Date: 2018-12-09  	       #
# Author: Xingdong Zuo         #
################################

# Stop when some intermediate command is failed
set -e

# Show Ubuntu version
echo '#######################################'
echo '# Your current Ubuntu OS information: # '
echo '#######################################'
lsb_release -a

# Update package list
sudo apt update
# Upgrade available packages
sudo apt upgrade
sudo apt dist-upgrade
# Install some packages
# zlib1g-dev: for compression e.g. gzip
# libjpeg-dev: JPEG support
# xvfb: Fake screen, useful for gym rendering on the server
# libav-tools: Audio and Video processing
# xorg-dev: X.org Window system
# python3-opengl: Python 3 OpenGL
# libboost-all-dev: Boost C++ library
# libsdl2-dev: About video processing
# swig: Useful for Box2D in gym
# libglew-dev: OpenGL stuff, useful for mujoco-py to work. 
# libglu1-mesa: for GLU
# libglu1-mesa-dev: for GLU
# libgl1-mesa-dev: for GL
# libxinerama1: for X Window System
# libxcursor1: for X cursor
# htop: CPU monitoring
# vim: Vim editor
sudo apt install zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python3-opengl libboost-all-dev libsdl2-dev swig libglew-dev libglu1-mesa libglu1-mesa-dev libgl1-mesa-dev libxinerama1 libxcursor1 htop vim
