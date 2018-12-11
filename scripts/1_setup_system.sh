#!/bin/bash

################################
# Install system-wide packages #
# Date: 2018-12-11  	       #
# Author: Xingdong Zuo         #
################################

# Stop when some intermediate command is failed
set -e

# Show Ubuntu version
sudo apt install -q -y lsb-release
echo '#######################################'
echo '# Your current Ubuntu OS information: # '
echo '#######################################'
lsb_release -a

# Update package list
sudo apt update
# Upgrade available packages
sudo apt -y upgrade
sudo apt -y dist-upgrade
# Install some packages
# libopenmpi-dev: support mpi4py
# cmake: for C
# zlib1g-dev: for compression e.g. gzip
# libboost-all-dev: Boost C++ library
# python3-dev: Python 3 library
# libjpeg-dev: JPEG support
# ffmpeg: Audio and Video processing
# python3-opengl: Python 3 OpenGL
# python-pyglet: pyglet
# libsdl2-dev: video processing
# swig: Useful for Box2D in gym
# libglew-dev: OpenGL stuff, useful for mujoco-py to work. 
# libosmesa6-dev: for Mujoco
# patchelf: ELF executables
# xorg-dev: X.org Window system
# libxinerama1: for X Window System
# libxcursor1: for X cursor
# xvfb: Fake screen, useful for gym rendering on the server
# htop: CPU monitoring
# vim: Vim editor
sudo apt install -q -y libopenmpi-dev cmake zlib1g-dev libboost-all-dev python3-dev libjpeg-dev ffmpeg python3-opengl python-pyglet libsdl2-dev swig libglew-dev libosmesa6-dev patchelf xorg-dev libxinerama1 libxcursor1 xvfb htop vim
