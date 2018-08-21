#!/bin/bash


########################################
# Install NVIDIA Drivers/CUDA/cuDNN    #
# Date: 2018-08-10                     #
# Author: Xingdong Zuo                 #
########################################

# Stop when some intermediate command is failed
set -e

# Create list of variables
# Find corresponding Driver runfile from Advanced Driver Search (with Beta version)
# http://www.nvidia.com/Download/Find.aspx
export DRIVER_URL=http://us.download.nvidia.com/XFree86/Linux-x86_64/396.51/NVIDIA-Linux-x86_64-396.51.run
export DRIVER_RUNFILE=NVIDIA-Linux-x86_64-396.51.run
# Find CUDA runfile from NVIDIA website
# https://developer.nvidia.com/cuda-downloads
export CUDA_URL=https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda_9.2.148_396.37_linux
export CUDA_RUNFILE=cuda_9.2.148_396.37_linux
export CUDA_VERSION=9.2
# cuDNN file: Download from https://developer.nvidia.com/rdp/cudnn-download
export CUDNN_FILE=cudnn-9.2-linux-x64-v7.2.1.38.tgz

# Disable nouveau driver: otherwise distribution provided pre-install will fail
if [ -e /etc/modprobe.d/blacklist-nouveau.conf ]
then
    read -p 'Make sure your system is just freshly rebooted, and press Enter to continue...'
else
    sudo sh -c 'echo "blacklist nouveau" >> /etc/modprobe.d/blacklist-nouveau.conf'
    sudo sh -c 'echo "options nouveau modeset=0" >> /etc/modprobe.d/blacklist-nouveau.conf'
    sudo update-initramfs -u
    echo 'Nouveau driver disabled successfully !'
    read -p 'Press Enter to reboot...'
    sudo reboot
fi

# Remove previous drivers
sudo apt purge -y nvidia*
sudo apt autoremove -y
echo 'Successfully removed previous NVIDIA driver !'

# Install required dependencies
sudo apt install -y build-essential dkms xorg xorg-dev

# Stop gdm (GNOME) service
sudo service gdm stop
echo 'Stopped gdm service !'

# Stop systemd-logind.service, otherwise installation cannot continue sometimes
sudo systemctl stop systemd-logind.service

##########
# Driver #
##########
# Download the runfile, make it executable and run it
wget $DRIVER_URL
chmod +x ./$DRIVER_RUNFILE  # make it executable
sudo ./$DRIVER_RUNFILE --dkms --no-opengl-files  # no OpenGL to support xvfb for fake screen on server
echo 'NVIDIA driver installation is completed !'

# Remove driver runfile
rm ./$DRIVER_RUNFILE

########
# CUDA #
########
# Download the runfile, make it executable and run it
wget $CUDA_URL
chmod +x ./$CUDA_RUNFILE  # make it executable
sudo ./$CUDA_RUNFILE  --no-opengl-libs  # no OpenGL to support xvfb for fake screen on server

# Append environment variables to .bashrc
echo "# Appended by CUDA installer" >> ~/.bashrc
echo "export PATH=\$PATH:/usr/local/cuda-$CUDA_VERSION/bin" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda-$CUDA_VERSION/lib64" >> ~/.bashrc
echo "export CUDA_HOME=/usr/local/cuda-$CUDA_VERSION/" >> ~/.bashrc
echo ""  >> ~/.bashrc
. ~/.bashrc  # refresh bashrc

echo 'CUDA installation is completed !'

# Remove CUDA runfile
rm ./$CUDA_RUNFILE

#########
# cuDNN #
#########
tar -xzvf $CUDNN_FILE
# Copy files to CUDA toolkit
sudo cp cuda/include/cudnn.h /usr/local/cuda-$CUDA_VERSION/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-$CUDA_VERSION/lib64
sudo chmod a+r /usr/local/cuda-$CUDA_VERSION/include/cudnn.h /usr/local/cuda-$CUDA_VERSION/lib64/libcudnn*

# Verify driver
echo '#####################################'
echo '# Verify NVIDIA driver installation #'
echo '#####################################'
/usr/bin/nvidia-smi
cat /proc/driver/nvidia/version

# Verify CUDA and nvcc
echo '#####################################'
echo '# Verify CUDA and nvcc installation #'
echo '#####################################'
/usr/local/cuda-$CUDA_VERSION/bin/nvcc --version
cat /usr/local/cuda-$CUDA_VERSION/version.txt

# Verify cuDNN
cat /usr/local/cuda-$CUDA_VERSION/include/cudnn.h | grep CUDNN_MAJOR -A 2
