# Instruction to set up the environment

## Install NVIDIA Drivers
- Download GPU Driver file: http://www.nvidia.com/Download/Find.aspx
- Disable nouveau driver: otherwise distribution provided pre-install will fail

```
sudo sh -c 'echo "blacklist nouveau" >> /etc/modprobe.d/blacklist-nouveau.conf'
sudo sh -c 'echo "options nouveau modeset=0" >> /etc/modprobe.d/blacklist-nouveau.conf'
sudo update-initramfs -u
sudo reboot
```
- Remove old drivers: `sudo apt purge -y nvidia*`
- Install dependencies: `sudo apt install -y build-essential dkms xorg xorg-dev`
- Stop gdm (GNOME) service: `sudo service gdm stop`
- Stop systemd-logind.service: `sudo systemctl stop systemd-logind.service`
- Install driver file: no OpenGL to support xvfb for fake screen on server

```
chmod +x ./FILE
sudo ./FILE --dkms --no-opengl-files
```

## Install CUDA
- Download CUDA runfile: https://developer.nvidia.com/cuda-downloads
- Install CUDA: no OpenGL to support xvfb for fake screen on server

```
chmod +x ./FILE
sudo ./FILE --no-opengl-libs
```

- Add environment variables to `~/.bashrc`:

```
export CUDA_HOME=/usr/local/cuda-VER/
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
```

## Install cuDNN
- Download cuDNN file: https://developer.nvidia.com/rdp/cudnn-download
- Install

```
tar -xzvf FILE
sudo cp cuda/include/cudnn.h $CUDA_HOME/include
sudo cp cuda/lib64/libcudnn* $CUDA_HOME/lib64
sudo chmod a+r $CUDA_HOME/include/cudnn.h $CUDA_HOME/lib64/libcudnn*
```
