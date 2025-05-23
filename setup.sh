#!/bin/bash

# system package update and install
apt-get update
apt install sudo
sudo apt-get install -y default-jdk
# sudo apt-get install -y cmake build-essential

# lightgbm install
pip install --upgrade --force-reinstall ipykernel
pip install \
   --no-binary lightgbm \
   --config-settings=cmake.define.USE_OPENMP=OFF \
   'lightgbm==4.0.0'

# install python package
pip install -r /root/default/qppmlc/requirements.txt
