#!/bin/sh

#first install anaconda
conda create --name motornet python=3.8
conda activate motornet
pip3 install --upgrade pip
conda install numpy -y
conda install -c apple tensorflow-deps -y
pip3 install tensorflow-macos
conda install pandas matplotlib scipy scikit-learn jupyter notebook -y
