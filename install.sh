#!/bin/sh

#first install anaconda
#then conda create --name motornet python=3.8
#then conda activate motornet

pip3 install --upgrade pip
conda install numpy -y
conda install -c apple tensorflow-deps -y
pip3 install tensorflow-macos
conda install pandas matplotlib scipy scikit-learn jupyter notebook -y


# to remove: 
# conda env remove --name motornet