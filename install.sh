#!/bin/sh

# without relying on anaconda:

python3 -m venv ~/venvs/motornet_env
source ~/venvs/motornet_env/bin/activate
python3 -m pip install -U pip
pip install motornet

# optional stuff for plg using VSCode
pip install ipykernel notebook jupyter




###########

# pip install tensorflow-macos
# pip install tensorflow-metal # for GPU-based stuff

#first install anaconda
#then conda create --name motornet python=3.8
#then conda activate motornet

pip3 install --upgrade pip
conda install numpy -y
conda install -c apple tensorflow-deps -y
pip3 install tensorflow-macos
conda install pandas matplotlib scipy scikit-learn jupyter ipython ipykernel notebook -y

# to remove: 
# conda env remove --name motornet
