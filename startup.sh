#!/bin/bash
hash git || (echo "Y" | sudo apt-get install git)
git config --global user.email "mkfrl09@gmail.com"
git config --global user.name "michaelfarrell76"
hash luarocks || (echo "Y" | sudo apt-get install luarocks)
hash pip || (echo "Y" | sudo apt-get install python-pip)


source ~/.profile

hash th || ( git clone https://github.com/torch/distro.git ~/torch --recursive; cd ~/torch; bash install-deps; (echo "yes" | ./install.sh); cd ..)
source ~/.profile
luarocks install env
sudo apt-get install libzmq3-dev libzmq3

git clone https://github.com/clementfarabet/lua---parallel.git
cd lua---parallel
luarocks make

cd ..


if (luarocks list | grep -q rnn)
then
  echo "Rnn installed"
else
 luarocks install rnn
fi

if (luarocks list | grep -q parallel)
then
  echo "parallel installed"
else
 luarocks install parallel
fi

if (luarocks list | grep -q env)
then
  echo "env installed"
else
 echo "HERE"
 luarocks install env
 echo "HERE@"
fi

if (luarocks list | grep -q hdf5)
then
  echo "hdf5 installed"
else
 echo "Y" | sudo apt-get install libhdf5-serial-dev hdf5-tools
 git clone https://github.com/deepmind/torch-hdf5.git
 cd torch-hdf5
 luarocks make hdf5-0-0.rockspec
 cd ..
fi

if ! [ -e "Singularity" ]
then
 git clone https://github.com/michaelfarrell76/End-To-End-Generative-Dialogue.git
 cd Singularity
 mkdir data
 cd seq2seq-elements
 mkdir data
 cd ../..
fi

if ! [ -e "anaconda2" ]
then
wget http://repo.continuum.io/archive/Anaconda2-4.0.0-Linux-x86_64.sh
bash Anaconda2-4.0.0-Linux-x86_64.sh -b
echo 'export PATH="/home/michaelfarrell/anaconda2/bin:$PATH"' > .bashrc
echo "RESTART YOUR TERMINAL"
else
 conda install h5py
fi

source ~/.profile

