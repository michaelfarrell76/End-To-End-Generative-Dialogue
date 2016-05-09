#!/bin/bash

if hash git &> /dev/null
then
	echo -e "\033[0;32mgit installed\033[0m"
else
	echo -e "\033[0;34mInstalling git ...\033[0m"
	(echo "Y" | sudo apt-get install git) > /dev/null  
fi

if hash luarocks &> /dev/null
then
	echo -e "\033[0;32mluarocks installed\033[0m"
else
	echo -e "\033[0;34mInstalling luarocks ...\033[0m"
	(echo "Y" | sudo apt-get install luarocks) &> /dev/null  
fi

if hash pip &> /dev/null
then
	echo -e "\033[0;32mpython-pip installed\033[0m"
else
	echo -e "\033[0;34mInstalling python-pip ...\033[0m"
	(echo "Y" | sudo apt-get install python-pip) > /dev/null  
fi

source ~/.profile

if hash th &> /dev/null
then
	echo -e "\033[0;32mtorch installed\033[0m"
else
	echo -e "\033[0;34mInstalling torch ...\033[0m"
	git clone https://github.com/torch/distro.git ~/torch --recursive &> /dev/null
	cd ~/torch
	bash install-deps 2&>1 > /dev/null 
	echo "yes" | ./install.sh 2&>1 > /dev/null 
	cd ..
	source ~/.profile
fi


if [ -e "lua---parallel" ]	
then
	echo -e "\033[0;32mparallel installed\033[0m"
else
	echo -e "\033[0;34mInstalling parallel ...\033[0m"
	git clone https://github.com/clementfarabet/lua---parallel.git &> /dev/null
	cd lua---parallel
	luarocks make > /dev/null
	cd ..
fi

if (luarocks list | grep -q rnn) &> /dev/null  
then
	echo -e "\033[0;32mrnn installed\033[0m"
else
	echo -e "\033[0;34mInstalling rnn ...\033[0m"
	luarocks install rnn &> /dev/null  
fi

if (luarocks list | grep -q env) &> /dev/null  
then
	echo -e "\033[0;32menv installed\033[0m"
else
	echo -e "\033[0;34mInstalling env ...\033[0m"
	luarocks install env &> /dev/null  
fi

if (luarocks list | grep -q hdf5) &> /dev/null  
then
	echo -e "\033[0;hdf5 installed\033[0m"
else
	echo -e "\033[0;34mInstalling hdf5 ...\033[0m"
 	echo "Y" | sudo apt-get install libhdf5-serial-dev hdf5-tools > /dev/null  
 	git clone https://github.com/deepmind/torch-hdf5.git &> /dev/null
 	cd torch-hdf5
 	luarocks make hdf5-0-0.rockspec &> /dev/null  
 	cd ..
fi

if [ -e "End-To-End-Generative-Dialogue" ]
then 
	echo -e "\033[0;34mPulling End-To-End-Generative-Dialogue repo changes ...\033[0m"
	cd End-To-End-Generative-Dialogue
	git pull &> /dev/null
	cd ..
else
	echo -e "\033[0;34mCloning repo End-To-End-Generative-Dialogue ...\033[0m"
 	git clone https://github.com/michaelfarrell76/End-To-End-Generative-Dialogue.git &> /dev/null
 	mkdir End-To-End-Generative-Dialogue/data
fi

if [ -e "anaconda2" ]
then 
	echo -e "\033[0;anaconda installed\033[0m"
	echo -e "\033[0;34mInstalling h5py ...\033[0m"
	echo "y" | conda install h5py &> /dev/null
else
	echo -e "\033[0;34mDownloading anaconda ...\033[0m"
	wget http://repo.continuum.io/archive/Anaconda2-4.0.0-Linux-x86_64.sh &> /dev/null
	echo -e "\033[0;34mInstalling anaconda ...\033[0m"
	bash Anaconda2-4.0.0-Linux-x86_64.sh -b > /dev/null
	rm Anaconda2-4.0.0-Linux-x86_64.sh
	echo 'export PATH="/home/michaelfarrell/anaconda2/bin:$PATH"' > .bashrc
	echo -e "\033[0;33mIn order for python to be run, you must logout and log back in\033[0m" 
fi

