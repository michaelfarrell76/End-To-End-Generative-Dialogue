# End-to-End Generative Dialogue

 A neural conversational model.

----
## To run
```
cd End-To-End-Generative-Dialogue/src

python preprocess.py

python preprocess.py # --seqlength 5 # For micro dataset (~500 sentences)

th train.lua -data_file data/conv-train.hdf5 -val_data_file data/conv-val.hdf5 -save_file conv-model -gpuid 1

th run_beam.lua -model conv-model.t7 -src_file data/dev_src_words.txt -targ_file data/dev_targ_words.txt -output_file pred.txt -src_dict data/src.dict -targ_dict data/targ.dict
```
To run in parallel
```
th train.lua -data_file data/conv-train.hdf5 -val_data_file data/conv-val.hdf5 -save_file conv-model -gpuid -1 -parallel
```
NB: the MovieTriples dataset is not publicly available. Training on arbitrary dialogue will be supported soon.

----
## Primary contributors

[Kevin Yang](https://github.com/kyang01)

[Michael Farrell](https://github.com/michaelfarrell76)

[Colton Gyulay](https://github.com/cgyulay)

----
## Relevant links

- https://medium.com/chat-bots/the-complete-beginner-s-guide-to-chatbots-8280b7b906ca#.u1jngyhzc
- https://www.youtube.com/watch?v=IK0t38Al4_E
- https://github.com/julianser/hed-dlg
- https://docs.google.com/document/d/1KKP8ZRZJbZweazZvz4cHZkvVnzFQApJJEySpZ5JLdwc/edit
- http://arxiv.org/pdf/1507.04808.pdf
- http://arxiv.org/pdf/1511.06931v6.pdf
- https://www.aclweb.org/anthology/P/P15/P15-1152.pdf
- http://arxiv.org/pdf/1603.09457v1.pdf
- https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/
- https://www.reddit.com/r/MachineLearning/comments/3ukvc6/datasets_of_one_to_one_conversations/
- http://arxiv.org/pdf/1412.3555v1.pdf
- https://github.com/clementfarabet/lua---parallel
- http://www.aclweb.org/anthology/P02-1040.pdf
- http://victor.chahuneau.fr/notes/2012/07/03/kenlm.html
- https://cloud.google.com/compute/docs/troubleshooting

----
## TODO

#### Preprocessing (preprocess.py)

- Add subTle datset cleaning to preprocessing code (and any other additional datasets we may need)
- Modify preprocessing code to have longer sequences (rather than just (U_1, U_2, U_3), have (U_1, ..., U_n) for some n. With this we could try to add more memory to the model we currently have now)
- Modify preprocessing code to return entire conversations (rather than fixing n, have the entire back and forth of a conversation together. This could be useful for trying to train a model more specific to our objective. This could be used for testing how the model does for a specific conversation )
- Finish cleaning up file (i.e. finish factoring code. I started this but things are going to be modified when subTle is added so I never finished. It shouldn't be bad at all)

#### LUA

- get beam working
- run each of the models for 10 epochs-ish? -> save the model, record results
- implement RNN model
- experiment with HRED model
- heirarchical model 
- add in error rate stuff

----
## GCLOUD stuff

sudo apt-get install git
sudo apt-get install luarocks

git config --global user.email "mkfrl09@gmail.com"
git config --global user.name "michaelfarrell76"

In order to set up your instance, you're going to need to install torch

git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh

cd .. 

You'll also need to install all relevant packages, so 

luarocks install rnn
luarocks install parallel


sudo apt-get install libhdf5-serial-dev hdf5-tools
git clone https://github.com/deepmind/torch-hdf5.git
cd torch-hdf5
luarocks make hdf5-0-0.rockspec

cd ..

You'll also need to install Anaconda
wget http://repo.continuum.io/archive/Anaconda2-4.0.0-Linux-x86_64.sh
bash Anaconda2-4.0.0-Linux-x86_64.sh


git clone https://github.com/michaelfarrell76/Singularity.git
cd Singularity
mkdir data
cd data
mkdir MovieTriples

(Disconnect from shell, reconnect from shell if python/torch aren't loading properly)

Afterwards, you'll want to clone the repo and git pull. 

Next, you'll want to put the MovieTriples dataset into the correct director. First navigate into the MovieTriples
directory on your local drive, then run the below:
gcloud compute copy-files . train-conv2:~/stash/singularity/data/MovieTriples --zone us-east1-c

Next, run the command for preprocessing, and you should be good to go!

cd 
cd Singularity
cd seq2seq-elements
python preprocess.py

------------------------------------------------------------------------------

For figuring out IP adddressses between Google compute instances, you can use their instance name 
and the nslookup command. For example, if you were in another Google Compute instance and 
wanted to figure out the ip address for the instance model-1, you can run the command below command

nslookup model-1

This should return to you the local IP address of model-1 that you can use to connect with. 


https://cloud.google.com/compute/docs/instances/connecting-to-instance#generatesshkeypairx

----
## Acknowledgments

Our implementation utilizes code from the following:

* [Yoon Kim's seq2seq-attn repo](https://github.com/harvardnlp/seq2seq-attn)
* [Element rnn library](https://github.com/Element-Research/rnn)
* [Facebook's neural attention model](https://github.com/facebook/NAMAS)
