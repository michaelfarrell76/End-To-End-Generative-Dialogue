# End-to-End Generative Dialogue

 A neural conversational model.

----
## Running Code

Before anything can be run, the MovieTriples dataset is first required. 

First create the data directory
```
mkdir data
```
and copy into the directory the MovieTriples dataset. 

Your directory should look like:
```
.
├── data	     
│   └── MovieTriples
|        ├── ...
|        ...
├── src
...
```

Code is run from the /src folder
```
cd End-To-End-Generative-Dialogue/src
```
#### Preprocessing Code

```
python preprocess.py
```
For micro dataset
```
python preprocess.py --seqlength 5 # For micro dataset (~500 sentences)
```
#### Running the model
```
th train.lua -data_file data/conv-train.hdf5 -val_data_file data/conv-val.hdf5 -save_file conv-model -gpuid 1 #Runs on gpu

th run_beam.lua -model conv-model.t7 -src_file data/dev_src_words.txt -targ_file data/dev_targ_words.txt -output_file pred.txt -src_dict data/src.dict -targ_dict data/targ.dict
```

### Running Code in Parallel

#### Locally

To run a worker with 4 parallel clients on your own computer:
```
th train.lua -data_file data/conv-train.hdf5 -val_data_file data/conv-val.hdf5 -save_file conv-model -parallel -n_proc 4

```
#### Locally through localhost

To run a worker with 1 parallel client on your own computer running through localhost (which is more similar to how things will work when running through the google server). There is only 1 parallel client since it requires that you input your password while connecting to your own computer through ssh. I didn't want to deal with passwords so I just spawn one worker,input the password, and see if it works. There is no point to use this in practice since its just slightly more inefficient than the previous command. Use this as a benchmark for developing the remote server training. 

In order for this to work, you must first **enable Remote Login in System Preferences/Sharing**

**Note**: You have to specify the location of the src folder from the home directory of your computer:

i.e. **PATH_TO_SRC = Desktop/GoogleDrive/FinalProject/Singularity/src/**
```
th train.lua -data_file data/conv-train.hdf5 -val_data_file data/conv-val.hdf5 -save_file conv-model -parallel -n_proc 1 -localhost -extension PATH_TO_SRC

```
#### In dev: through remote gcloud servers

You must first set up an ssh key to connect to the servers. 

Replace USERNAME with your own username.

i.e. USERNAME = michaelfarrell

```
ssh-keygen -t rsa -f ~/.ssh/gcloud-sshkey -C USERNAME
```
Hit enter twice and a key should have been generated.

```
cat ~/.ssh/gcloud-sshkey.pub
```
And then copy the key that is printed out.

Next you must add the key to the set of public keys. 

- Login to our google compute account. 
- Go to compute engine dashboard
- Go to metdata tab
- Go to ssh-key subtab
- Click edit
- Add the key you copied as a new line

Restrict access:

```
chmod 400 ~/.ssh/gcloud-sshkey
```

Next create your own instance group if you have not created one already. 

- Go to the 'Instance groups' tab
- Create instance group
- Give the group a name, i.e. training-group-dev
- Give a description
- Set zone to us-central1-b
- Use instance template
- Choose "mike-instance-template-1"
- Set the number of instances
- Create
- Wait for the instances to launch
- 



Currently attempting to run with the parallel workers running remotely on the servers with the code below.
```
th train.lua -data_file data/conv-train.hdf5 -val_data_file data/conv-val.hdf5 -save_file conv-model -parallel -n_proc 4 -remote -extension End-To-End-Generative-Dialogue/src/

```
### Notes:

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

#### Parallel (parallel_functions.lua)
- Add way to do localhost without password on server
- Get working on google servers
- Make sure server setup is correctly done

#### General 

- Start result collection of some sort. Maybe have some datasheet and when we run a good model we record the results?
- run each of the models for 10 epochs-ish? -> save the model, record results ^
- implement RNN model
- experiment with HRED model
- heirarchical model 
- add in error rate stuff when reporting

----
## Acknowledgments

Our implementation utilizes code from the following:

* [Yoon Kim's seq2seq-attn repo](https://github.com/harvardnlp/seq2seq-attn)
* [Element rnn library](https://github.com/Element-Research/rnn)
* [Facebook's neural attention model](https://github.com/facebook/NAMAS)
