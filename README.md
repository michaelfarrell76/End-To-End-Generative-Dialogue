# End-to-End Generative Dialogue

A neural conversational model.

## Requirements

This code is written in Lua, and an installation of [Torch](https://github.com/torch/torch7/) is assumed. Training requires a few packages which can easily be installed through [LuaRocks](https://github.com/keplerproject/luarocks) (which comes with a Torch installation). Datasets are formatted and loaded using [hdf5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format), which can be installed using this [guide](https://github.com/deepmind/torch-hdf5/blob/master/doc/usage.md).
```bash
$ luarocks install nn
$ luarocks install rnn
```
If you want to train on an Nvidia GPU using CUDA, you'll need to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) as well as the `cutorch` and `cunn` packages:
```bash
$ luarocks install cutorch
$ luarocks install cunn
```
If you'd like to chat with your trained model, you'll need the `penlight` package:
```bash
$ luarocks install penlight
```
## Usage

### Data

Input data is stored in the `data` directory. At the moment, this code is only compatible with the MovieTriples dataset, as defined by [Serban et al., 2015](http://arxiv.org/abs/1507.04808). Unzip the MovieTriples dataset and place its contents into `data/MovieTriples`. *Note: the MovieTriples dataset is not publicly available, though training on arbitrary dialogue will be supported soon.*

**Preprocessing** is done using Python:
```bash
$ cd src
$ python preprocess.py
```
A limited dataset (limited input length) can also be used for testing:
```bash
$ python preprocess.py --seqlength 5
```

### Training

You can start training the model using `train.lua`.
```bash
$ th train.lua -data_file data/conv-train.hdf5 -val_data_file data/conv-val.hdf5 -save_file conv-model -gpuid 1
```
Here we are setting the flag `gpuid` to 1, which trains using the GPU. You can train on the CPU by omitting this flag or setting its value to -1. For a full list of model settings, please consult `$ th train.lua -help`.

**Checkpoints** are created after each epoch, and are saved within the `src` directory. Each checkpoint's name indicates the number of completed epochs of training as well as that checkpoint's [perplexity](https://en.wikipedia.org/wiki/Perplexity), essentially a measure of how confused the checkpoint is in its predictions. The lower the number, the better the checkpoint's predictions (and output text while sampling).

### Sampling

Given a checkpoint file, we can generate responses to input dialogue examples:
```bash
$ th run_beam.lua -model conv-model_epoch4.00_39.19.t7 -src_file data/dev_src_words.txt -targ_file data/dev_targ_words.txt -output_file pred.txt -src_dict data/src.dict -targ_dict data/targ.dict
```

### Chatting

It's also possible to chat directly with a checkpoint:
```bash
$ th chat.lua -model conv-model_epoch4.00_39.19.t7 -targ_dict data/targ.dict
```
These models have a tendency to respond tersely and vaguely. It's a work in progress!

## Advanced Usage

We have implemented support for training the model using [Distributed SGD](https://github.com/michaelfarrell76/Distributed-SGD) to farm our clients that will simultaneously compute gradients/update parameters, both remotely and locally.

### Setup

In order to run code in parallel, you need the [Distributed SGD](https://github.com/michaelfarrell76/Distributed-SGD) which contains this directory (End-To-End-Generative-Dialogue) as a submodule:
```bash
$ git clone --recursive https://github.com/michaelfarrell76/Distributed-SGD.git
$ cd Distributed-SGD/lua-lua
```
Next make sure that you have the correct [requirements](https://github.com/michaelfarrell76/Distributed-SGD/tree/master/lua-lua#requirements) installed via the Requirements section of the [Distributed SGD](https://github.com/michaelfarrell76/Distributed-SGD) repo.

Next setup your data folder in Distributed-SGD/lua-lua/End-To-End-Generative-Dialogue according to the instructions in the [data](https://github.com/michaelfarrell76/End-To-End-Generative-Dialogue#data) section. Once this is setup correctly, you should have a folder
Distributed-SGD/lua-lua/End-To-End-Generative-Dialogue/data/MovieTriples full with the MovieTriples dataset files. 

Finally, run the preprocessing code:
```bash
$ cd Distributed-SGD/lua-lua/End-To-End-Generative-Dialogue/src
$ python preprocess.py
```

### Running code in parallel

##### Local 

To run a worker with 2 parallel clients on your own machine:
```bash
$ th train.lua -data_file data/conv-train.hdf5 -val_data_file data/conv-val.hdf5 -save_file conv-model -parallel -n_proc 2
```

##### localhost

You can start a worker that talks to its clients through localhost.

First follow the setup [instructions](https://github.com/michaelfarrell76/Distributed-SGD/tree/master/lua-lua#remote---localhost) under the 'Remote -localhost' header, through the 'Allow ssh connections' subheader. 

Once you've successfully added your ssh to the list of authorized keys, and allowed for remote login, you can run this through localhost with the following command:
```bash
$ PATH_TO_SRC=Desktop/GoogleDrive/FinalProject/Distributed-SGD/lua-lua/End-To-End-Generative-Dialogue/src/
$ PATH_TO_TORCH=/Users/michaelfarrell/torch/install/bin/th
$ th train.lua -data_file data/conv-train.hdf5 -val_data_file data/conv-val.hdf5 -save_file conv-model -parallel -n_proc 2 -localhost -extension $PATH_TO_SRC -torch_path $PATH_TO_TORCH
```

### Running remotely on gcloud servers

Ideally, you can have the clients farmed out to different remote computers instead of running locally. 

First, zip the data folder that contains the MovieTriples dataset in the main End-To-End-Generative-Dialogue folder:
```bash
$ cd End-To-End-Generative-Dialogue
$ zip -r data.zip data
```

Next navigate to the 'lua-lua' folder and follow the setup [instructions](https://github.com/michaelfarrell76/Distributed-SGD/tree/master/lua-lua#remote---gcloud) under the 'Remote -gcloud' header. Follow these instructions exactly as they are written through the section 'Adding ssh keys again', except in the section labeled 'Setup the disk', instead of running the command
```bash
$  source setup_image.sh
```
in the 'Setup the disk' step, you should run
```bash
$  source End-To-End-Generative-Dialogue/src/setup_image_End-to-End.sh
```
which contains the necessary changes to the setup for this task.

Once the servers are all setup and you are connected to your host server, you should navigate back to the src directory and you can run the train.lua file on remote servers:

```bash
$ cd Distributed-SGD/lua-lua/End-To-End-Generative-Dialogue/src
$ EXTENSION=Distributed-SGD/lua-lua/End-To-End-Generative-Dialogue/src/
$ TORCH_PATH=/home/michaelfarrell/torch/install/bin/th
$ th train.lua -data_file data/conv-train.hdf5 -val_data_file data/conv-val.hdf5 -save_file conv-model -parallel -remote -extension $EXTENSION -torch_path $TORCH_PATH -n_proc 12

```

**Running the remote server:** 
If the servers have been initialized, you will first want to connect to one of them:
```bash
$ ssh -o "StrictHostKeyChecking no" -i ~/.ssh/gcloud-sshkey $USERNAME@$IP_ADDR

```
Once connected, you need to again setup an ssh key as listed in the instructions: "Set up an ssh key to connect to our servers" above.
Once the key is created and added to the account, then:
```bash
$ cd End-To-End-Generative-Dialogue/src
$ th train.lua -data_file data/conv-train.hdf5 -val_data_file data/conv-val.hdf5 -save_file conv-model -parallel -remote -extension End-To-End-Generative-Dialogue/src/ -torch_path /home/michaelfarrell/torch/install/bin/th -n_proc 4

```

## Primary contributors

[Kevin Yang](https://github.com/kyang01)

[Michael Farrell](https://github.com/michaelfarrell76)

[Colton Gyulay](https://github.com/cgyulay)

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

## TODO

**Preprocessing (preprocess.py)**
- Add subTle datset cleaning to preprocessing code (and any other additional datasets we may need)
- Modify preprocessing code to have longer sequences (rather than just (U_1, U_2, U_3), have (U_1, ..., U_n) for some n. With this we could try to add more memory to the model we currently have now)
- Modify preprocessing code to return entire conversations (rather than fixing n, have the entire back and forth of a conversation together. This could be useful for trying to train a model more specific to our objective. This could be used for testing how the model does for a specific conversation )
- Finish cleaning up file (i.e. finish factoring code. I started this but things are going to be modified when subTle is added so I never finished. It shouldn't be bad at all)

**Parallel (parallel_functions.lua)**
- Add way to do localhost without password on server
- Get working on google servers
- Make sure server setup is correctly done

**General**
- Start result collection of some sort. Maybe have some datasheet and when we run a good model we record the results?
- Run each of the models for 10 epochs-ish? -> save the model, record results ^
- Experiment with HRED model
- Add word error rate when reporting

## Acknowledgments

Our implementation utilizes code from the following:

* [Yoon Kim's seq2seq-attn repo](https://github.com/harvardnlp/seq2seq-attn)
* [Element rnn library](https://github.com/Element-Research/rnn)
* [Facebook's neural attention model](https://github.com/facebook/NAMAS)
