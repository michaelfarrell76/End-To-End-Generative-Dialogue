# End-to-End Generative Dialogue

 A neural conversational model.

----
#### To run
```
cd End-To-End-Generative-Dialogue/src
python preprocess.py
python preprocess.py # seqlength 5 # For micro dataset (~500 sentences)

th train.lua -data_file data/conv-train.hdf5 -val_data_file data/conv-val.hdf5 -save_file conv-model -gpuid 1

th run_beam.lua -model conv-model.t7 -src_file data/dev_src_words.txt -targ_file data/dev_targ_words.txt -output_file pred.txt -src_dict data/src.dict -targ_dict data/targ.dict
```
To run in parallel
```
th train.lua -data_file data/conv-train.hdf5 -val_data_file data/conv-val.hdf5 -save_file conv-model -gpuid -1 -parallel
```

----
####Primary contributors

[Kevin Yang](https://github.com/kyang01)

[Michael Farrell](https://github.com/michaelfarrell76)

[Colton Gyulay](https://github.com/cgyulay)

----
#### Relevant links

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
#### TODO

- get beam working
- run each of the models for 10 epochs-ish? -> save the model, record results
- implement RNN model

- implement a way to demo the stuff using chat interface, demo can be done using terminal
- modify preprocess.py in seq2seq-elements to make the data directory if doesnt exist
- experiment with HRED model


- add subTle dataset pretraining
- heirarchical model 
- add in error rate stuff

----
#### Acknowledgments

Our implementation utilizes code from the following:

* [Yoon Kim's seq2seq-attn repo](https://github.com/harvardnlp/seq2seq-attn)
* [Element rnn library](https://github.com/Element-Research/rnn)
* [Facebook's neural attention model](https://github.com/facebook/NAMAS)
