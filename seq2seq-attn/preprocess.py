#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create the data for the LSTM.
"""

import os
import sys
import argparse
import numpy as np
import h5py
import itertools
from collections import defaultdict

class Indexer:
    def __init__(self, symbols = ["*blank*","<unk>","<s>","</s>"]):
        # This needs to be changed so that it loads the proper mappings from word to characters
        self.vocab = defaultdict(int)
        self.PAD = symbols[0]
        self.UNK = symbols[1]
        self.BOS = symbols[2]
        self.EOS = symbols[3]
        self.d = {self.PAD: 1, self.UNK: 2, self.BOS: 3, self.EOS: 4}

    def add_w(self, ws):
        for w in ws:
            if w not in self.d:
                self.d[w] = len(self.d) + 1
            
    def convert(self, w):
        return self.d[w] if w in self.d else self.d[self.UNK]

    def convert_sequence(self, ls):
        return [self.convert(l) for l in ls]

    def clean(self, s):
        s = s.replace(self.PAD, "")
        # s = s.replace(self.UNK, "")
        s = s.replace(self.BOS, "")
        s = s.replace(self.EOS, "")
        return s
        
    def write(self, outfile):
        out = open(outfile, "w")
        items = [(v, k) for k, v in self.d.iteritems()]
        items.sort()
        for v, k in items:
            print >>out, k.encode('utf-8'), v
        out.close()

    # This sorts the vocab according to frequency count and reduces the 
    # vocab to a certain specified amount
    def prune_vocab(self, k):
        vocab_list = [(word, count) for word, count in self.vocab.iteritems()]
        vocab_list.sort(key = lambda x: x[1], reverse=True)
        k = min(k, len(vocab_list))
        self.pruned_vocab = {pair[0]:pair[1] for pair in vocab_list[:k]}
        for word in self.pruned_vocab:
            if word not in self.d:
                self.d[word] = len(self.d) + 1

    def load_vocab(self, vocab_file):
        self.d = {}
        for line in open(vocab_file, 'r'):
            v, k = line.decode("utf-8").strip().split()
            self.d[v] = int(k)
            
def pad(ls, length, symbol):
    if len(ls) >= length:
        return ls[:length]
    return ls + [symbol] * (length -len(ls))
        
def get_data(args):
    src_indexer = Indexer(["<blank>","<unk>","<s>","</s>"])
    target_indexer = Indexer(["<blank>","<unk>","<s>","</s>"])    
    
    def make_vocab(srcfile, targetfile, seqlength, max_word_l=0, chars=0):
        num_sents = 0
        for _, (src_orig, targ_orig) in \
                enumerate(itertools.izip(open(srcfile,'r'), open(targetfile,'r'))):
            src_orig = src_indexer.clean(src_orig.decode("utf-8").strip())
            targ_orig = target_indexer.clean(targ_orig.decode("utf-8").strip())
            targ = targ_orig.strip().split()
            src = src_orig.strip().split()
            if len(targ) > seqlength or len(src) > seqlength or len(targ) < 1 or len(src) < 1:
                continue
            num_sents += 1
            for word in targ:                                
                target_indexer.vocab[word] += 1
                
            for word in src:                 
                src_indexer.vocab[word] += 1
                
        return max_word_l, num_sents
                
    def convert(srcfile, targetfile, batchsize, seqlength, outfile, num_sents,
                max_word_l, max_sent_l=0,chars=0, unkfilter=0):
        
        newseqlength = seqlength + 2 #add 2 for EOS and BOS
        targets = np.zeros((num_sents, newseqlength), dtype=int)
        target_output = np.zeros((num_sents, newseqlength), dtype=int)
        sources = np.zeros((num_sents, newseqlength), dtype=int)
        source_lengths = np.zeros((num_sents,), dtype=int)
        target_lengths = np.zeros((num_sents,), dtype=int)
        dropped = 0
        sent_id = 0
        for _, (src_orig, targ_orig) in \
                enumerate(itertools.izip(open(srcfile,'r'), open(targetfile,'r'))):

            # Loading the sentences
            # src_orig = src_indexer.clean(src_orig.decode("utf-8").strip())
            # targ_orig = target_indexer.clean(targ_orig.decode("utf-8").strip())
            # targ = [target_indexer.BOS] + targ_orig.strip().split() + [target_indexer.EOS]
            # src =  [src_indexer.BOS] + src_orig.strip().split() + [src_indexer.EOS]

            # # We're bounding the length of a sequence for a file
            # if len(targ) > newseqlength or len(src) > newseqlength or len(targ) < 3 or len(src) < 3:
            #     dropped += 1
            #     continue                   
            # targ = pad(targ, newseqlength+1, target_indexer.PAD)
            # for word in targ:
            #     word = word if word in target_indexer.d else target_indexer.UNK                 
            # targ = target_indexer.convert_sequence(targ)
            # targ = np.array(targ, dtype=int)

            # src = pad(src, newseqlength, src_indexer.PAD)
            # src = src_indexer.convert_sequence(src)
            # src = np.array(src, dtype=int)
            
            # Drops all unknown characters
            # if unkfilter > 0:
            #     targ_unks = float((targ[:-1] == 2).sum())
            #     src_unks = float((src == 2).sum())                
            #     if unkfilter < 1: #unkfilter is a percentage if < 1
            #         targ_unks = targ_unks/(len(targ[:-1])-2)
            #         src_unks = src_unks/(len(src)-2)
            #     if targ_unks > unkfilter or src_unks > unkfilter:
            #         dropped += 1
            #         continue

            src = src_indexer.clean(src_orig.decode("utf-8").strip())
            targ = target_indexer.clean(targ_orig.decode("utf-8").strip())

            src = src.split()
            targ = targ.split()

            src = np.array([int(ind) for ind in src])
            targ = np.array([int(ind) for ind in targ] + [src_indexer.d[src_indexer.PAD]])
            # import bpdb; bpdb.set_trace()

            targets[sent_id] = np.array(targ[:-1],dtype=int)
            target_lengths[sent_id] = (targets[sent_id] != 1).sum()
            target_output[sent_id] = np.array(targ[1:],dtype=int)                    
            sources[sent_id] = np.array(src, dtype=int)
            source_lengths[sent_id] = (sources[sent_id] != 1).sum()            

            sent_id += 1
            if sent_id % 10000 == 0:
                print("{}/{} sentences processed".format(sent_id, num_sents))

        #print(sources)
        #print(targets)

        #break up batches based on source lengths
        source_lengths = source_lengths[:sent_id]
        source_sort = np.argsort(source_lengths) 

        sources = sources[source_sort]
        targets = targets[source_sort]
        target_output = target_output[source_sort]
        target_l = target_lengths[source_sort]
        source_l = source_lengths[source_sort]

        curr_l = 0
        l_location = [] #idx where sent length changes
        
        for j,i in enumerate(source_sort):
            if source_lengths[i] > curr_l:
                curr_l = source_lengths[i]
                l_location.append(j+1)
        l_location.append(len(sources))

        #get batch sizes
        curr_idx = 1
        batch_idx = [1]
        nonzeros = []
        batch_l = []
        batch_w = []
        target_l_max = []
        for i in range(len(l_location)-1):
            while curr_idx < l_location[i+1]:
                curr_idx = min(curr_idx + batchsize, l_location[i+1])
                batch_idx.append(curr_idx)
        for i in range(len(batch_idx)-1):
            batch_l.append(batch_idx[i+1] - batch_idx[i])            
            batch_w.append(source_l[batch_idx[i]-1])
            nonzeros.append((target_output[batch_idx[i]-1:batch_idx[i+1]-1] != 1).sum().sum())
            target_l_max.append(max(target_l[batch_idx[i]-1:batch_idx[i+1]-1]))

        # Write output
        f = h5py.File(outfile, "w")
        
        f["source"] = sources
        f["target"] = targets
        f["target_output"] = target_output
        f["target_l"] = np.array(target_l_max, dtype=int)
        f["target_l_all"] = target_l        
        f["batch_l"] = np.array(batch_l, dtype=int)
        f["batch_w"] = np.array(batch_w, dtype=int)
        f["batch_idx"] = np.array(batch_idx[:-1], dtype=int)
        f["target_nonzeros"] = np.array(nonzeros, dtype=int)
        f["source_size"] = np.array([len(src_indexer.d)])
        f["target_size"] = np.array([len(target_indexer.d)])

        print("Saved {} sentences (dropped {} due to length/unk filter)".format(
            len(f["source"]), dropped))
        f.close()                

    # print("First pass through data to get vocab...")
    # max_word_l, num_sents_train = make_vocab(args.srcfile, args.targetfile,
    #                                          args.seqlength, 0, 0)
    # print("Number of sentences in training: {}".format(num_sents_train))
    # max_word_l, num_sents_valid = make_vocab(args.srcvalfile, args.targetvalfile,
    #                                          args.seqlength, max_word_l, 0)
    # print("Number of sentences in valid: {}".format(num_sents_valid))    

    # #prune and write vocab
    # src_indexer.prune_vocab(args.srcvocabsize)
    # target_indexer.prune_vocab(args.targetvocabsize)
    if args.srcvocabfile != '':
        # You can try and load vocabulary here for both the source and target files 
        print('Loading pre-specified source vocab from ' + args.srcvocabfile)
        # import bpdb; bpdb.set_trace()
        src_indexer.load_vocab(args.srcvocabfile)
    if args.targetvocabfile != '':
        print('Loading pre-specified target vocab from ' + args.targetvocabfile)
        target_indexer.load_vocab(args.targetvocabfile)

    #################################
    # TODO: These are the target and src indexers that we need to replicate
    #################################
    src_indexer.write(args.outputfile + ".src.dict")
    target_indexer.write(args.outputfile + ".targ.dict")
    #################################

    print("Source vocab size: Original = {}, Pruned = {}".format(len(src_indexer.vocab), 
                                                          len(src_indexer.d)))
    print("Target vocab size: Original = {}, Pruned = {}".format(len(target_indexer.vocab), 
                                                          len(target_indexer.d)))

    # We really are only passing in max_word_l. I'm not really sure what that is. 
    # Need to also generate train and test dict files

    num_sents_valid = 196185
    num_sents_train = 196185
    max_word_l = 0

    convert(args.srcvalfile, args.targetvalfile, args.batchsize, args.seqlength,
             args.outputfile + "-val.hdf5", num_sents_valid,
             max_word_l, 0, 0, args.unkfilter)
    convert(args.srcfile, args.targetfile, args.batchsize, args.seqlength,
             args.outputfile + "-train.hdf5", num_sents_train, max_word_l,
             0, 0, args.unkfilter)
        
def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--srcvocabsize', help="Size of source vocabulary, constructed "
                                                "by taking the top X most frequent words. "
                                                " Rest are replaced with special UNK tokens.",
                                                type=int, default=10000)
    parser.add_argument('--targetvocabsize', help="Size of target vocabulary, constructed "
                                                "by taking the top X most frequent words. "
                                                "Rest are replaced with special UNK tokens.",
                                                type=int, default=10000)
    parser.add_argument('--srcfile', help="Path to source training data, "
                                           "where each line represents a single "
                                           "source/target sequence.", required=True)
    parser.add_argument('--targetfile', help="Path to target training data, "
                                           "where each line represents a single "
                                           "source/target sequence.", required=True)
    parser.add_argument('--srcvalfile', help="Path to source validation data.", required=True)
    parser.add_argument('--targetvalfile', help="Path to target validation data.", required=True)
    parser.add_argument('--batchsize', help="Size of each minibatch.", type=int, default=16)
    # Make sure to configure this to the correct value -> May be 300 or something. I believe 
    # This is not including the start and end characters. 
    parser.add_argument('--seqlength', help="Maximum sequence length. Sequences longer "
                                               "than this are dropped.", type=int, default=50)
    parser.add_argument('--outputfile', help="Prefix of the output file names. ", type=str, required=True)
    parser.add_argument('--maxwordlength', help="For the character models, words are "
                                           "(if longer than maxwordlength) or zero-padded "
                                            "(if shorter) to maxwordlength", type=int, default=35)
    parser.add_argument('--srcvocabfile', help="If working with a preset vocab, "
                                          "then including this will ignore srcvocabsize and use the"
                                          "vocab provided here.",
                                          type = str, default='data/src.dict')
    parser.add_argument('--targetvocabfile', help="If working with a preset vocab, "
                                         "then including this will ignore targetvocabsize and "
                                         "use the vocab provided here.",
                                          type = str, default='data/targ.dict')
    parser.add_argument('--unkfilter', help="Ignore sentences with too many UNK tokens. "
                                       "Can be an absolute count limit (if > 1) "
                                       "or a proportional limit (0 < unkfilter < 1).",
                                          type = float, default = 0)
    args = parser.parse_args(arguments)
    get_data(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
