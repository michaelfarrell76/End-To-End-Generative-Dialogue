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
import pickle
import sys
import re
import codecs
from itertools import izip

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
            src_orig = src_indexer.clean(src_orig.decode("utf-8").strip())
            targ_orig = target_indexer.clean(targ_orig.decode("utf-8").strip())
            targ = [target_indexer.BOS] + targ_orig.strip().split() + [target_indexer.EOS]
            src =  [src_indexer.BOS] + src_orig.strip().split() + [src_indexer.EOS]
            max_sent_l = max(len(targ), len(src), max_sent_l)

            # We're bounding the length of a sequence for a file
            if len(targ) > newseqlength or len(src) > newseqlength or len(targ) < 3 or len(src) < 3:
                dropped += 1
                continue                   

            targ = pad(targ, newseqlength+1, target_indexer.PAD)
            for word in targ:
                word = word if word in target_indexer.d else target_indexer.UNK                 
            targ = target_indexer.convert_sequence(targ)
            targ = np.array(targ, dtype=int)

            src = pad(src, newseqlength, src_indexer.PAD)
            src = src_indexer.convert_sequence(src)
            src = np.array(src, dtype=int)
            
            # Drops all unknown characters
            if unkfilter > 0:
                targ_unks = float((targ[:-1] == 2).sum())
                src_unks = float((src == 2).sum())                
                if unkfilter < 1: #unkfilter is a percentage if < 1
                    targ_unks = targ_unks/(len(targ[:-1])-2)
                    src_unks = src_unks/(len(src)-2)
                if targ_unks > unkfilter or src_unks > unkfilter:
                    dropped += 1
                    continue
                
            targets[sent_id] = np.array(targ[:-1],dtype=int)
            target_lengths[sent_id] = (targets[sent_id] != 1).sum()
            target_output[sent_id] = np.array(targ[1:],dtype=int)                    
            sources[sent_id] = np.array(src, dtype=int)
            source_lengths[sent_id] = (sources[sent_id] != 1).sum()            

            sent_id += 1
            if sent_id % 100000 == 0:
                print("{}/{} sentences processed".format(sent_id, num_sents))

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
        return max_sent_l

    print("First pass through data to get vocab...")
    max_word_l, num_sents_train = make_vocab(args.srcfile, args.targetfile,
                                             args.seqlength, 0, 0)
    print("Number of sentences in training: {}".format(num_sents_train))
    max_word_l, num_sents_valid = make_vocab(args.srcvalfile, args.targetvalfile,
                                             args.seqlength, max_word_l, 0)
    print("Number of sentences in valid: {}".format(num_sents_valid))    

    #prune and write vocab
    src_indexer.prune_vocab(args.srcvocabsize)
    target_indexer.prune_vocab(args.targetvocabsize)
    if args.srcvocabfile != '':
        # You can try and load vocabulary here for both the source and target files 
        print('Loading pre-specified source vocab from ' + args.srcvocabfile)
        src_indexer.load_vocab(args.srcvocabfile)
    if args.targetvocabfile != '':
        print('Loading pre-specified target vocab from ' + args.targetvocabfile)
        target_indexer.load_vocab(args.targetvocabfile)
        
    src_indexer.write(args.outputfile + ".src.dict")
    target_indexer.write(args.outputfile + ".targ.dict")
    
    print("Source vocab size: Original = {}, Pruned = {}".format(len(src_indexer.vocab), 
                                                          len(src_indexer.d)))
    print("Target vocab size: Original = {}, Pruned = {}".format(len(target_indexer.vocab), 
                                                          len(target_indexer.d)))

    max_sent_l = 0
    max_sent_l = convert(args.srcvalfile, args.targetvalfile, args.batchsize, args.seqlength,
                         args.outputfile + "-val.hdf5", num_sents_valid,
                         max_word_l, max_sent_l, 0, args.unkfilter)
    max_sent_l = convert(args.srcfile, args.targetfile, args.batchsize, args.seqlength,
                         args.outputfile + "-train.hdf5", num_sents_train, max_word_l,
                         max_sent_l, 0, args.unkfilter)
    
    print("Max sent length (before dropping): {}".format(max_sent_l))    

def format_data(directory, train_valid_split, seq_length, args):
    # Loading all the possible files into memory
    with open(directory + 'Training.triples.pkl') as f:
        train_set = pickle.load(f)
        
    with open(directory + 'Validation.triples.pkl') as f:
        valid_set = pickle.load(f)
        
    with open(directory + 'Test.triples.pkl') as f:
        test_set = pickle.load(f)
        
    with open(directory + 'Word2Vec_WordEmb.pkl') as f:
        emb_wordvec = pickle.load(f)
        
    with open(directory + 'MT_WordEmb.pkl') as f:
        emb_mt = pickle.load(f)

    # Implement the word indices according to the format for the seq2seq model

    # Make sure that the word_indices are 1 indexed for lua

    # Do a swap with the embeddings and word_indices so it follows the conventions for indices 
    # self.d = {self.PAD: 1, self.UNK: 2, self.BOS: 3, self.EOS: 4}

    with open(directory + 'Training.dict.pkl') as f:
        word_mappings = pickle.load(f)

    # All the swaps necessary to make the formatting consistent with seq2seq (sorry, it's so messy)
    del_ind = []
    for i in range(len(word_mappings)):
        word_mapping = word_mappings[i]
        if word_mapping[0] == '<unk>' or word_mapping[0] == '<s>' or \
            word_mapping[0] == '</s>' or word_mapping[0] == '.' or \
            word_mapping[0] == "'":
                del_ind.append(i)

    del_ind.sort(reverse=True)
    for ind in del_ind:
        del word_mappings[ind]
            
    word_mappings.append(('<blank>', 1, 0, 0))
    word_mappings.append(('<unk>', 2, 190588, 89059))
    word_mappings.append(('<s>', 3, 588827, 785135))
    word_mappings.append(('</s>', 4, 588827, 785135))
    word_mappings.append(('.', 10003, 855616, 192250))
    word_mappings.append(("'", 10004, 457542, 160249))
    word_mappings.append(('<t>', 10005, 0, 0))

    # Sanity check
    check_mappings = range(1, len(word_mappings)+1)
    for word_mapping in word_mappings:
        check_mappings.remove(word_mapping[1])
    assert check_mappings == []

    # The changes that need to occur in the actual text examples are: 
    # ., 3 -> 10003
    # ', 4-> 10004
    # <unk>, 0 -> 2
    # <s>, 1 -> 3
    # </s>, 2 -> 4

    data_sets = [train_set, valid_set, test_set]
    for i in range(len(data_sets)):
        for j in range(len(data_sets[i])):
            line = data_sets[i][j]
            for k in range(len(line)):
                ind = line[k]
                if ind == 3:
                    line[k] = 10003
                elif ind == 4:
                    line[k] = 10004
                elif ind == 0:
                    line[k] = 2
                elif ind == 1:
                    line[k] = 3
                elif ind == 2:
                    line[k] = 4
            data_sets[i][j] = line

    # Move through the list of words and indices and generate a dictionary
    # matching the indices to words

    # indices -> word
    indices_to_word = {}
    for word_ex in word_mappings: 
        indices_to_word[word_ex[1]] = word_ex[0]
        
    # word -> indices
    word_to_indices = {}
    for word_ex in word_mappings: 
        word_to_indices[word_ex[0]] = word_ex[1]

    # Apply above basic parsing to all contexts and outputs

    PADDING = word_to_indices['<blank>']
    END_OF_CONV = word_to_indices['<t>']

    full_context = []
    full_output = []
    max_len_context = 0
    max_len_output = 0 

    pattern = [word_to_indices['</s>'], word_to_indices['<s>']]

    data_set_contexts = []
    data_set_outputs = []
    for data_set in [train_set, valid_set, test_set]:
        PADDING = word_to_indices['<blank>']
        END_OF_CONV = word_to_indices['<t>']

        full_context = []
        full_output = []
        max_len_context = 0
        max_len_output = 0 

        for i in range(len(data_set)):
            break_pt = []
            for ind in range(len(data_set[i]))[::-1]:
                if pattern == data_set[i][ind:ind+2]:
                    break_pt.append(ind)

            context = data_set[i][:break_pt[0]]
            output = data_set[i][break_pt[0]+2:]

            context = context + [word_to_indices['</s>']]
            output = [word_to_indices['<s>']] + output

            # Start of sentence and end of sentence is ONLY used at the end
            # We create a new character that represents the start and end of a conversation
            context = context[:break_pt[1]] + [END_OF_CONV] + context[break_pt[1]+2:]


            # Cap the target and src length at 302 words to make computation simpler, goes up to ~1500
            if len(context) > 52:
                continue
            if len(output) > 52:
                continue

            max_len_output = max(max_len_output, len(output))
            max_len_context = max(max_len_context, len(context))
            max_len_output = 52
            max_len_context = 52

            full_context.append(context)
            full_output.append(output)

        # Add padding to all contexts and outputs
        for i in range(len(full_context)):
            full_context[i] = full_context[i] + [PADDING] * (max_len_context - len(full_context[i]))
            full_output[i] = full_output[i] + [PADDING] * (max_len_output - len(full_output[i]))

        data_set_contexts.append(full_context)
        data_set_outputs.append(full_output)
        
    train_full_context = data_set_contexts[0]
    train_full_output = data_set_outputs[0]
    valid_full_context = data_set_contexts[1]
    valid_full_output = data_set_outputs[1]

    # This is super inefficient, put it together last minute. Don't judge :)
    f =  open(args.output_directory + 'train_src_indices.txt', 'w')
    for context in train_full_context: 
        for ind in context:
            f.write(str(ind) + ' ')
        f.write('\n')
    f.close()

    f =  open(args.output_directory + 'train_targ_indices.txt', 'w')
    for output in train_full_output: 
        for ind in output:
            f.write(str(ind) + ' ')
        f.write('\n')
    f.close()

    f =  open(args.output_directory + 'dev_src_indices.txt', 'w')
    for context in valid_full_context: 
        for ind in context:
            f.write(str(ind) + ' ')
        f.write('\n')
    f.close()

    f =  open(args.output_directory + 'dev_targ_indices.txt', 'w')
    for output in valid_full_output: 
        for ind in output:
            f.write(str(ind) + ' ')
        f.write('\n')
    f.close()

    with open(args.output_directory + 'targ.dict', 'w') as f: 
        for i in range(1, len(indices_to_word)+1):
            f.write(indices_to_word[i] + ' ' + str(i) + '\n')
            
    with open(args.output_directory + 'src.dict', 'w') as f: 
        for i in range(1, len(indices_to_word)+1):
            f.write(indices_to_word[i] + ' ' + str(i) + '\n')
            
            
    special_indices = [1, 2, 3, 4,]
    train_full_context_words = []
    for context in train_full_context:
        context_words = []
        for ind in context:
            if ind not in special_indices:
                context_words.append(indices_to_word[ind])
        train_full_context_words.append(' '.join(context_words))
    f =  open(args.output_directory + 'train_src_words.txt', 'w')
    for context in train_full_context_words: 
        f.write(str(context) + ' \n')
    f.close()

    valid_full_context_words = []
    for context in valid_full_context:
        context_words = []
        for ind in context:
            if ind not in special_indices:
                context_words.append(indices_to_word[ind])
        valid_full_context_words.append(' '.join(context_words))
    f =  open(args.output_directory + 'dev_src_words.txt', 'w')
    for context in valid_full_context_words: 
        f.write(str(context) + ' \n')
    f.close()

    train_full_output_words = []
    for output in train_full_output:
        output_words = []
        for ind in output:
            if ind not in special_indices:
                output_words.append(indices_to_word[ind])
        train_full_output_words.append(' '.join(output_words))
    f =  open(args.output_directory + 'train_targ_words.txt', 'w')
    for output in train_full_output_words: 
        f.write(str(output) + ' \n')
    f.close()

    valid_full_output_words = []
    for output in valid_full_output:
        output_words = []
        for ind in output:
            if ind not in special_indices:
                output_words.append(indices_to_word[ind])
        valid_full_output_words.append(' '.join(output_words))
    f =  open(args.output_directory + 'dev_targ_words.txt', 'w')
    for output in valid_full_output_words: 
        f.write(str(output) + ' \n')
    f.close()

    emb_wordvec_upd = np.vstack((emb_wordvec[0], np.zeros((300, 2)).T))
    emb_wordvec_upd[10001] = emb_wordvec[0][3][:]
    emb_wordvec_upd[10002] = emb_wordvec[0][4][:]
    emb_wordvec_upd[3] = emb_wordvec[0][10001][:]
    emb_wordvec_upd[4] = emb_wordvec[0][10002][:]
    emb_wordvec_upd = np.roll(emb_wordvec_upd, 1, axis=0)
    f = h5py.File(args.output_directory + 'word_vecs.hdf5', 'w')
    f['word_vecs'] = emb_wordvec_upd
    f.close()

    print('Done formatting the data from Movie Triples dataset')

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
    parser.add_argument('--output_directory', help="Folder to hold output of data", default='data/')


    parser.add_argument('--srcfile', help="Path to source training data, "
                                           "where each line represents a single "
                                           "source/target sequence.", default='train_src_words.txt')
    parser.add_argument('--targetfile', help="Path to target training data, "
                                           "where each line represents a single "
                                           "source/target sequence.", default='train_targ_words.txt')
    parser.add_argument('--srcvalfile', help="Path to source validation data.", default='data/dev_src_words.txt')
    parser.add_argument('--targetvalfile', help="Path to target validation data.", default='dev_targ_words.txt')
    parser.add_argument('--batchsize', help="Size of each minibatch.", type=int, default=32)
    parser.add_argument('--seqlength', help="Maximum sequence length. Sequences longer "
                                               "than this are dropped.", type=int, default=50)
    parser.add_argument('--outputfile', help="Prefix of the output file names. ", type=str, default='conv')
    parser.add_argument('--maxwordlength', help="For the character models, words are "
                                           "(if longer than maxwordlength) or zero-padded "
                                            "(if shorter) to maxwordlength", type=int, default=35)
    parser.add_argument('--srcvocabfile', help="If working with a preset vocab, "
                                          "then including this will ignore srcvocabsize and use the"
                                          "vocab provided here.",
                                          type = str, default='src.dict')
    parser.add_argument('--targetvocabfile', help="If working with a preset vocab, "
                                         "then including this will ignore targetvocabsize and "
                                         "use the vocab provided here.",
                                          type = str, default='targ.dict')
    parser.add_argument('--unkfilter', help="Ignore sentences with too many UNK tokens. "
                                       "Can be an absolute count limit (if > 1) "
                                       "or a proportional limit (0 < unkfilter < 1).",
                                          type = float, default = 0)
    parser.add_argument('--data_directory', help="Folder of MovieTriples", default='../data/MovieTriples/')
    
    args = parser.parse_args(arguments)
    args.srcfile = args.output_directory + args.srcfile
    args.targetfile = args.output_directory + args.targetfile
    args.srcvalfile = args.output_directory + args.srcvalfile
    args.targetvalfile = args.output_directory + args.targetvalfile
    args.outputfile = args.output_directory + args.outputfile
    args.srcvocabfile = args.output_directory + args.srcvocabfile
    args.targetvocabfile = args.output_directory + args.targetvocabfile

    data_directory = args.data_directory
    train_valid_split = 0.8
    format_data(data_directory, train_valid_split, args.seqlength, args)
    get_data(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
