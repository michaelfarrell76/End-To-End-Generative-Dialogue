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
        sent_thrown_out = 0
        for _, (src_orig, targ_orig) in \
                enumerate(itertools.izip(open(srcfile,'r'), open(targetfile,'r'))):
            src_orig = src_indexer.clean(src_orig.decode("utf-8").strip())
            targ_orig = target_indexer.clean(targ_orig.decode("utf-8").strip())
            targ = targ_orig.strip().split()
            src = src_orig.strip().split()
            if len(targ) > seqlength or len(src) > seqlength or len(targ) < 1 or len(src) < 1:
                sent_thrown_out = sent_thrown_out + 1
                continue
            num_sents += 1
            for word in targ:                                
                target_indexer.vocab[word] += 1
                
            for word in src:                 
                src_indexer.vocab[word] += 1
        print('Number of sentences thrown out', sent_thrown_out)
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
                if sent_id > 3000000:
                    break

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

    if args.subtle:
        max_word_l, num_sents_subtle = make_vocab(args.srcsubtlefile, args.targetsubtlefile,
                                                 args.seqlength, max_word_l, 0)
        print("Number of sentences in subtle: {}".format(num_sents_subtle))    

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
    print("Max sent length: {}".format(max_sent_l))    

    if args.subtle:
        max_sent_l = convert(args.srcsubtlefile, args.targetsubtlefile, args.batchsize, args.subtle_seqlength,
                             args.outputfile + "-subtle.hdf5", num_sents_subtle, max_word_l,
                             max_sent_l, 0, args.unkfilter)

    
    print("Max sent length for subtle: {}".format(max_sent_l))    


def format_data(args):
    '''
        Formats the data so it can be passed into get_data function
    '''

    def clean_word_mappings(data_dict):
        '''
            Function makes necessary changes to the word_mappings
            so that they match up with the MovieTriples design. 
            The changes to the mappings are then applied to the
            training validation and test set of MovieTriples.

            Function could be changed to remove some of the hard-coding
        '''

        # Words that need to be altered
        bad_words = ['<unk>', '<s>', '</s>', '.', "'"]

        #the new mappings that need to be added
        new_mappings = [('<blank>', 1, 0, 0), ('<unk>', 2, 190588, 89059), ('<s>', 3, 588827, 785135), ('</s>', 4, 588827, 785135), ('.', 10003, 855616, 192250), ("'", 10004, 457542, 160249), ('<t>', 10005, 0, 0)]

        # Bad incides
        del_ind = []
        for i in range(len(data_dict['MovieTriples']['word_mappings'])):
            word_mapping = data_dict['MovieTriples']['word_mappings'][i]
            if word_mapping[0] in bad_words:
                    del_ind.append(i)

        # Delete the bad indicies in reverse order
        del_ind.sort(reverse=True)
        for ind in del_ind:
            del data_dict['MovieTriples']['word_mappings'][ind]

        # Add the new mappings
        for n_m in new_mappings:    
            data_dict['MovieTriples']['word_mappings'].append(n_m)

        # Sanity check
        check_mappings = range(1, len(data_dict['MovieTriples']['word_mappings']) + 1)
        for word_mapping in data_dict['MovieTriples']['word_mappings']:
            check_mappings.remove(word_mapping[1])
        assert check_mappings == []

        # Make changes to the dataset
        data_sets = [data_dict['MovieTriples']['train_set'], data_dict['MovieTriples']['valid_set'],
                     data_dict['MovieTriples']['test_set']]
        if args.subtle:
            data_sets.append(data_dict['MovieTriples']['subtle_set'])

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

        return data_dict


    def load_data():
        ''' 
            Assuming all inputs files are .pkl files 

            Loads all data files into a data_dict thats first layer
            is the name of the data folder that is processed, and the 
            second layer is the variable name of the loaded file. 

            i.e. 

            data_dict['MovieTriples']['train_set'] = pickle.load(...)
        '''
        data_dict = {}
        for data_set in args.input_files:
            data_dict[data_set] = {}
            for var_name, data_file in args.input_files[data_set].iteritems():
                with open('%s%s/%s' % (args.input_directory, data_set, data_file)) as f:
                    data_dict[data_set][var_name] = pickle.load(f)

        return data_dict

    def write_indicies_to_file(filename, y):
        '''
            Write the contents of y into filename
        '''
        f =  open(filename, 'w')
        for context in y: 
            for ind in context:
                f.write(str(ind) + ' ')
            f.write('\n')
        f.close()

    def write_vocab_to_file(filename):
        ''' 
            Write the vocabulary to filename
        '''
        with open(filename, 'w') as f: 
            for i in range(1, len(indices_to_word)+1):
                f.write(indices_to_word[i] + ' ' + str(i) + '\n')

    def write_words_to_file(filename, indices_dict):
        '''
            Write the examples to files as words, removing special
            indices
        '''
        lst = []
        for context in indices_dict:
            context_words = []
            for ind in context:
                if ind not in special_indices:
                    context_words.append(indices_to_word[ind])
            lst.append(' '.join(context_words))


        f =  open(filename, 'w')
        for context in lst: 
            f.write(str(context) + ' \n')
        f.close()

    # Load in datafiles
    data_dict = load_data()

    # Fix the wordmappings for Movie Triples
    data_dict = clean_word_mappings(data_dict)

    # indices -> word
    indices_to_word = {}
    for word_ex in data_dict['MovieTriples']['word_mappings']: 
        indices_to_word[word_ex[1]] = word_ex[0]
        
    # word -> indices
    word_to_indices = {}
    for word_ex in data_dict['MovieTriples']['word_mappings']: 
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
    data_sets = [data_dict['MovieTriples']['train_set'], data_dict['MovieTriples']['valid_set'], 
                    data_dict['MovieTriples']['test_set']]

    if args.subtle:
        data_sets.append(data_dict['MovieTriples']['subtle_set'])

    for j in range(len(data_sets)):
        data_set = data_sets[j]
        PADDING = word_to_indices['<blank>']
        END_OF_CONV = word_to_indices['<t>']

        full_context = []
        full_output = []

        # Make the subtle dataset a bit shorter
        if j != 3:
            max_len_context = 52
            max_len_output = 52
        else:
            # import bpdb; bpdb.set_trace()
            max_len_context = 22
            max_len_output = 22

        for i in range(len(data_set)):
            break_pt = []
            for ind in range(len(data_set[i]))[::-1]:
                if pattern == data_set[i][ind:ind+2]:
                    break_pt.append(ind)

            if break_pt == []:
                continue

            context = data_set[i][:break_pt[0]]
            output = data_set[i][break_pt[0]+2:]

            context = context + [word_to_indices['</s>']]
            output = [word_to_indices['<s>']] + output


            # There is only one utterance in the subtle set
            if j != 3:
                # Start of sentence and end of sentence is ONLY used at the end
                # We create a new character that represents the start and end of a conversation
                context = context[:break_pt[1]] + [END_OF_CONV] + context[break_pt[1]+2:]

            # Cap the target and src length at 302 words to make computation simpler, goes up to ~1500
            if len(context) > max_len_context:
                continue
            if len(output) > max_len_output:
                continue

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

    if args.subtle:
        subtle_full_context = data_set_contexts[3]
        subtle_full_output = data_set_outputs[3]

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    write_indicies_to_file(args.srcfile_ind, train_full_context)
    write_indicies_to_file(args.targetfile_ind, train_full_output)
    write_indicies_to_file(args.srcvalfile_ind, valid_full_context)
    write_indicies_to_file(args.targetvalfile_ind, valid_full_output)

    if args.subtle:
        write_indicies_to_file(args.srcsubtlefile_ind, subtle_full_context)
        write_indicies_to_file(args.targetsubtlefile_ind, subtle_full_output)

    write_vocab_to_file(args.targetvocabfile)
    write_vocab_to_file(args.srcvocabfile)
            
    special_indices = [1, 2, 3, 4,]

    write_words_to_file(args.srcfile, train_full_context)
    write_words_to_file(args.srcvalfile, valid_full_context)
    write_words_to_file(args.targetfile, train_full_output)
    write_words_to_file(args.targetvalfile, valid_full_output)

    if args.subtle:
        write_words_to_file(args.srcsubtlefile, subtle_full_context)    
        write_words_to_file(args.targetsubtlefile, subtle_full_output)
            
    # Additional embeddings
    np.random.seed(9844)
    additional_vectors = np.random.uniform(-0.1, 0.1, (2, 300))
    additional_vectors = additional_vectors / np.linalg.norm(additional_vectors)

    emb_wordvec_upd = np.vstack((data_dict['MovieTriples']['emb_wordvec'][0], additional_vectors))
    emb_wordvec_upd[10001] = data_dict['MovieTriples']['emb_wordvec'][0][3][:]
    emb_wordvec_upd[10002] = data_dict['MovieTriples']['emb_wordvec'][0][4][:]
    emb_wordvec_upd[3] = data_dict['MovieTriples']['emb_wordvec'][0][10001][:]
    emb_wordvec_upd[4] = data_dict['MovieTriples']['emb_wordvec'][0][10002][:]
    emb_wordvec_upd = np.roll(emb_wordvec_upd, 1, axis=0)
    f = h5py.File(args.output_directory + 'word_vecs.hdf5', 'w')
    f['word_vecs'] = emb_wordvec_upd
    f.close()

    print('Done formatting the data from Movie Triples dataset')


def main(arguments):
    '''
        Main function parses the command arguments and sets up the 
        paths to each of the input/output files. Output files can be set
        via command line while input files should be hard coded into the 
        preprocessing file
    '''
    parser = argparse.ArgumentParser(
                description=__doc__,
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Directories of input raw files and output processed files
    parser.add_argument('--input_directory', help="Path to the folder that contains all of "
                                                   "the raw data files to be used for preprocessing", 
                                                   default='../data/')
    parser.add_argument('--output_directory', help="Path to the folder that will hold all of "
                                                    "the datafiles generated by this program",
                                                default='data/')
    parser.add_argument('--outputfile', help="Prefix of the output file names. ", type=str, default='conv')

    # Vocabularys
    parser.add_argument('--srcvocabsize', help="Size of source vocabulary, constructed "
                                                "by taking the top X most frequent words. "
                                                " Rest are replaced with special UNK tokens.",
                                                type=int, default=10000)
    parser.add_argument('--targetvocabsize', help="Size of target vocabulary, constructed "
                                                "by taking the top X most frequent words. "
                                                "Rest are replaced with special UNK tokens.",
                                                type=int, default=10000)
    parser.add_argument('--srcvocabfile', help="If working with a preset vocab, "
                                          "then including this will ignore srcvocabsize and use the"
                                          "vocab provided here.",
                                          type = str, default='src.dict')
    parser.add_argument('--targetvocabfile', help="If working with a preset vocab, "
                                         "then including this will ignore targetvocabsize and "
                                         "use the vocab provided here.",
                                          type = str, default='targ.dict')

    # Filenames of processed files filled with words
    parser.add_argument('--srcfile', help="Filename of source training data, "
                                           "where each line represents a single "
                                           "source/target sequence. Will be placed in "
                                           "the args.output_directory folder",
                                           default='train_src_words.txt')
    parser.add_argument('--targetfile', help="Path to target training data, "
                                           "where each line represents a single "
                                           "source/target sequence. Will be placed in "
                                           "the args.output_directory folder", 
                                           default='train_targ_words.txt')
    parser.add_argument('--srcvalfile', help="Filename of source validation data.", default='dev_src_words.txt')
    parser.add_argument('--targetvalfile', help="Filename of target validation data.", default='dev_targ_words.txt')
    parser.add_argument('--srcsubtlefile', help="Filename of source subtle data.", default='subtle_src_words.txt')
    parser.add_argument('--targetsubtlefile', help="Filename of target subtle data.", default='subtle_targ_words.txt')

    # Filenames of processed files filled with indices
    parser.add_argument('--srcfile_ind', help="Filename of source training data, "
                                           "where each line represents a single "
                                           "source/target sequence. Will be placed in "
                                           "the args.output_directory folder",
                                           default='train_src_indices.txt')
    parser.add_argument('--targetfile_ind', help="Path to target training data, "
                                           "where each line represents a single "
                                           "source/target sequence. Will be placed in "
                                           "the args.output_directory folder", 
                                           default='train_targ_indices.txt')
    parser.add_argument('--srcvalfile_ind', help="Filename of source validation data.", default='dev_src_indices.txt')
    parser.add_argument('--targetvalfile_ind', help="Filename of target validation data.", default='dev_targ_indices.txt')
    parser.add_argument('--srcsubtlefile_ind', help="Filename of source subtle data.", default='subtle_src_indices.txt')
    parser.add_argument('--targetsubtlefile_ind', help="Filename of target subtle data.", default='subtle_targ_indices.txt')

    # Preprocess Modifications
    parser.add_argument('--batchsize', help="Size of each minibatch.", type=int, default=64)
    parser.add_argument('--seqlength', help="Maximum sequence length. Sequences longer "
                                               "than this are dropped.", type=int, default=50)
    parser.add_argument('--subtle_seqlength', help="Maximum sequence length for subtle data. Sequences longer "
                                               "than this are dropped.", type=int, default=15)
    parser.add_argument('--maxwordlength', help="For the character models, words are "
                                           "(if longer than maxwordlength) or zero-padded "
                                            "(if shorter) to maxwordlength", type=int, default=35)
    parser.add_argument('--unkfilter', help="Ignore sentences with too many UNK tokens. "
                                       "Can be an absolute count limit (if > 1) "
                                       "or a proportional limit (0 < unkfilter < 1).",
                                          type = float, default = 0)
    parser.add_argument('--subtle', help="Preprocess the subtle dataset, takes substantially longer"
                                         "because subtle dataset is ~5,000,000 lines", default=False)

    args = parser.parse_args(arguments)

    # Dictionary holding the locations of the input files
    args.input_files = { 'MovieTriples' : { 'train_set' : 'Training.triples.pkl', 
                                           'valid_set' : 'Validation.triples.pkl',
                                           'test_set' : 'Test.triples.pkl',
                                           'emb_wordvec' : 'Word2Vec_WordEmb.pkl',
                                           'emb_mt' : 'MT_WordEmb.pkl',
                                           'word_mappings' : 'Training.dict.pkl'
                                          }
                        }

    # Append on output directory to the output files
    output_files = ['srcfile', 'targetfile', 'srcvalfile', 'targetvalfile', 'srcfile_ind', 'targetfile_ind', 
                    'srcvalfile_ind', 'targetvalfile_ind', 'outputfile', 'srcvocabfile', 'targetvocabfile']

    if args.subtle:
        args.input_files['MovieTriples']['subtle_set'] = 'Subtle_Dataset.triples.pkl'
        output_files = output_files + ['srcsubtlefile_ind', 'targetsubtlefile_ind', 'srcsubtlefile', 'targetsubtlefile']
                    
    for o_f in output_files:
        setattr(args, o_f, args.output_directory + getattr(args, o_f))

    # print args.output_directory
    # Call preprocessing code
    format_data(args)
    get_data(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
