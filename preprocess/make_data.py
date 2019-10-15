# -*- coding: utf-8 -*-
"""
Created on Mon May  1 01:53:46 2017

@author: DinghanShen
"""

import csv
import numpy as np
import os
import re
import cPickle
import string
import pdb
import sys
import operator


MAXLEN = 300
VOC_SIZE = 20000

#==============================================================================
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9():,\.!?\t]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"e\.g\.,", " ", string)
    string = re.sub(r"a\.k\.a\.", " ", string)
    string = re.sub(r"i\.e\.,", " ", string)
    string = re.sub(r"i\.e\.", " ", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"br", "", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r":", " : ", string)
    #string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"u\.s\.", " us ", string)
    return string.strip().lower()



loadpath = sys.argv[1]

x = []
with open(loadpath, 'rb') as f:
    for line in f:
        x.append(clean_str(line))

np.random.shuffle(x)

#lab = []
# sent = []
# vocab = {}





# for i in range(60000):
#     lab.append(clean_str(x[i].split(",")[0]))
#     m = re.search(',', x[i])
#     temp = clean_str(x[i][m.start()+1:]).split()
#     if len(temp) > 300:
#         temp = temp[:300]
#         sent.append(temp)
#     else:
#         sent.append(temp)
#     t = set(temp)
#     for word in t:
#         if word in vocab:
#             vocab[word] += 1
#         else:
#             vocab[word] = 1

# loadpath = 'train.csv'

# x = []
# with open(loadpath, 'rb') as f:
#     for line in f:
#         x.append(line)


sent = []
vocab = {}
for i in range(len(x)):
    # train_lab.append(clean_str(x[i].split(",")[0]))
    t = x[i].split()
    line = x[i].split('\t')
    if len(line) == 8 and all([len(z)<MAXLEN for z in line]):
        sent.append(line)
    for word in t:
        if word in vocab:
            vocab[word] += 1
        else:
            vocab[word] = 1
    
#==============================================================================
# pdb.set_trace()

print('create ixtoword and wordtoix lists...')

# v = [x for x, y in vocab.iteritems() if y >= 30]

# create ixtoword and wordtoix lists
ixtoword = {}
# period at the end of the sentence. make first dimension be end token
ixtoword[0] = '_PAD'
ixtoword[1] = '_GO'
ixtoword[2] = '_END'
ixtoword[3] = '_UNK'
wordtoix = {}
wordtoix['_PAD'] = 0
wordtoix['_GO'] = 1
wordtoix['_END'] = 2
wordtoix['_UNK'] = 3
ix = 4
for w,_ in sorted(vocab.items(), key=operator.itemgetter(1),reverse=True):
    if ix < VOC_SIZE:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1
pdb.set_trace()
def convert_word_to_ix(data):
    result = []
    for conv in data:
        temp_c = []
        for sent in conv:
            temp = []
            for w in sent.split():
                if w in wordtoix:
                    temp.append(wordtoix[w])
                else:
                    temp.append(3)
            temp.append(2)
            temp_c.append(temp)
        result.append(temp_c)
    return result

train_x = sent[:800000]
# train_y = train_lab[:1300000]
val_x = sent[800000:900000]
# val_y = train_lab[1300000:]
test_x = sent[900000:1000000]
# test_y = lab

train_x = convert_word_to_ix(train_x)
val_x = convert_word_to_ix(val_x)
test_x = convert_word_to_ix(test_x)
#pdb.set_trace()

cPickle.dump([train_x, val_x, test_x, wordtoix, ixtoword], open("twitter_small.p", "wb"))

a,b = len(sent)-20000 , len(sent)-10000 
train_x = sent[:a]
# train_y = train_lab[:1300000]
val_x = sent[a:b]
# val_y = train_lab[1300000:]
test_x = sent[b:]
# test_y = lab

train_x = convert_word_to_ix(train_x)
val_x = convert_word_to_ix(val_x)
test_x = convert_word_to_ix(test_x)


cPickle.dump([train_x, val_x, test_x, wordtoix, ixtoword], open("twitter_full.p", "wb"))