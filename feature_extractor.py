
'''
 * @Author: Yizhe Zhang 
 * @Date: 2018-11-26 23:24:19 
 * @Last Modified by:   Yizhe Zhang 
 * @Desc: feature extractor
 '''

import os, sys

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib import framework
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.python.platform import tf_logging as logging
import cPickle
import numpy as np
import os
import codecs

from math import floor
from operator import add
from pdb import set_trace as bp

from model import *

from utils import prepare_data_for_cnn, prepare_data_for_rnn, get_minibatches_idx, normalizing, normalizing_sum, restore_from_save, tensors_key_in_file,\
    prepare_for_bleu, cal_BLEU_4, cal_entropy, cal_relevance, sent2idx, _clip_gradients_seperate_norm, merge_two_dicts, read_test, reshaping

import copy
import argparse
import socket
import datetime
from cons_discriminator import encoder
from s2s import Options, s2s
from s2s_context import feature_vector

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model.')
    parser.add_argument('--gpuid', '-g', type=int, default=0)  
    parser.add_argument('--data_name', '-dn', default="newdata2/twitter_tiny.p")
    parser.add_argument('--continuing', '-c', action='store_true', default=False)
    parser.add_argument('--global_feature', '-global', action='store_true', default=False)
    parser.add_argument('--binary_feature', '-b', action='store_true', default=False)

    parser.add_argument('--learning_rate', '-l', type=float, default=1e-5)
    parser.add_argument('--keep_ratio', '-k', type=float, default=1.0)
    parser.add_argument("--n_hid", '-nh', type=int, default=500)
    parser.add_argument("--lambda_z", type=float, default=1)
    parser.add_argument("--submodel", default="")
    parser.add_argument("--temp", type=int, default=1)
    parser.add_argument("--local_feature", '-local', action='store_true', default=False)
    parser.add_argument("--n_context", type=int, default=4)
    
    parser.add_argument('--agg_model', default="mean")
    parser.add_argument('--relax_d', action='store_true', default=False)
    parser.add_argument('--bit', type=int, default=None)
    parser.add_argument('--use_tgt', action='store_true', default=False)
    parser.add_argument('--feed', action='store_true', default=False)
    parser.add_argument('--file_path', '-f', default=None)
    parser.add_argument("--n_z", type=int, default=100)
    parser.add_argument('--test_file', '-tf', default="/newdata2/test.txt")
    parser.add_argument('--num_turn', type=int, default=8)    

    parser.add_argument('--global_d', default="discriminator")
    parser.add_argument('--local_d', default="discriminator_local")
    args = parser.parse_args()
    print(args)


    profile = False
    logging.set_verbosity(logging.INFO)
    GPUID = args.gpuid
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)


class COptions(Options):
    def init(self, args):
        self.n_context = args.n_context
        self.relax_d = args.relax_d
        self.binary_feature = args.binary_feature
        
        self.load_from_pretrain = True
        self.use_tgt_z = args.use_tgt
        self.test = True
        self.z_loss =  "L2" 
        self.agg_model = args.agg_model
        self.multiple_src = True
        self.name = "conv_s2s_c" + str(self.n_context) + "_dim_" + str(self.n_hid) + ("_g" if self.global_feature else "") + ("_l" if self.local_feature else "") 
        self.bit = args.bit
        self.int_num = 2
        self.global_d = args.global_d
        self.local_d = args.local_d
        self.feed_generated = args.feed
        self.test_file = args.test_file
        self.num_turn = args.num_turn
        self.verbose = False
        self.embedding_score = True

    def post(self, args):
        self.log_path = self.log_path + ("_condtgt" if self.use_tgt_z else "") + ("_feed" if self.feed_generated else "") + ("_bin" if self.binary_feature else "")   
        self.data_name = args.data_name
        if args.file_path:
            self.save_path = args.file_path + self.name
            self.log_path = args.file_path + self.name + ("_condtgt" if self.use_tgt_z else "") + ("_feed" if self.feed_generated else "") + ("_bin" if self.binary_feature else "")   


def get_features(src, tgt, is_train, opt, opt_t=None, is_reuse_generator = None):
    if not opt_t: opt_t = opt
    W_norm_d = embedding_only(opt, prefix = 'd_', is_reuse = None) 
    res = {}
    z_all, _ = encoder(src[0], W_norm_d, opt, l_temp = 1, num_outputs = opt.n_z, prefix = 'd_' , is_reuse = tf.AUTO_REUSE, is_prob=None)
    z_all_l, _ = encoder(src[0], W_norm_d, opt, l_temp = 1, num_outputs = opt.n_z, prefix = 'l_' , is_reuse = tf.AUTO_REUSE, is_prob=None)
    res['z'] = z_all
    res['z_l'] = z_all_l
    return res
    

def main():
    
    

    opt = COptions(args)
    opt_t = COptions(args)
    

    loadpath = (opt.data_dir + "/" + opt.data_name) 
    print "loadpath:" + loadpath
    x = cPickle.load(open(loadpath, "rb"))
    train, val, test = x[0], x[1], x[2]
    wordtoix, ixtoword = x[3], x[4]

    if opt.test:
        test_file = opt.data_dir + opt.test_file 
        test = read_test(test_file, wordtoix)
        
    opt.n_words = len(ixtoword) 
    opt_t.n_words = len(ixtoword)
    opt_t.maxlen = opt_t.maxlen - opt_t.filter_shape + 1
    opt_t.update_params(args)
    print datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    print dict(opt)
    print('Total words: %d' % opt.n_words)

  
    for d in ['/gpu:0']:
        with tf.device(d):
            src_ = [tf.placeholder(tf.int32, shape=[opt.batch_size, opt.sent_len]) for _ in range(opt.n_context)]
            tgt_ = tf.placeholder(tf.int32, shape=[opt_t.batch_size, opt_t.sent_len])
            
            is_train_ = tf.placeholder(tf.bool, name = 'is_train')
            res_1_ = get_features(src_, tgt_, is_train_, opt, opt_t)
            merged = tf.summary.merge_all()

    uidx = 0
    graph_options=tf.GraphOptions(build_cost_model=1)
    
    config = tf.ConfigProto(log_device_placement = False, allow_soft_placement=True, graph_options=graph_options)
    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=np.inf)
    saver = tf.train.Saver()

    run_metadata = tf.RunMetadata()

    with tf.Session(config = config) as sess:
        train_writer = tf.summary.FileWriter(opt.log_path + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(opt.log_path + '/test', sess.graph)
        sess.run(tf.global_variables_initializer())
        if opt.restore:
            try:       
                t_vars = tf.trainable_variables()  
                if opt.load_from_pretrain:
                    d_vars = [var for var in t_vars if var.name.startswith('d_')]
                    l_vars = [var for var in t_vars if var.name.startswith('l_')]
                    restore_from_save(d_vars, sess, opt, load_path = opt.restore_dir + "/save/" + opt.global_d)
                    if opt.local_feature:
                        restore_from_save(l_vars, sess, opt, load_path = opt.restore_dir + "/save/" + opt.local_d)
                else:
                    loader = restore_from_save(t_vars, sess, opt, load_path = opt.save_path)

            except Exception as e:
                print 'Error: '+str(e)
                print("No saving session, using random initialization")
                sess.run(tf.global_variables_initializer())
        loss_d , loss_g = 0, 0

        if opt.test:
            iter_num = np.int(np.floor(len(test)/opt.batch_size))+1 
            z_all, z_all_l = [], []
            for i in range(iter_num):
                test_index = range(i * opt.batch_size,(i+1) * opt.batch_size)
                sents = [test[t%len(test)] for t in test_index]
                src = [[sents[i][0] for i in range(opt.batch_size)]]
                tgt = [sents[i][0] for i in range(opt.batch_size)]
                x_batch = [prepare_data_for_cnn(src_i, opt) for src_i in src] 
                print "Source:" + u' '.join([ixtoword[x] for s in x_batch for x in s[0] if x != 0]).encode('utf-8').strip()
                y_batch = prepare_data_for_rnn(tgt, opt_t, is_add_GO = False) 
                feed = merge_two_dicts( {i: d for i, d in zip(src_, x_batch)}, {tgt_: y_batch, is_train_: 0}) 
                res_1 = sess.run(res_1_, feed_dict=feed)
                z_all.extend(res_1['z'])  
                z_all_l.extend(res_1['z_l'])                        

            save_path_z = opt.log_path + '.global.z.txt'
            print save_path_z
            if os.path.exists(save_path_z):
                os.remove(save_path_z) 
            with open(save_path_z, "a") as myfile:
                for line in z_all[:len(test)]:
                    for z_it in line:
                        myfile.write(str(z_it) + '\t')
                    myfile.write('\n')
            
            save_path_z = opt.log_path + '.local.z.txt'
            print save_path_z
            if os.path.exists(save_path_z):
                os.remove(save_path_z) 
            with open(save_path_z, "a") as myfile:
                for line in z_all_l[:len(test)]:
                    for z_it in line:
                        myfile.write(str(z_it) + '\t')
                    myfile.write('\n')
       
if __name__ == '__main__':
    main()
