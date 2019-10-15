# -*- coding: utf-8 -*-
'''
 * @Author: Yizhe Zhang 
 * @Date: 2018-10-22 23:24:19 
 * @Last Modified by:   Yizhe Zhang 
 * @Last Modified time: 2018-10-22 23:24:19 
 * @Desc: controllable s2s 
 '''


## 152.3.214.203/6006

import os, sys

# os.environ['LD_LIBRARY_PATH'] = '/home/yizhe/cudnn/cuda/lib64'
# os.environ['CPATH'] = '/home/yizhe/cudnn/cuda/include'
# os.environ['LIBRARY_PATH'] = '/home/yizhe/cudnn/cuda'
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
#import scipy.io as sio
from math import floor
from operator import add
from pdb import set_trace as bp

from model import *

from utils import prepare_data_for_cnn, prepare_data_for_rnn, get_minibatches_idx, normalizing, normalizing_sum, restore_from_save, tensors_key_in_file,\
    prepare_for_bleu, cal_BLEU_4, cal_entropy, cal_relevance, sent2idx, _clip_gradients_seperate_norm, merge_two_dicts, read_test, reshaping
# import gensim
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
    parser.add_argument("--submodel", default="newdata1_NG")
    parser.add_argument("--temp", type=int, default=1)
    parser.add_argument("--local_feature", '-local', action='store_true', default=False)
    parser.add_argument("--n_context", type=int, default=4)
    # parser.add_argument("--test", '-t', action='store_true', default=False)
    parser.add_argument('--agg_model', default="mean")
    parser.add_argument('--relax_d', action='store_true', default=False)
    parser.add_argument('--bit', type=int, default=None)
    parser.add_argument('--use_tgt', action='store_true', default=False)
    parser.add_argument('--feed', action='store_true', default=False)
    parser.add_argument('--file_path', '-f', default=None)
    parser.add_argument("--n_z", type=int, default=100)
    parser.add_argument('--test_file', '-tf', default="/newdata2/test.txt")
    parser.add_argument('--num_turn', type=int, default=8)    

    # parser.add_argument('--global_d', default="discriminator")
    # parser.add_argument('--local_d', default="discriminator_local")
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
        # self.maxlen = 33 + self.n_context*16
        self.load_from_pretrain = False
        self.use_tgt_z = args.use_tgt
        self.test = True
        self.z_loss =  "L2" # "cross_entropy"  # "L2"
        self.agg_model = args.agg_model
        self.multiple_src = True
        self.name = "conv_s2s_c" + str(self.n_context) + "_dim_" + str(self.n_hid) + ("_g" if self.global_feature else "") + ("_l" if self.local_feature else "") 
        self.bit = args.bit
        self.int_num = 2
        # self.global_d = "discriminator"
        # self.local_d = "discriminator_local"
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
    z_all, z_tgt, loss_pred_z = feature_vector(src, tgt, is_train, W_norm_d, opt, prefix = 'd_')
    if opt.local_feature:
        z_all_l, z_tgt_l, loss_pred_z_l = feature_vector(src, tgt, is_train, W_norm_d, opt, prefix = 'l_')   #  B Z 
        z_all = tf.concat([z_all, z_all_l], axis = 1 )
        z_tgt = tf.concat([z_tgt, z_tgt_l], axis = 1 )
        loss_pred_z += loss_pred_z_l 
    res['z'] = z_all
    res['z_tgt'] = z_tgt 
    res['z_loss_pred'] = loss_pred_z
    return res
    
def generate_resp(src, tgt, z_all, is_train, opt, opt_t=None, is_reuse_generator = None):
    W_norm_d = embedding_only(opt, prefix = 'd_', is_reuse = True) 
    # for idx, val in enumerate(np.linspace(0.0, 1.0, num=opt.int_num)):
    #with tf.variable_scope('z_temp', reuse=None):
    # z_temp = tf.get_variable("z_temp", dtype = tf.float32, shape=(opt.batch_size, opt.n_z), trainable = False, initializer = tf.zeros_initializer)
    # #z_temp = z_temp[:,opt.bit].assign(tf.constant(val, dtype = tf.float32, shape=(opt.batch_size, 1)))
    # z_temp = z_temp[:,:].assign(z_all)
    # z_temp = z_temp[:,opt.bit].assign(val)
    res = {}
    if opt.multiple_src:
        syn_sent, syn_one_hot, H_dec, sup_loss, sample_loss, sup_loss_all = s2s(z_all, src, tgt, opt, is_softargmax = False, is_reuse = is_reuse_generator, prefix ='g_')
    else:
        syn_sent, syn_one_hot, H_dec, sup_loss, sample_loss, sup_loss_all = s2s(z_all, src[-1], tgt, opt, is_softargmax = False, is_reuse = is_reuse_generator, prefix ='g_')
    
    is_logit = (opt.z_loss != 'L2')
    if opt.global_feature:
        z_hat, _ = encoder(syn_one_hot, W_norm_d, opt,  num_outputs = opt.n_z,  l_temp = 1, prefix = 'd_' , is_reuse = True, is_prob=True, is_padded= False, is_logit=is_logit)
        if opt.local_feature:
            z_hat_l, _ = encoder(syn_one_hot, W_norm_d, opt,  num_outputs = opt.n_z,  l_temp = 1, prefix = 'l_' , is_reuse = True, is_prob=True, is_padded= False, is_logit=is_logit)
            z_hat = tf.concat([z_hat, z_hat_l], axis = 1 )  #  B Z 
        
        if opt.z_loss == 'L2':
            z_loss = tf.reduce_mean(tf.square(z_all - z_hat))
        else:
            z_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = z_all, logits = z_hat))

        res['z_hat'] = z_hat 
    #     res['z_loss'] = z_loss
    res['syn_sent'] = syn_sent 
    return res


def main():
    #global n_words
    # Prepare training and testing data

    opt = COptions(args)
    opt_t = COptions(args)
    # opt_t.n_hid = opt.n_z

    loadpath = (opt.data_dir + "/" + opt.data_name) #if opt.not_philly else '/hdfs/msrlabs/xiag/pt-data/cons/data_cleaned/twitter_small.p'
    print "loadpath:" + loadpath
    x = cPickle.load(open(loadpath, "rb"))
    train, val, test = x[0], x[1], x[2]
    wordtoix, ixtoword = x[3], x[4]

    if opt.test:
        test_file = opt.data_dir + opt.test_file 
        test = read_test(test_file, wordtoix)
        # test = [ x for x in test if all([2<len(x[t])<opt.maxlen - 4 for t in range(opt.num_turn)])]
    # train_filtered = [ x for x in train if all([2<len(x[t])<opt.maxlen - 4 for t in range(opt.num_turn)])]
    # val_filtered = [ x for x in val if all([2<len(x[t])<opt.maxlen - 4 for t in range(opt.num_turn)])]
    # print ("Train: %d => %d" % (len(train), len(train_filtered)))
    # print ("Val: %d => %d" % (len(val), len(val_filtered)))
    # train, val = train_filtered, val_filtered
    # del train_filtered, val_filtered

    opt.n_words = len(ixtoword) 
    opt_t.n_words = len(ixtoword)
    opt_t.maxlen = opt_t.maxlen - opt_t.filter_shape + 1
    opt_t.update_params(args)
    print datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    print dict(opt)
    print('Total words: %d' % opt.n_words)

    # print dict(opt)
    # if opt.model == 'cnn_rnn':
    #     opt_t.maxlen = opt_t.maxlen - opt_t.filter_shape + 1
    #     opt_t.update_params(args)
        # print dict(opt_t)


    #for d in ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']:
    for d in ['/gpu:0']:
        with tf.device(d):
            src_ = [tf.placeholder(tf.int32, shape=[opt.batch_size, opt.sent_len]) for _ in range(opt.n_context)]
            tgt_ = tf.placeholder(tf.int32, shape=[opt_t.batch_size, opt_t.sent_len])
            z_ = tf.placeholder(tf.float32, shape=[opt_t.batch_size , opt.n_z * (2 if opt.local_feature else 1)])
            is_train_ = tf.placeholder(tf.bool, name = 'is_train')
            res_1_ = get_features(src_, tgt_, is_train_, opt, opt_t)
            res_2_ = generate_resp(src_, tgt_, z_, is_train_, opt, opt_t)
            merged = tf.summary.merge_all()

    #tensorboard --logdir=run1:/tmp/tensorflow/ --port 6006
    #writer = tf.train.SummaryWriter(opt.log_path, graph=tf.get_default_graph())

    uidx = 0
    graph_options=tf.GraphOptions(build_cost_model=1)
    #config = tf.ConfigProto(log_device_placement = False, allow_soft_placement=True, graph_options=tf.GraphOptions(build_cost_model=1))
    config = tf.ConfigProto(log_device_placement = False, allow_soft_placement=True, graph_options=graph_options)
    # config.gpu_options.per_process_gpu_memory_fraction = 0.70
    #config = tf.ConfigProto(device_count={'GPU':0})
    #config.gpu_options.allow_growth = True

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
                #pdb.set_trace()
                t_vars = tf.trainable_variables()  
                #t_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) #tf.trainable_variables()

                # if opt.load_from_pretrain:
                #     d_vars = [var for var in t_vars if var.name.startswith('d_')]
                #     g_vars = [var for var in t_vars if var.name.startswith('g_')]
                #     g_vars = [var for var in t_vars if var.name.startswith('g_')]
                #     g_vars = [var for var in t_vars if var.name.startswith('g_')]
                #     g_vars = [var for var in t_vars if var.name.startswith('g_')]
                #     g_vars = [var for var in t_vars if var.name.startswith('g_')]
                #     l_vars = [var for var in t_vars if var.name.startswith('l_')]
                #     #restore_from_save(g_vars, sess, opt, prefix = 'g_', load_path=opt.restore_dir + "/save/generator2")
                #     restore_from_save(d_vars, sess, opt, load_path = opt.restore_dir + "/save/" + opt.global_d)
                #     if opt.local_feature:
                #         restore_from_save(l_vars, sess, opt, load_path = opt.restore_dir + "/save/" + opt.local_d)
                # else:
                loader = restore_from_save(t_vars, sess, opt, load_path = opt.save_path)


            except Exception as e:
                print 'Error: '+str(e)
                print("No saving session, using random initialization")
                sess.run(tf.global_variables_initializer())
        loss_d , loss_g = 0, 0

        if opt.test:
            iter_num = np.int(np.floor(len(test)/opt.batch_size))+1
            res_all = []
            val_tgt_all =[]
            for i in range(iter_num):
                test_index = range(i * opt.batch_size,(i+1) * opt.batch_size)
                sents = [test[t%len(test)] for t in test_index]
                for idx in range(opt.n_context,opt.num_turn):
                    src = [[sents[i][idx-turn] for i in range(opt.batch_size)] for turn in range(opt.n_context,0,-1)]
                    tgt = [sents[i][idx] for i in range(opt.batch_size)] 
                    val_tgt_all.extend(tgt)
                    if opt.feed_generated and idx!= opt.n_context:
                        src[-1] = [[x for x in p if x!=0] for p in res_all[-opt.batch_size:]]

                    x_batch = [prepare_data_for_cnn(src_i, opt) for src_i in src] # Batch L
                    y_batch = prepare_data_for_rnn(tgt, opt_t, is_add_GO = False) 
                    
                    feed = merge_two_dicts( {i: d for i, d in zip(src_, x_batch)}, {tgt_: y_batch, is_train_: 0}) # do not use False
                    res_1 = sess.run(res_1_, feed_dict=feed)
                    z_all = np.array(res_1['z'])

                    
                    feed = merge_two_dicts( {i: d for i, d in zip(src_, x_batch)}, {tgt_: y_batch, z_: z_all, is_train_: 0}) # do not use False
                    res_2 = sess.run(res_2_, feed_dict=feed)
                    res_all.extend(res_2['syn_sent'])

                    # bp()
   
            val_tgt_all = reshaping(val_tgt_all, opt)
            res_all = reshaping(res_all, opt)
            
            save_path = opt.log_path + '.resp.txt'
            if os.path.exists(save_path):
                os.remove(save_path) 
            for idx in range(len(test)*(opt.num_turn-opt.n_context)):
                with open(save_path, "a") as resp_f:
                    resp_f.write(u' '.join([ixtoword[x] for x in res_all[idx] if x != 0 and x != 2]).encode('utf-8').strip() + ('\n' if idx%(opt.num_turn-opt.n_context) == opt.num_turn-opt.n_context-1 else '\t') )
            print ("save to:" + save_path)

            if opt.verbose:
                save_path = opt.log_path + '.tgt.txt'
                if os.path.exists(save_path):
                    os.remove(save_path) 
                for idx in range(len(test)*(opt.num_turn-opt.n_context)):
                    with open(save_path, "a") as tgt_f:
                        tgt_f.write(u' '.join([ixtoword[x] for x in val_tgt_all[idx] if x != 0 and x != 2]).encode('utf-8').strip() + ('\n' if idx%(opt.num_turn-opt.n_context) == opt.num_turn-opt.n_context-1 else '\t') )
                print ("save to:" + save_path)

            val_set = [prepare_for_bleu(s) for s in val_tgt_all]
            gen = [prepare_for_bleu(s) for s in res_all]
            [bleu1s,bleu2s,bleu3s,bleu4s] = cal_BLEU_4(gen, {0: val_set}, is_corpus = opt.is_corpus)
            etp_score, dist_score = cal_entropy(gen)

            # print save_path
            print 'Val BLEU: ' + ' '.join([str(round(it,3)) for it in (bleu1s,bleu2s,bleu3s,bleu4s)])
            # print 'Val Rouge: ' + ' '.join([str(round(it,3)) for it in (rouge1,rouge2,rouge3,rouge4)])
            print 'Val Entropy: ' + ' '.join([str(round(it,3)) for it in (etp_score[0],etp_score[1],etp_score[2],etp_score[3])])
            print 'Val Diversity: ' + ' '.join([str(round(it,3)) for it in (dist_score[0],dist_score[1],dist_score[2],dist_score[3])])
            # print 'Val Relevance(G,A,E): ' + ' '.join([str(round(it,3)) for it in (rel_score[0],rel_score[1],rel_score[2])])
            print 'Val Avg. length: ' + str(round(np.mean([len([y for y in x if y!=0]) for x in res_all]),3)) 
            if opt.embedding_score:
                with open("../../ssd0/consistent_dialog/data/GoogleNews-vectors-negative300.bin.p", 'rb') as pfile:
                    embedding = cPickle.load(pfile)
                rel_score = cal_relevance(gen, val_set, embedding)
                print 'Val Relevance(G,A,E): ' + ' '.join([str(round(it,3)) for it in (rel_score[0],rel_score[1],rel_score[2])])


            if not opt.global_feature or opt.bit == None: exit(0)

        if opt.test:
            iter_num = np.int(np.floor(len(test)/opt.batch_size))+1 
            for int_idx in range(opt.int_num):
                res_all = []
                z1,z2,z3 = [],[],[]
                val_tgt_all =[]
                for i in range(iter_num):
                    test_index = range(i * opt.batch_size,(i+1) * opt.batch_size)
                    sents = [test[t%len(test)] for t in test_index]
                    for idx in range(opt.n_context,opt.num_turn):
                        src = [[sents[i][idx-turn] for i in range(opt.batch_size)] for turn in range(opt.n_context,0,-1)]
                        tgt = [sents[i][idx] for i in range(opt.batch_size)]
                        val_tgt_all.extend(tgt)
                        if opt.feed_generated and idx!= opt.n_context:
                            src[-1] = [[x for x in p if x!=0] for p in res_all[-opt.batch_size:]]

                        x_batch = [prepare_data_for_cnn(src_i, opt) for src_i in src] # Batch L
                        y_batch = prepare_data_for_rnn(tgt, opt_t, is_add_GO = False) 
                        feed = merge_two_dicts( {i: d for i, d in zip(src_, x_batch)}, {tgt_: y_batch, is_train_: 0}) # do not use False
                        res_1 = sess.run(res_1_, feed_dict=feed)
                        z_all = np.array(res_1['z'])
                        z_all[:,opt.bit] = np.array([1.0/np.float(opt.int_num-1) * int_idx for _ in range(opt.batch_size)])
                        
                        feed = merge_two_dicts( {i: d for i, d in zip(src_, x_batch)}, {tgt_: y_batch, z_: z_all, is_train_: 0}) # do not use False
                        res_2 = sess.run(res_2_, feed_dict=feed)
                        res_all.extend(res_2['syn_sent'])
                        z1.extend(res_1['z'])                        
                        z2.extend(z_all)
                        z3.extend(res_2['z_hat'])
                        
                        # bp()

                val_tgt_all = reshaping(val_tgt_all, opt)
                res_all = reshaping(res_all, opt)
                z1 = reshaping(z1, opt)
                z2 = reshaping(z2, opt)
                z3 = reshaping(z3, opt)
                
                save_path = opt.log_path  + 'bit' + str(opt.bit) + '.'+ str(1.0/np.float(opt.int_num-1) * int_idx) +'.int.txt'
                if os.path.exists(save_path):
                    os.remove(save_path) 
                for idx in range(len(test)*(opt.num_turn-opt.n_context)):
                    with open(save_path, "a") as resp_f:
                        resp_f.write(u' '.join([ixtoword[x] for x in res_all[idx] if x != 0 and x != 2]).encode('utf-8').strip() + ('\n' if idx%(opt.num_turn-opt.n_context) == opt.num_turn-opt.n_context-1 else '\t') )
                print ("save to:" + save_path)

                save_path_z = opt.log_path  + 'bit' + str(opt.bit) + '.'+ str(1.0/np.float(opt.int_num-1) * int_idx) +'.z.txt'
                if os.path.exists(save_path_z):
                    os.remove(save_path_z) 
                for idx in range(len(test)*(opt.num_turn-opt.n_context)):
                    with open(save_path_z, "a") as myfile:
                        #ary = np.array([z1[idx][opt.bit], z2[idx][opt.bit], z3[idx][opt.bit]])
                        #myfile.write(np.array2string(ary, formatter={'float_kind':lambda x: "%.2f" % x}) + ('\n' if idx%(opt.num_turn-opt.n_context) == opt.num_turn-opt.n_context-1 else '\t'))
                        myfile.write(str(z3[idx][opt.bit]) + ('\n' if idx%(opt.num_turn-opt.n_context) == opt.num_turn-opt.n_context-1 else '\t'))

                
                val_set = [prepare_for_bleu(s) for s in val_tgt_all]
                gen = [prepare_for_bleu(s) for s in res_all]
                [bleu1s,bleu2s,bleu3s,bleu4s] = cal_BLEU_4(gen, {0: val_set}, is_corpus = opt.is_corpus)
                etp_score, dist_score = cal_entropy(gen)

                print save_path
                print 'Val BLEU: ' + ' '.join([str(round(it,3)) for it in (bleu1s,bleu2s,bleu3s,bleu4s)])
                # print 'Val Rouge: ' + ' '.join([str(round(it,3)) for it in (rouge1,rouge2,rouge3,rouge4)])
                print 'Val Entropy: ' + ' '.join([str(round(it,3)) for it in (etp_score[0],etp_score[1],etp_score[2],etp_score[3])])
                print 'Val Diversity: ' + ' '.join([str(round(it,3)) for it in (dist_score[0],dist_score[1],dist_score[2],dist_score[3])])
                # print 'Val Relevance(G,A,E): ' + ' '.join([str(round(it,3)) for it in (rel_score[0],rel_score[1],rel_score[2])])
                print 'Val Avg. length: ' + str(round(np.mean([len([y for y in x if y!=0]) for x in res_all]),3)) 

        

       
if __name__ == '__main__':
    main()
