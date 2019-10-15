# -*- coding: utf-8 -*-
'''
 * @Author: Yizhe Zhang 
 * @Date: 2018-08-15 13:53:07 
 * @Last Modified by:   Xiang Gao
 * @Last Modified time: 2018-11-13
 '''

import os, sys, argparse

GPUID = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers

from tensorflow.contrib import framework
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.tensorboard.plugins import projector

import cPickle
import numpy as np
import os
import scipy.io as sio
from math import floor
import pdb
import socket
import datetime

from model import *
from model_attn import *
from utils import prepare_data_for_cnn, prepare_data_for_rnn, get_minibatches_idx, normalizing, restore_from_save, \
    prepare_for_bleu, cal_BLEU, sent2idx, tensors_key_in_file, binary_round


logging.set_verbosity(logging.INFO)
# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS


class Options(object):
    def __init__(self, args):

        self.binary_feature = bool(args.bin) # Binary value # 'C' Continuous value
        self.model = args.model # 'D': DSSM 'C' Concat 'N' innerprod
        self.task = args.task # L for local  C for consecutive pair   G for global

        self.fix_emb = False
        self.reuse_w = False
        self.reuse_cnn = False
        self.reuse_discrimination = True  # reuse cnn for discrimination
        self.restore = bool(args.restore)
        self.tanh = True  # activation fun for the top layer of cnn, otherwise relu

    

        self.maxlen = 49 #113#253
        self.n_words = None
        self.filter_shape = 5
        self.filter_size = 300
        self.embed_size = 300
        self.lr = args.lr
        self.stride = [2, 2, 2]  # for two layer cnn/deconv , use self.stride[0]
        self.batch_size = args.batch_size
        self.max_epochs = args.max_epochs
        # self.n_gan = 900  # self.filter_size * 3
        self.n_hid = args.n_hid

        self.reg = args.reg
        self.reg_corr = args.reg_corr

        hostname = socket.gethostname()
        #not_philly = hostname == 'MININT-3LHNLKS' or 'HP-Z8' in hostname or 'GCR' in hostname
        not_philly = (os.getenv('PT_DATA_DIR') is None)
        temp_data_dir = "/home/yizhe/ssd0/consistent_dialog/preprocessing" if 'HP-Z8' in hostname else "."
        temp_output_dir = "/home/yizhe/ssd0/consistent_dialog/output/" if 'HP-Z8' in hostname else "output"
        self.data_dir = temp_data_dir if not_philly else os.getenv('PT_DATA_DIR') + '/cons'
        self.restore_dir = temp_output_dir if not_philly else os.getenv('PT_DATA_DIR') + '/cons/model'
        self.output_dir = temp_output_dir if not_philly else os.getenv('PT_OUTPUT_DIR')
        self.data_name = args.data_name

        self.name = "d_" + "dim_" + str(self.n_hid) + "_filter_" + str(self.filter_size) + "_" + self.model + "_task_" + "01"+ self.task + ("_Bin" if self.binary_feature else "")
        if args.encoder != 'conv':
            self.name += '_' + args.encoder

        print('@'*20) 
        print('name: '+self.name)
        print('hostname: '+hostname)
        print('data_dir: '+str(self.data_dir))
        if self.restore:
            print('restore_dir: '+str(self.restore_dir))
        print('output_dir: '+str(self.output_dir))
        print('@'*20) 
        
        self.save_path = self.output_dir + '/' + self.name + "/save/" + self.name
        self.log_path = self.output_dir + '/' + self.name + "/log/" + self.name
        if args.load_path == '':
            self.load_path = self.restore_dir + '/' + self.name + "/save/" + self.name
        else:
            self.load_path = args.load_path
            assert(self.restore)

        #sys.stdout = open(self.log_path + 'stdout.txt', 'w')

        self.verbose = False
        self.print_freq = 100
        self.valid_freq = 1000


        # annealed ST estimation
        self.l_temp =  1
        self.l_temp_max = 1
        self.l_temp_factor = 1.0

        self.test = bool(args.test)
        
        # batch norm & dropout
        self.batch_norm = False
        self.cnn_layer_dropout = True
        self.dropout = False
        self.dropout_ratio = args.keep_ratio     # 1. means no dropout
        self.is_train = True

        # self.discrimination = False
        self.H_dis = 200

        self.sent_len = self.maxlen + 2 * (self.filter_shape - 1)
        self.sent_len2 = np.int32(floor((self.sent_len - self.filter_shape) / self.stride[0]) + 1)
        self.sent_len3 = np.int32(floor((self.sent_len2 - self.filter_shape) / self.stride[1]) + 1)
        self.sent_len4 = np.int32(floor((self.sent_len3 - self.filter_shape)/self.stride[2]) + 1)

        self.encoder = args.encoder
        if self.encoder == 'conv':
            self.n_conv_layer = 3
            print ('Use %d conv/deconv layers' % self.n_conv_layer)
        elif self.encoder == 'attn':
            self.n_block = args.n_block
            self.n_head = 10         # should n_hid % n_head == 0
            self.attn_pdrop = 0.1   # used OpenAI default val
            self.resid_pdrop = 0.1  # used OpenAI default val
            self.afn ='gelu'        # used OpenAI default val
            print ('Use %d self_attn layers, %i heads' % (self.n_block, self.n_head))
        else:
            raise ValueError


    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value



def cons_disc(x_1, x_2, y, opt, l_temp = 1):
    # print x.get_shape()  # batch L

    res = {}
    
    logits, H_1, H_2, H_1_1, H_2_1 = pair_discriminator(x_1, x_2, opt, l_temp)

    corr1 = correlation_cost(H_1_1)
    corr2 = correlation_cost(H_2_1)

    res['logits'] = logits
    res['y_pred'] = (logits > 0)
    # res['H_1'] = H_1
    # res['H_2'] = H_2
    res['H_1'] = H_1_1
    res['H_2'] = H_2_1
    res['corr'] = tf.sqrt((corr1 + corr2)/2)
    
    if opt.model == 'D':
        y_pred = logits
        loss = tf.reduce_mean(y * tf.log(y_pred))
    else:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = logits)) 
  
    # encourage binary and disentangle
    loss = loss \
            + opt.reg* tf.reduce_mean( tf.square(tf.ones_like(H_1_1)-H_1_1) * tf.square(H_1_1) ) \
            + opt.reg* tf.reduce_mean( tf.square(tf.ones_like(H_2_1)-H_2_1) * tf.square(H_2_1) )
    if opt.reg_corr != 0:
        loss += opt.reg_corr* (corr1 + corr2)



    tf.summary.scalar('loss', loss)
    
    train_op = layers.optimize_loss(
        loss,
        framework.get_global_step(),
        optimizer='Adam',
        learning_rate=opt.lr)

    return res, loss, train_op

def pair_discriminator(src, tgt, opt, l_temp = 1, prefix = 'd_', is_prob_src = False, is_prob_tgt= False, is_reuse = None):
    W_norm_d = embedding_only(opt, prefix = prefix, is_reuse = is_reuse)   # V E
    H_src, H_src_1 = encoder(src, W_norm_d, opt, l_temp = l_temp, prefix = prefix , is_reuse = is_reuse, is_prob=is_prob_src)
    H_tgt, H_tgt_1 = encoder(tgt, W_norm_d, opt, l_temp = l_temp, prefix = prefix , is_reuse = True, is_prob=is_prob_tgt)

    if opt.model == 'D':
        logits = tf.reduce_sum(normalizing(H_src, 1)*normalizing(H_tgt, 1),1)
    elif opt.model == 'C':
        logits = classifier_2layer(tf.concat([H_src, H_tgt], 1), opt, prefix= prefix, is_reuse = is_reuse)
        logits = tf.squeeze(logits)
    else:  # N

        logits = tf.reduce_sum((H_src*H_tgt - 0.5),1)

    return logits, H_src, H_tgt, H_src_1, H_tgt_1   #tf.squeeze(tf.concat([H_src, H_tgt], 1))

def encoder(x, W_norm_d, opt, l_temp = 1, num_outputs = None, prefix = 'd_', is_prob = False, is_reuse = None, is_padded = True, is_logit = False):
    if not num_outputs: num_outputs = opt.n_hid
    if is_prob:
        x_emb = tf.tensordot(x, W_norm_d, [[2],[0]])
    else:
        x_emb = tf.nn.embedding_lookup(W_norm_d, x)   # batch L emb
    if not is_padded:  # pad the input with pad_emb
        pad_emb = tf.expand_dims(tf.expand_dims(W_norm_d[0],0),0) # 1*v
        x_emb = tf.concat([tf.tile(pad_emb, [opt.batch_size, opt.filter_shape-1, 1]), x_emb],1)


    if opt.encoder == 'conv':
        x_emb = tf.expand_dims(x_emb,3)   # [batch, L, emb, 1]
        if opt.n_conv_layer == 3:
            H = conv_model_3layer(x_emb, opt, prefix = prefix, is_reuse = is_reuse, num_outputs = num_outputs)
                                          # [batch, 1, 1, n_hid]
        else:
            raise NotImplementedError
    elif opt.encoder == 'attn':
        H = self_attn(x_emb, opt, prefix=prefix, is_reuse=is_reuse)
                                          # [batch, n_hid]
    else:
        raise ValueError

    # Note here last layer is a linear one
    H_logit = H*l_temp
    H = tf.nn.sigmoid(H_logit)


    if opt.binary_feature:
        
        H_b = binary_round(H)
        #H = binary_round(H) *2 - 1
    else:
        H_b = H
    if is_logit:
        return tf.squeeze(H_b), tf.squeeze(H_logit)
    else:
        return tf.squeeze(H_b), tf.squeeze(H)


def rand_pair(task, data_name):
    if 'twitter' in data_name:
        # totally 8 turns
        match    = [(1, 5), (1, 7), (2, 6), (2, 8), (3, 7), (4, 8)]
        mismatch = [(1, 6), (1, 8), (2, 7), (3, 8)]
    elif 'maluuba' in data_name:
        # totally 11 turns,
        # no first turn, as it's always "hello how may i help you ?"
        match    = [(2, 6), (2, 8), (2, 10), (3, 7), (3, 9), (3, 11), (4, 8), (4, 10), (5, 9), (5, 11), (6, 10)]
        mismatch = [(2, 7), (2, 9), (2, 11), (3, 8), (3, 10), (4, 9), (4, 11)]

    if task == 'L':
        if np.random.random() > 0.5:
            pairs = match
        else:
            pairs = mismatch
    elif task == 'G':
        pairs = match + mismatch
    else:
        raise ValueError

    # reverse so that C knows order doesn't matter
    pairs += [(j, i) for i, j in pairs]

    i, j = pairs[np.random.randint(len(pairs))]
    return i - 1, j - 1  # zero-based

def main(opt):
    # global n_words
    # Prepare training and testing data
    
    
    data_path = opt.data_dir + "/" + opt.data_name
    print('loading '+data_path)
    x = cPickle.load(open(data_path, "rb"))
    train, val, test = x[0], x[1], x[2]
    wordtoix, ixtoword = x[3], x[4]


    opt.n_words = len(ixtoword) 
    print datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    print dict(opt)
    print('Total words: %d' % opt.n_words)

    with tf.device('/gpu:1'):
        x_1_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.sent_len])
        x_2_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.sent_len])
        y_ = tf.placeholder(tf.float32, shape=[opt.batch_size,])
        l_temp_ = tf.placeholder(tf.float32, shape=[])
        res_, loss_ ,train_op = cons_disc(x_1_, x_2_, y_, opt, l_temp_)
        merged = tf.summary.merge_all()

    

    uidx = 0
    
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    
    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=np.inf)
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(opt.log_path + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(opt.log_path + '/test', sess.graph)
        sess.run(tf.global_variables_initializer()) # feed_dict={x_: np.zeros([opt.batch_size, opt.sent_len]), x_org_: np.zeros([opt.batch_size, opt.sent_len])}

        if opt.restore:
            print('-'*20)
            print("Loading variables from '%s'." % opt.load_path)
            try:
                #pdb.set_trace()
                t_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) #tf.trainable_variables()
                #print([var.name[:-2] for var in t_vars]              
                save_keys = tensors_key_in_file(opt.load_path)
                ss = [var for var in t_vars if var.name[:-2] in save_keys.keys()]
                ss = [var.name for var in ss if var.get_shape() == save_keys[var.name[:-2]]]
                loader = tf.train.Saver(var_list= [var for var in t_vars if var.name in ss])
                loader.restore(sess, opt.load_path)
                print("Loaded variables:"+str(ss))
                print('-'*20)

            except Exception as e:
                print 'Error: '+str(e)
                exit()
                print("No saving session, using random initialization")
                sess.run(tf.global_variables_initializer())

        # train
        # if don't want to train, set max_epochs=0

        for epoch in range(opt.max_epochs):
            print("Starting epoch %d" % epoch)
            opt.l_temp = min(opt.l_temp * opt.l_temp_factor, opt.l_temp_max)
            print("Annealing temperature " + str(opt.l_temp))
            kf = get_minibatches_idx(len(train), opt.batch_size, shuffle=True)
            for _, train_index in kf:
                uidx += 1
                sents = [train[t] for t in train_index]
                indice = [rand_pair(opt.task, opt.data_name) for _ in range(opt.batch_size)]
                if opt.task == 'L':

                    x_1 = [sents[i][idx[0]] for i, idx in enumerate(indice)]
                    x_2 = [sents[i][idx[1]] for i, idx in enumerate(indice)]
                    y_batch = [(i1-i2)%2 == 0 for i1,i2 in indice]
                elif opt.task == 'C':

                    batch_indice = np.concatenate([np.random.permutation(opt.batch_size/2) , range(opt.batch_size/2, opt.batch_size)]) 
                    y_batch = (range(opt.batch_size) == batch_indice)

                    rn = np.random.choice(7,size = opt.batch_size)
 
                    x_1 = [sents[i][idx[0]] for i, idx in enumerate(indice)]
                    x_2 = [sents[batch_indice[i]][idx[1]] for i, idx in enumerate(indice)]
                else: # G
                    batch_indice = np.concatenate([np.random.permutation(opt.batch_size/2) , range(opt.batch_size/2, opt.batch_size)]) 
                    y_batch = (range(opt.batch_size) == batch_indice)
              
                    x_1 = [sents[i][idx[0]] for i, idx in enumerate(indice)]
                    x_2 = [sents[batch_indice[i]][idx[1]] for i, idx in enumerate(indice)]
                x_1_batch = prepare_data_for_cnn(x_1, opt)  # Batch L
                x_2_batch = prepare_data_for_cnn(x_2, opt)  # Batch L

                feed = {x_1_: x_1_batch, x_2_: x_2_batch, y_:np.float32(y_batch),l_temp_:opt.l_temp}
                _, loss = sess.run([train_op, loss_], feed_dict=feed)



                if uidx % opt.print_freq == 1:
                    print("Iteration %d: loss %f " % (uidx, loss))
                    res = sess.run(res_, feed_dict=feed)
                    if opt.verbose:
                        print("logits:" + str(res['logits']))
                        print("H1:" + str(res['H_1'][0]))
                        print("H2:" + str(res['H_2'][0]))
                    # print("H2:" + str(res['H_1'][0]*res['H_2'][0]-0.5))
                    acc = sum(np.equal(res['y_pred'],y_batch))/np.float(opt.batch_size)
                    print("Accuracy: %f" % (acc))
                    print("y_mean: %f" % (np.mean(y_batch)))
                    print("corr:" + str(res['corr']))

                    sys.stdout.flush()
                    summary = sess.run(merged, feed_dict=feed)
                    train_writer.add_summary(summary, uidx)

                if uidx % opt.valid_freq == 1:
                    acc, loss_val, y_mean, corr = 0, 0, 0, 0
                    indice = [rand_pair(opt.task, opt.data_name) for _ in range(opt.batch_size)]
                    for i in range(100):
                        valid_index = np.random.choice(len(test), opt.batch_size)
                        sents = [test[t] for t in valid_index]
                        if opt.task == 'L':
                           
                            x_1 = [sents[i][idx[0]] for i, idx in enumerate(indice)]
                            x_2 = [sents[i][idx[1]] for i, idx in enumerate(indice)]
                            y_batch = [(i1-i2)%2 == 0 for i1,i2 in indice]
                        elif opt.task == 'C':
     
                            batch_indice = np.concatenate([np.random.permutation(opt.batch_size/2) , range(opt.batch_size/2, opt.batch_size)]) 
                            y_batch = (range(opt.batch_size) == batch_indice)
                     
                            rn = np.random.choice(7,size = opt.batch_size)
                        
                            x_1 = [sents[i][idx[0]] for i, idx in enumerate(indice)]
                            x_2 = [sents[batch_indice[i]][idx[1]] for i, idx in enumerate(indice)]
                        else: # G
                            batch_indice = np.concatenate([np.random.permutation(opt.batch_size/2) , range(opt.batch_size/2, opt.batch_size)]) 
                            y_batch = (range(opt.batch_size) == batch_indice)
                            x_1 = [sents[i][idx[0]] for i, idx in enumerate(indice)]
                            x_2 = [sents[batch_indice[i]][idx[1]] for i, idx in enumerate(indice)]

                        x_1_batch = prepare_data_for_cnn(x_1, opt)  # Batch L
                        x_2_batch = prepare_data_for_cnn(x_2, opt)  # Batch L

                        feed = {x_1_: x_1_batch, x_2_: x_2_batch, y_:np.float32(y_batch),l_temp_:opt.l_temp}
                        loss_val += sess.run(loss_, feed_dict=feed)
                        res = sess.run(res_, feed_dict=feed)
                        acc += sum(np.equal(res['y_pred'],y_batch))/np.float(opt.batch_size)
                        y_mean += np.mean(y_batch)
                        corr += res['corr']

                    loss_val = loss_val / 100.0
                    acc = acc / 100.0
                    y_mean = y_mean / 100.0
                    corr = corr / 100.0
                    print("Validation loss %.4f " % (loss_val))
                    print("Validation accuracy: %.4f" % (acc))
                    print("Validation y_mean: %.4f" % (y_mean))
                    print("Validation corr: %.4f" % (corr))
                    print("")
                    sys.stdout.flush()
                    
                    summary = sess.run(merged, feed_dict=feed)
                    test_writer.add_summary(summary, uidx)

            saver.save(sess, opt.save_path, global_step=epoch)


        # test

        if opt.test:
            print('Testing....')
            iter_num = np.int(np.floor(len(test)/opt.batch_size))+1
            for i in range(iter_num):
                if i%100 == 0:
                    print('Iter %i/%i'%(i, iter_num))
                test_index = range(i*opt.batch_size, (i+1)*opt.batch_size)
                test_sents = [test[t%len(test)] for t in test_index]
                indice = [(0,1),(2,3),(4,5),(6,7)]
                for idx in indice:
                    x_1 = [test_sents[i][idx[0]] for i in range(opt.batch_size)]
                    x_2 = [test_sents[i][idx[1]] for i in range(opt.batch_size)]
                    y_batch = [True for i in range(opt.batch_size)]
                    x_1_batch = prepare_data_for_cnn(x_1, opt)  # Batch L
                    x_2_batch = prepare_data_for_cnn(x_2, opt)  # Batch L

                    feed = {x_1_: x_1_batch, x_2_: x_2_batch, y_:np.float32(y_batch), l_temp_:opt.l_temp}
                    res = sess.run(res_, feed_dict=feed)
                    for d in range(opt.batch_size):
                        with open(opt.log_path + '.feature.txt', "a") as myfile:
                            myfile.write(str(test_index[d]) + "\t" + str(idx[0]) + "\t" + " ".join([ixtoword[x] for x in x_1_batch[d] if x != 0]) + "\t" + " ".join(map(str,res['H_1'][d]))+ "\n")
                            myfile.write(str(test_index[d]) + "\t" + str(idx[1]) + "\t" + " ".join([ixtoword[x] for x in x_2_batch[d] if x != 0]) + "\t" + " ".join(map(str,res['H_2'][d]))+ "\n")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='N')            # 'D': DSSM 'C' Concat 'N' innerprod
    parser.add_argument("--task", default='L')             # L for local  C for consecutive pair   G for global
    parser.add_argument("--bin", type=int, default=0)      # 1 for true, 0 for false
    parser.add_argument("--test", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--restore", type=int, default=0)
    parser.add_argument("--keep_ratio", type=float, default=1.)
    parser.add_argument("--data_name", default="twitter_small.p")
    parser.add_argument("--n_hid", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder", default='conv')        # 'conv' or 'attn'
    parser.add_argument("--n_block", type=int, default=3)        # 'conv' or 'attn'
    parser.add_argument('--reg', type=float, default=0.0)
    parser.add_argument('--reg_corr', type=float, default=0.0)
    parser.add_argument('--load_path',default='')
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    opt = Options(args)
    main(opt)
