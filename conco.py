# -*- coding: utf-8 -*-
'''
 * @Author: Yizhe Zhang 
 * @Date: 2018-09-07 11:59:19 
 * @Last Modified by:   Yizhe Zhang 
 * @Last Modified time: 2018-09-07 11:59:19 
 * @Desc: controllable response generation with CoCon
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
    prepare_for_bleu, cal_BLEU_4, cal_entropy, sent2idx, _clip_gradients_seperate_norm, merge_two_dicts, read_test, reshaping
import copy
import argparse
import socket
import datetime
from cons_discriminator import encoder
from s2s import Options, s2s

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model.')
    parser.add_argument('--gpuid', '-g', type=int, default=0)  
    parser.add_argument('--data_name', '-dn', default="newdata2/twitter_full.p")
    parser.add_argument('--continuing', '-c', action='store_true', default=False)
    parser.add_argument('--binary_feature', '-b', action='store_true', default=False)
    parser.add_argument('--global_feature', '-global', action='store_true', default=False)

    parser.add_argument('--learning_rate', '-l', type=float, default=1e-4)
    parser.add_argument('--keep_ratio', '-k', type=float, default=1.0)
    parser.add_argument("--n_hid", '-nh', type=int, default=500)
    parser.add_argument("--lambda_z", type=float, default=0)
    parser.add_argument("--submodel", default="newdata1_NG")
    parser.add_argument("--temp", type=int, default=1)
    parser.add_argument("--local_feature", '-local', action='store_true', default=False)
    parser.add_argument("--n_context", type=int, default=4)
    parser.add_argument("--test", '-t', action='store_true', default=False)
    parser.add_argument('--agg_model', default="mean")
    parser.add_argument('--relax_d', action='store_true', default=False)
    parser.add_argument('--global_d', default="discriminator")
    parser.add_argument('--local_d', default="discriminator_local")
    parser.add_argument("--use_tgt", action='store_true', default=False)
    parser.add_argument("--n_z", type=int, default=100)

    parser.add_argument('--z_loss', default="L2")
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
        self.use_tgt_z = args.use_tgt
        self.test = args.test
        self.z_loss =  args.z_loss # "cross_entropy"  # "L2"
        self.agg_model = args.agg_model    # nn_mean nn_concat mean max
        self.multiple_src = True
        self.name = "conv_s2s_c" + str(self.n_context) + "_dim_" + str(self.n_hid) + ("_g" if self.global_feature else "") + ("_l" if self.local_feature else "") + ("_condtgt" if self.use_tgt_z else "") 
        self.global_d = args.global_d
        self.local_d = args.local_d 


    
    def post(self, args):
        self.num_turn = 11 if "maluuba" in self.data_name else 8
        self.dropout = True if "maluuba" in self.data_name else False
        self.dropout_ratio = 0.5

        

def feature_vector(src, tgt, is_train, W_norm_d, opt, prefix = 'd_'):
    z_tgt, _ = encoder(tgt, W_norm_d, opt, l_temp = 1, num_outputs = opt.n_z, prefix = prefix , is_reuse = None, is_prob=None, is_padded= False)
    def train_graph():
        # tgt feature
        loss = tf.constant(0.0)
        if "nn" in opt.agg_model:
            z_src = cond_src()
            z_pred = z_prediction(z_src, opt)
            loss = tf.reduce_mean(tf.square(z_tgt - z_pred))
        return z_tgt, loss

    def z_prediction(z_src, opt):
        if opt.agg_model == "nn_concat":
            z_src = tf.concat(z_src, 1)  # B* [z_dim * n_context]
            H1 = layers.fully_connected(z_src, num_outputs = opt.n_z*4, activation_fn = tf.nn.relu, scope = 'g_topic_adapter_h1', reuse = tf.AUTO_REUSE)
            return layers.fully_connected(H1, num_outputs = opt.n_z, activation_fn = tf.nn.sigmoid, scope = 'g_topic_adapter_h3', reuse = tf.AUTO_REUSE)
        elif opt.agg_model == "nn_mean": # "nn_mean"
            with tf.variable_scope('g_avg', reuse=tf.AUTO_REUSE):
                avg_w = tf.get_variable('W', [opt.n_context, 1, 1])
            avg_w = tf.nn.softmax(avg_w, dim = 0)
            z_src = tf.stack(z_src) # opt.n_context * B * z_dim 
            weighted_sum = tf.reduce_sum(z_src * avg_w, axis = 0)
            return weighted_sum
        else:  # "nn_mean_fnn"
            with tf.variable_scope('g_avg', reuse=tf.AUTO_REUSE):
                avg_w = tf.get_variable('W', [opt.n_context, 1, 1])
            avg_w = tf.nn.softmax(avg_w, dim = 0)
            z_src = tf.stack(z_src) # opt.n_context * B * z_dim 
            weighted_sum = tf.reduce_sum(z_src * avg_w, axis = 0)
            H1 = layers.fully_connected(weighted_sum, num_outputs = opt.n_z*2, activation_fn = tf.nn.relu, scope = 'g_topic_adapter_h1', reuse = tf.AUTO_REUSE)
            return layers.fully_connected(H1, num_outputs = opt.n_z, activation_fn = tf.nn.sigmoid, scope = 'g_topic_adapter_h3', reuse = tf.AUTO_REUSE)

    def cond_src():
        z = []
        for src_i in src:
            z_i, z_i_soft = encoder(src_i, W_norm_d, opt, l_temp = 1, num_outputs = opt.n_z, prefix = prefix , is_reuse = tf.AUTO_REUSE, is_prob=None)
            if opt.binary_feature: z_i = z_i_soft
            z.append(z_i)
        if opt.local_feature:
            z = [z[i] for i in range(opt.n_context%2, opt.n_context, 2)]
        if "nn" in opt.agg_model:
            z_all = z
        elif opt.agg_model == "mean":
            z_all = tf.reduce_mean(tf.stack(z), axis = 0)  
        elif opt.agg_model == "max":
            z_all = tf.reduce_max(tf.stack(z), axis = 0)  
        else:
            raise ValueError('--agg_model argument error')
            exit(1)
        return z_all

    def test_graph():
        if "nn" in opt.agg_model:
            z_src = cond_src()
            z = z_prediction(z_src, opt)
        else:
            z = cond_src()
        if opt.binary_feature: z = tf.round(z)
        loss = tf.constant(0.0)
        return z, loss

    if not opt.use_tgt_z:
        z_input, loss = tf.cond(tf.equal(is_train, tf.constant(True)), train_graph, test_graph)
    else:
        z_input, loss = train_graph()

    return z_input, z_tgt, loss


def conditional_s2s(src, tgt, is_train, opt, opt_t=None, is_reuse_generator = None):
    if not opt_t: opt_t = opt
    W_norm_d = embedding_only(opt, prefix = 'd_', is_reuse = None) 
    res = {}
    z_all, z_tgt, loss_pred_z = feature_vector(src, tgt, is_train, W_norm_d, opt, prefix = 'd_')
    if opt.local_feature:
        z_all_l, z_tgt_l, loss_pred_z_l = feature_vector(src, tgt, is_train, W_norm_d, opt, prefix = 'l_')   #  B Z 
        z_all = tf.concat([z_all, z_all_l], axis = 1 )
        z_tgt = tf.concat([z_tgt, z_tgt_l], axis = 1 )
        loss_pred_z += loss_pred_z_l 

    if opt.multiple_src:
        syn_sent, syn_one_hot, H_dec, sup_loss, sample_loss, sup_loss_all = s2s(z_all, src, tgt, opt, is_reuse = is_reuse_generator, prefix ='g_')
    else:
        syn_sent, syn_one_hot, H_dec, sup_loss, sample_loss, sup_loss_all = s2s(z_all, src[-1], tgt, opt, is_reuse = is_reuse_generator, prefix ='g_')
    
    is_logit = (opt.z_loss != 'L2')
    if opt.global_feature:
        _, z_hat = encoder(syn_one_hot, W_norm_d, opt,  num_outputs = opt.n_z,  l_temp = 1, prefix = 'd_' , is_reuse = True, is_prob=True, is_padded= False, is_logit=is_logit)
        if opt.local_feature:
            _, z_hat_l = encoder(syn_one_hot, W_norm_d, opt,  num_outputs = opt.n_z,  l_temp = 1, prefix = 'l_' , is_reuse = True, is_prob=True, is_padded= False, is_logit=is_logit)
            z_hat = tf.concat([z_hat, z_hat_l], axis = 1 )  #  B Z 
        
        if opt.z_loss == 'L2':
            z_loss = tf.reduce_mean(tf.square(z_all - z_hat))
        else:
            z_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = z_all, logits = z_hat))
        res['z'] = z_all
        res['z_hat'] = z_hat 
        res['z_tgt'] = z_tgt 
        res['z_loss'] = z_loss
        res['z_loss_pred'] = loss_pred_z



    
    res['syn_sent'] = syn_sent 

    g_cost = sup_loss + (z_loss*opt.lambda_z if opt.global_feature else 0) + loss_pred_z

    tf.summary.scalar('sup_loss', sup_loss)
    if opt.global_feature:
        tf.summary.scalar('z_loss', z_loss)
    summaries = [
        "learning_rate",
        "loss",
    ]
   


    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if 'g_' in var.name]
    train_op_g = layers.optimize_loss(
        g_cost,
        framework.get_global_step(),
        optimizer=opt.optimizer,
        clip_gradients=(lambda grad: _clip_gradients_seperate_norm(grad, opt.clip_grad)) if opt.clip_grad else None,
        variables=(t_vars if opt.relax_d else g_vars),
        learning_rate=opt.lr_g,
        summaries=summaries)



    return res, g_cost, train_op_g


def main():
    #global n_words
    # Prepare training and testing data

    opt = COptions(args)
    opt_t = COptions(args)

    loadpath = (opt.data_dir + "/" + opt.data_name) 
    print "loadpath:" + loadpath
    x = cPickle.load(open(loadpath, "rb"))
    train, val, test = x[0], x[1], x[2]
    wordtoix, ixtoword = x[3], x[4]

    if opt.test:
        test_file = opt.data_dir + "/newdata2/test.txt"  
        test = read_test(test_file, wordtoix)
        test = [ x for x in test if all([2<len(x[t])<opt.maxlen - 4 for t in range(opt.num_turn)])]
    train_filtered = [ x for x in train if all([2<len(x[t])<opt.maxlen - 4 for t in range(opt.num_turn)])]
    val_filtered = [ x for x in val if all([2<len(x[t])<opt.maxlen - 4 for t in range(opt.num_turn)])]
    print ("Train: %d => %d" % (len(train), len(train_filtered)))
    print ("Val: %d => %d" % (len(val), len(val_filtered)))
    train, val = train_filtered, val_filtered
    del train_filtered, val_filtered

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
            res_, gan_cost_g_, train_op_g = conditional_s2s(src_, tgt_, is_train_, opt, opt_t)
            merged = tf.summary.merge_all()


    uidx = 0
    graph_options=tf.GraphOptions(build_cost_model=1)
    config = tf.ConfigProto(log_device_placement = False, allow_soft_placement=True, graph_options=graph_options)
    config.gpu_options.per_process_gpu_memory_fraction = 0.90

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
                    g_vars = [var for var in t_vars if var.name.startswith('g_')]
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
            res_all = []
            for i in range(iter_num):
                test_index = range(i * opt.batch_size,(i+1) * opt.batch_size)
                sents = [val[t] for t in test_index]
                for idx in range(opt.n_context,opt.num_turn):
                    src = [[sents[i][idx-turn] for i in range(opt.batch_size)] for turn in range(opt.n_context,0,-1)]
                    tgt = [sents[i][idx] for i in range(opt.batch_size)]
                    x_batch = [prepare_data_for_cnn(src_i, opt) for src_i in src] # Batch L
                    y_batch = prepare_data_for_rnn(tgt, opt_t, is_add_GO = False) 
                    feed = merge_two_dicts( {i: d for i, d in zip(src_, x_batch)}, {tgt_: y_batch, is_train_: 0}) # do not use False
                    res = sess.run(res_, feed_dict=feed)
                    res_all.extend(res['syn_sent'])
            
            # bp()
            res_all = reshaping(res_all, opt)

            for idx in range(len(test)*(opt.num_turn-opt.n_context)):
                with open(opt.log_path + '.resp.txt', "a") as resp_f:
                    resp_f.write(u' '.join([ixtoword[x] for x in res_all[idx] if x != 0 and x != 2]).encode('utf-8').strip() + ('\n' if idx%(opt.num_turn-opt.n_context) == 0 else '\t') )
            print ("save to:" + opt.log_path + '.resp.txt')
            exit(0)



        for epoch in range(opt.max_epochs):
            print("Starting epoch %d" % epoch)
            kf = get_minibatches_idx(len(train), opt.batch_size, shuffle=True)
            for _, train_index in kf:
                uidx += 1
                sents = [train[t] for t in train_index]
                for idx in range(opt.n_context,opt.num_turn):
                    src = [[sents[i][idx-turn] for i in range(opt.batch_size)] for turn in range(opt.n_context,0,-1)]
                    tgt = [sents[i][idx] for i in range(opt.batch_size)]

                    x_batch = [prepare_data_for_cnn(src_i, opt) for src_i in src] # Batch L

                    y_batch = prepare_data_for_rnn(tgt, opt_t, is_add_GO = False) 

                    feed = merge_two_dicts( {i: d for i, d in zip(src_, x_batch)}, {tgt_: y_batch, is_train_: 1})

                    _, loss_g = sess.run([train_op_g, gan_cost_g_],feed_dict=feed)

                if uidx%opt.print_freq == 0:
                    print("Iteration %d: loss G %f" %(uidx, loss_g))
                    res = sess.run(res_, feed_dict=feed)
                    if opt.global_feature:
                        print "z loss: " + str(res['z_loss']) 
                    if "nn" in opt.agg_model:
                        print "z pred_loss: " + str(res['z_loss_pred']) 
                    print "Source:" + u' '.join([ixtoword[x] for s in x_batch for x in s[0] if x != 0]).encode('utf-8').strip()
                    print "Target:" + u' '.join([ixtoword[x] for x in y_batch[0] if x != 0]).encode('utf-8').strip()
                    print "Generated:" + u' '.join([ixtoword[x] for x in res['syn_sent'][0] if x != 0]).encode('utf-8').strip()
                    print ""



                    sys.stdout.flush()
                    summary = sess.run(merged, feed_dict=feed)
                    train_writer.add_summary(summary, uidx)


                if uidx%opt.valid_freq == 1:
                    VALID_SIZE = 4096
                    valid_multiplier = np.int(np.floor(VALID_SIZE/opt.batch_size))
                    res_all, val_tgt_all, loss_val_g_all = [], [], []
                    if opt.global_feature:
                        z_loss_all = []
                    for val_step in range(valid_multiplier):
                        valid_index = np.random.choice(len(val), opt.batch_size)
                        sents = [val[t] for t in valid_index]
                        for idx in range(opt.n_context,opt.num_turn):
                            src = [[sents[i][idx-turn] for i in range(opt.batch_size)] for turn in range(opt.n_context,0,-1)]
                            tgt = [sents[i][idx] for i in range(opt.batch_size)]


                            val_tgt_all.extend(tgt)

                            x_batch = [prepare_data_for_cnn(src_i, opt) for src_i in src] # Batch L

                            y_batch = prepare_data_for_rnn(tgt, opt_t, is_add_GO = False) 

                            feed = merge_two_dicts( {i: d for i, d in zip(src_, x_batch)}, {tgt_: y_batch, is_train_: 0}) # do not use False

                            loss_val_g = sess.run([gan_cost_g_], feed_dict=feed)
                            loss_val_g_all.append(loss_val_g)

                            res = sess.run(res_, feed_dict=feed)
                            res_all.extend(res['syn_sent'])
                        if opt.global_feature:
                            z_loss_all.append(res['z_loss'])
                

                    print("Validation:  loss G %f " %( np.mean(loss_val_g_all)))
                    if opt.global_feature:
                        print "z loss: " + str(np.mean(z_loss_all))
                    print "Val Source:" + u' '.join([ixtoword[x] for s in x_batch for x in s[0] if x != 0]).encode('utf-8').strip()
                    print "Val Target:" + u' '.join([ixtoword[x] for x in y_batch[0] if x != 0]).encode('utf-8').strip()
                    print "Val Generated:" + u' '.join([ixtoword[x] for x in res['syn_sent'][0] if x != 0]).encode('utf-8').strip()
                    print ""
                    if opt.global_feature:
                        with open(opt.log_path + '.z.txt', "a") as myfile:
                            myfile.write("Iteration" + str(uidx) + "\n")
                            myfile.write("z_loss %f" %(np.mean(z_loss_all))+ "\n")
                            myfile.write("Val Source:" + u' '.join([ixtoword[x] for s in x_batch for x in s[0] if x != 0]).encode('utf-8').strip()+ "\n")
                            myfile.write("Val Target:" + u' '.join([ixtoword[x] for x in y_batch[0] if x != 0]).encode('utf-8').strip()+ "\n")
                            myfile.write("Val Generated:" + u' '.join([ixtoword[x] for x in res['syn_sent'][0] if x != 0]).encode('utf-8').strip()+ "\n")
                            myfile.write("Z_input, Z_recon, Z_tgt")
                            myfile.write(np.array2string(res['z'][0], formatter={'float_kind':lambda x: "%.2f" % x})+ "\n")
                            myfile.write(np.array2string(res['z_hat'][0], formatter={'float_kind':lambda x: "%.2f" % x})+ "\n\n")
                            myfile.write(np.array2string(res['z_tgt'][0], formatter={'float_kind':lambda x: "%.2f" % x})+ "\n\n")

                    val_set = [prepare_for_bleu(s) for s in val_tgt_all]
                    gen = [prepare_for_bleu(s) for s in res_all]
                    [bleu1s,bleu2s,bleu3s,bleu4s] = cal_BLEU_4(gen, {0: val_set}, is_corpus = opt.is_corpus)
                    etp_score, dist_score = cal_entropy(gen)

                    print 'Val BLEU: ' + ' '.join([str(round(it,3)) for it in (bleu1s,bleu2s,bleu3s,bleu4s)])
                    print 'Val Entropy: ' + ' '.join([str(round(it,3)) for it in (etp_score[0],etp_score[1],etp_score[2],etp_score[3])])
                    print 'Val Diversity: ' + ' '.join([str(round(it,3)) for it in (dist_score[0],dist_score[1],dist_score[2],dist_score[3])])
                    print 'Val Avg. length: ' + str(round(np.mean([len([y for y in x if y!=0]) for x in res_all]),3)) 
                    print ""
                    summary = sess.run(merged, feed_dict=feed)
                    summary2 = tf.Summary(value=[tf.Summary.Value(tag="bleu-2", simple_value=bleu2s),tf.Summary.Value(tag="etp-4", simple_value=etp_score[3])])

                    test_writer.add_summary(summary, uidx)
                    test_writer.add_summary(summary2, uidx)

                                
                if uidx%opt.save_freq == 0:
                    saver.save(sess, opt.save_path)

if __name__ == '__main__':
    main()
