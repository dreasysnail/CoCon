'''
 * @Author: Yizhe Zhang 
 * @Date: 2018-09-07 11:59:19 
 * @Last Modified by:   Yizhe Zhang 
 * @Last Modified time: 2018-09-07 11:59:19 
 * @Desc: controllable s2s 
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
    prepare_for_bleu, cal_BLEU_4, cal_entropy, sent2idx, _clip_gradients_seperate_norm
import copy
import argparse
import socket
import datetime
from cons_discriminator import encoder
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model.')
    parser.add_argument('--gpuid', '-g', type=int, default=0)  
    parser.add_argument('--data_name', '-dn', default="newdata2/twitter_full.p")
    parser.add_argument('--continuing', '-c', action='store_true', default=False)
    parser.add_argument('--global_feature', '-global', action='store_true', default=False)
    
    parser.add_argument('--learning_rate', '-l', type=float, default=1e-4)
    parser.add_argument('--keep_ratio', '-k', type=float, default=1.0)
    parser.add_argument("--n_hid", '-nh', type=int, default=100)
    parser.add_argument("--lambda_z", type=int, default=10)
    parser.add_argument("--submodel", default="")
    parser.add_argument("--temp", type=int, default=1)
    parser.add_argument("--local_feature", '-local', action='store_true', default=False)
    parser.add_argument("--n_z", type=int, default=100)
    args = parser.parse_args()
    print(args)
    profile = False
    
    
    
    logging.set_verbosity(logging.INFO)
    
    GPUID = args.gpuid
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)
class Options(object):
    def __init__(self, args):      
        self.encoder = 'conv'
        self.n_conv_layer = 3
        
        self.rnn_share_emb = True 
        self.fix_emb = False
        self.reuse_cnn = False
        self.restore = True
        self.tanh = False  
        self.model = 'cnn_rnn' 
        self.is_fed_h = True
        self.W_emb = None
        self.cnn_W = None
        self.cnn_b = None
        self.maxlen = 49
        self.n_words = None
        self.filter_shape = 5
        self.filter_size = 300
        self.embed_size = 300
        
        self.use_tgt_z = False
        self.binary_feature = False
        self.local_feature = args.local_feature
        
        self.layer = 3
        self.stride = [2, 2, 2]   
        self.batch_size = 32
        self.max_epochs = 50
        self.n_hid = args.n_hid  
        self.multiplier = 2
        self.L = args.temp
        self.batch_norm = False
        self.dropout = False
        self.dropout_ratio = 1
        self.z_prior = 'u' 
        self.n_z = args.n_z
        self.lr_g = args.learning_rate 
        self.bp_truncation = None
        self.lambda_z = args.lambda_z
        self.g_fix = False
        self.global_feature = args.global_feature
        self.optimizer = 'Adam' 
        self.clip_grad = None 
        self.attentive_emb = False
        self.relu_w = False
        self.decode_epsilon = 0
        self.batch_norm = False
        self.cnn_layer_dropout = True
        self.dropout = False
        self.dropout_ratio = args.keep_ratio     
        self.is_train = True
        self.data_size = None 
        self.name = "conv_s2s_" + "dim_" + str(self.n_hid)  + ("_g" if self.global_feature else "") + ("_l" if self.local_feature else "")
        self.reuse_discrimination = False
        self.load_from_pretrain = not args.continuing
        self.print_freq = 100
        self.valid_freq = 2000
        
        self.save_freq = 20000
        self.is_corpus = False 
        self.load_from_ae = False
        self.discrimination = False
        self.test = False
        
    
        self.init(args)
        self.update_params(args)
        self.post(args)
        print ('Use model %s' % self.model)
        print ('Use %d conv/deconv layers' % self.layer)
    def init(self, args):
        pass
    def post(self, args):
        pass
    def update_params(self, args):
        hostname = socket.gethostname()
        self.not_philly = hostname == 'MININT-3LHNLKS' or 'HP-Z8' in hostname or 'GCR' in hostname
        temp_data_dir = "../../ssd0/consistent_dialog/preprocessing" if 'HP-Z8' in hostname else "data"
        temp_output_dir = "../../ssd0/consistent_dialog/output/" + ("maluuba/" if "maluuba" in args.data_name else "") if 'HP-Z8' in hostname else "output"
        self.data_dir = temp_data_dir if self.not_philly else os.getenv('PT_DATA_DIR') + '/cons/data'
        self.restore_dir = temp_output_dir if self.not_philly else os.getenv('PT_DATA_DIR') + '/cons/model'
        self.restore_dir = self.restore_dir + "/" + args.submodel
        self.output_dir = temp_output_dir if self.not_philly else os.getenv('PT_OUTPUT_DIR')
        self.data_name = args.data_name if not self.test else "newdata2/twitter_tiny.p"
        self.save_path = self.output_dir + '/' + self.name + "/save/" + self.name
        self.log_path = self.output_dir + '/' + self.name + "/log/" + self.name
        self.load_path = self.restore_dir + '/' + self.name + "/save/" + self.name
        
        self.sent_len = self.maxlen + 2*(self.filter_shape-1)
        self.sent_len2 = np.int32(floor((self.sent_len - self.filter_shape)/self.stride[0]) + 1)
        self.sent_len3 = np.int32(floor((self.sent_len2 - self.filter_shape)/self.stride[1]) + 1)
        self.sent_len4 = np.int32(floor((self.sent_len3 - self.filter_shape)/self.stride[2]) + 1)
        print('@'*20) 
        print('name: '+self.name)
        print('hostname: '+hostname)
        print('data_dir: '+str(self.data_dir))
        if self.restore:
            print('restore_dir: '+str(self.restore_dir))
        print('output_dir: '+str(self.output_dir))
        print('@'*20) 
    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value
def s2s(z, all_src, tgt, opt, is_prob_src = False, is_reuse = None, is_softargmax = True, is_sampling = False, prefix = 'g_'):      
    z = tf.expand_dims(tf.expand_dims(z,1),1)  
    W_norm = embedding_only(opt, prefix = prefix, is_reuse = is_reuse)  
    res = {}
    H_enc_all = []
    if not isinstance(all_src, list): all_src = [all_src]
    for src in all_src:
        x_emb = tf.nn.embedding_lookup(W_norm, src)
        x_emb = tf.expand_dims(x_emb, 3)  
        H_enc_temp, res = conv_encoder(x_emb, opt, res, prefix = prefix, is_reuse = tf.AUTO_REUSE) 
        H_enc_temp = tf.nn.relu(H_enc_temp)
        H_enc_all.append(H_enc_temp)
    H_enc = tf.concat(H_enc_all, axis=3)
    H_enc = layers.fully_connected(H_enc, num_outputs = opt.n_hid, activation_fn = tf.nn.sigmoid, scope = prefix + 'H_adapter', reuse = is_reuse)
    
    if opt.global_feature:
        H_dec = tf.concat([H_enc, z], axis=3)
    else: 
        H_dec = H_enc
    if not opt.rnn_share_emb:
        W_norm_rnn = embedding_only(opt, prefix = prefix + '_dec', is_reuse = is_reuse)
        W_norm_dec = W_norm_rnn
    else:
        W_norm_dec = W_norm
    
    sup_loss, _, _ , sup_loss_all = lstm_decoder_embedding(H_dec, tgt, W_norm_dec, opt, add_go = True, is_reuse=is_reuse, is_fed_h = opt.is_fed_h, prefix = prefix)
    sample_loss, syn_sent, logits, _ = lstm_decoder_embedding(H_dec, tf.ones_like(tgt), W_norm_dec, opt, add_go = True, feed_previous=True, is_reuse=True, is_softargmax = is_softargmax, is_sampling = is_sampling, is_fed_h = opt.is_fed_h, prefix = prefix)
    prob = [tf.nn.softmax(l*opt.L) for l in logits]
    prob = tf.stack(prob,1)  
    return syn_sent, prob, H_dec, sup_loss, sample_loss, sup_loss_all
def conditional_s2s(src, tgt, z,  opt, opt_t=None, is_reuse_generator = None):
    if not opt_t: opt_t = opt
    res = {}
    if opt.use_tgt_z:
        
        W_norm_d = embedding_only(opt, prefix = 'd_', is_reuse = None) 
        z, _ = encoder(tgt, W_norm_d, opt, l_temp = 1, prefix = 'd_' , is_reuse = None, is_prob=None, is_padded= False)
    syn_sent, syn_one_hot, H_dec, sup_loss, sample_loss, sup_loss_all = s2s(z, src, tgt, opt, is_reuse = is_reuse_generator, prefix ='g_')
    
    if opt.global_feature:
        z_hat, _ = encoder(syn_one_hot, W_norm_d, opt, l_temp = 1, prefix = 'd_' , is_reuse = True, is_prob=True, is_padded= False)
        z_loss = tf.reduce_sum(tf.square(z - z_hat))/opt.batch_size/opt.n_hid
        res['z'] = z
        res['z_hat'] = z_hat 
        res['z_loss'] = z_loss 
    
    res['syn_sent'] = syn_sent 
    g_cost = sup_loss + (z_loss*opt.lambda_z if opt.global_feature else 0)
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
        
        variables=g_vars,
        learning_rate=opt.lr_g,
        summaries=summaries)
    return res, g_cost, train_op_g
def main():
    
    
    opt = Options(args)
    opt_t = Options(args)
    opt_t.n_hid = opt.n_z
    loadpath = (opt.data_dir + "/" + opt.data_name) 
    print "loadpath:" + loadpath
    x = cPickle.load(open(loadpath, "rb"))
    train, val, test = x[0], x[1], x[2]
    wordtoix, ixtoword = x[3], x[4]
    
    
    opt.n_words = len(ixtoword) 
    print datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    print dict(opt)
    print('Total words: %d' % opt.n_words)
    opt.n_words = len(ixtoword)
    opt_t.n_words = len(ixtoword)
    print dict(opt)
    if opt.model == 'cnn_rnn':
        opt_t.maxlen = opt_t.maxlen - opt_t.filter_shape + 1
        opt_t.update_params(args)
        print dict(opt_t)
    print('Total words: %d' % opt.n_words)
    for d in ['/gpu:0']:
        with tf.device(d):
            src_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.sent_len])
            tgt_ = tf.placeholder(tf.int32, shape=[opt_t.batch_size, opt_t.sent_len])
            z_ = tf.placeholder(tf.float32, shape=[opt.batch_size, opt.n_z])
            res_, gan_cost_g_, train_op_g = conditional_s2s(src_, tgt_, z_, opt, opt_t)
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
                    restore_from_save(g_vars, sess, opt, opt.restore_dir + "/save/generator")
                    restore_from_save(d_vars, sess, opt, opt.restore_dir + "/save/discriminator")
                else:
                    loader = restore_from_save(t_vars, sess, opt)  
            except Exception as e:
                print 'Error: '+str(e)
                print("No saving session, using random initialization")
                sess.run(tf.global_variables_initializer())
        loss_d , loss_g = 0, 0
        for epoch in range(opt.max_epochs):
            print("Starting epoch %d" % epoch)
            kf = get_minibatches_idx(len(train), opt.batch_size, shuffle=True)
            for _, train_index in kf:
                uidx += 1
                indice = [(x,x+1) for x in range(7)]
                sents = [train[t] for t in train_index]
                for idx in indice:
                    src = [sents[i][idx[0]] for i in range(opt.batch_size)]
                    tgt = [sents[i][idx[1]] for i in range(opt.batch_size)]
                    src_permutated = src 
                    x_batch = prepare_data_for_cnn(src_permutated, opt) 
                    y_batch = prepare_data_for_rnn(tgt, opt_t, is_add_GO = False) 
                    if opt.z_prior == 'g':
                        z_batch = np.random.normal(0,1,(opt.n_hid, opt.n_z)).astype('float32')
                    else:
                        z_batch = np.random.uniform(-1,1,(opt.batch_size, opt.n_z)).astype('float32')
                    
                    feed = {src_: x_batch, tgt_: y_batch, z_:z_batch}
                    _, loss_g = sess.run([train_op_g, gan_cost_g_],feed_dict=feed)
                if uidx%opt.print_freq == 0:
                    print("Iteration %d: loss G %f" %(uidx, loss_g))
                    res = sess.run(res_, feed_dict=feed)
                    print "Source:" + u' '.join([ixtoword[x] for x in x_batch[0] if x != 0]).encode('utf-8').strip()
                    print "Target:" + u' '.join([ixtoword[x] for x in y_batch[0] if x != 0]).encode('utf-8').strip()
                    print "Generated:" + u' '.join([ixtoword[x] for x in res['syn_sent'][0] if x != 0]).encode('utf-8').strip()
                    print ""
                    sys.stdout.flush()
                    summary = sess.run(merged, feed_dict=feed)
                    train_writer.add_summary(summary, uidx)
                    
                    
                if uidx%opt.valid_freq == 1:
                    VALID_SIZE = 4096
                    valid_multiplier = np.int(np.floor(VALID_SIZE/opt.batch_size))
                    res_all, val_tgt_all, loss_val_d_all, loss_val_g_all = [], [], [], []
                    if opt.global_feature:
                        z_loss_all = []
                    for val_step in range(valid_multiplier):
                        valid_index = np.random.choice(len(test), opt.batch_size)
                        indice = [(x,x+1) for x in range(7)]
                        val_sents = [test[t] for t in valid_index]
                        for idx in indice:
                            val_src = [val_sents[i][idx[0]] for i in range(opt.batch_size)]
                            val_tgt = [val_sents[i][idx[1]] for i in range(opt.batch_size)]
                            val_tgt_all.extend(val_tgt)
                            val_src_permutated = val_src 
                            
                            x_val_batch = prepare_data_for_cnn(val_src, opt) 
                            
                            y_val_batch = prepare_data_for_rnn(val_src, opt_t, is_add_GO = False) if opt.model == 'cnn_rnn' else prepare_data_for_cnn(val_src, opt_t)   
                            if opt.z_prior == 'g':
                                z_val_batch = np.random.normal(0,1,(opt.batch_size, opt.n_z)).astype('float32')
                            else:
                                z_val_batch = np.random.uniform(-1,1,(opt.batch_size, opt.n_z)).astype('float32')
                            feed_val = {src_: x_val_batch, tgt_: y_val_batch, z_:z_val_batch}
                            loss_val_g = sess.run([gan_cost_g_], feed_dict=feed_val)
                            loss_val_g_all.append(loss_val_g)
                            res = sess.run(res_, feed_dict=feed_val)
                            res_all.extend(res['syn_sent'])
                        if opt.global_feature:
                            z_loss_all.append(res['z_loss'])
                        
                    
                    
                    print("Validation:  loss G %f " %( np.mean(loss_val_g_all)))
                    
                    print "Val Source:" + u' '.join([ixtoword[x] for x in val_src[0] if x != 0]).encode('utf-8').strip()
                    print "Val Target :" + u' '.join([ixtoword[x] for x in val_tgt[0] if x != 0]).encode('utf-8').strip()
                    print "Val Generated:" + u' '.join([ixtoword[x] for x in res['syn_sent'][0] if x != 0]).encode('utf-8').strip()
                    print ""
                    if opt.global_feature:
                        with open(opt.log_path + '.z.txt', "a") as myfile:
                            myfile.write("Iteration" + str(uidx) + "\n")
                            myfile.write("z_loss %f" %(np.mean(z_loss_all))+ "\n")
                            myfile.write("Val Source:" + u' '.join([ixtoword[x] for x in val_src[0] if x != 0]).encode('utf-8').strip()+ "\n")
                            myfile.write("Val Target :" + u' '.join([ixtoword[x] for x in val_tgt[0] if x != 0]).encode('utf-8').strip()+ "\n")
                            myfile.write("Val Generated:" + u' '.join([ixtoword[x] for x in res['syn_sent'][0] if x != 0]).encode('utf-8').strip()+ "\n")
                            myfile.write(np.array2string(res['z'][0], formatter={'float_kind':lambda x: "%.2f" % x})+ "\n")
                            myfile.write(np.array2string(res['z_hat'][0], formatter={'float_kind':lambda x: "%.2f" % x})+ "\n\n")
                    val_set = [prepare_for_bleu(s) for s in val_tgt_all]
                    gen = [prepare_for_bleu(s) for s in res_all]
                    
                    
                    
                    [bleu1s,bleu2s,bleu3s,bleu4s] = cal_BLEU_4(gen, {0: val_set}, is_corpus = opt.is_corpus)
                    
                    etp_score, dist_score = cal_entropy(gen)
                    
                    
                    
                    print 'Val BLEU: ' + ' '.join([str(round(it,3)) for it in (bleu1s,bleu2s,bleu3s,bleu4s)])
                    
                    print 'Val Entropy: ' + ' '.join([str(round(it,3)) for it in (etp_score[0],etp_score[1],etp_score[2],etp_score[3])])
                    print 'Val Diversity: ' + ' '.join([str(round(it,3)) for it in (dist_score[0],dist_score[1],dist_score[2],dist_score[3])])
                    
                    print 'Val Avg. length: ' + str(round(np.mean([len([y for y in x if y!=0]) for x in res_all]),3)) 
                    print ""
                    summary = sess.run(merged, feed_dict=feed_val)
                    summary2 = tf.Summary(value=[tf.Summary.Value(tag="bleu-2", simple_value=bleu2s),tf.Summary.Value(tag="etp-4", simple_value=etp_score[3])])
                    test_writer.add_summary(summary, uidx)
                    test_writer.add_summary(summary2, uidx)
                if uidx%opt.save_freq == 0:
                    saver.save(sess, opt.save_path)
    
    
    
if __name__ == '__main__':
    main()
