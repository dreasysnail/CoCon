import numpy as np
import tensorflow as tf
from collections import OrderedDict
import nltk
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from tensorflow.python import pywrap_tensorflow
from pdb import set_trace as bp
import data_utils as dp
import data_utils
import sys, os
from tensorflow.python.ops import clip_ops
from tensorflow.python.framework import ops
from collections import defaultdict
import codecs
import cPickle
from tensorflow.python.ops import math_ops, variable_scope
from embedding_metrics import greedy_match, extrema_score, average_score

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)

def sent2idx(text, wordtoix, opt, is_cnn = True):
    
    sent = [wordtoix[x] for x in text.split()]
    
    return prepare_data_for_cnn([sent for i in range(opt.batch_size)], opt)
    


def prepare_data_for_cnn(seqs_x, opt): 
    maxlen=opt.maxlen
    filter_h=opt.filter_shape
    lengths_x = [len(s) for s in seqs_x]
    # print lengths_x
    if maxlen != None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
            else:
                new_seqs_x.append(s_x[l_x-maxlen+1:])
                new_lengths_x.append(maxlen-1)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        
        if len(lengths_x) < 1  :
            return None, None
    
    pad = filter_h -1
    x = []   
    for rev in seqs_x:    
        xx = []
        for i in xrange(pad):
            xx.append(0)
        for idx in rev:
            xx.append(idx)
        while len(xx) < maxlen + 2*pad:
            xx.append(0)
        x.append(xx)
    x = np.array(x,dtype='int32')
    return x   
    
    
def prepare_data_for_rnn(seqs_x, opt, is_add_GO = True):
    
    maxlen=opt.sent_len -2 #+ opt.filter_shape - 1 # 49
    lengths_x = [len(s) for s in seqs_x]
    # print lengths_x
    if maxlen != None:
        new_seqs_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen-2:
                new_seqs_x.append(s_x)
            else:
                #new_seqs_x.append(s_x[l_x-maxlen+1:])
                new_seqs_x.append(s_x[:maxlen-2]+[2])

        seqs_x = new_seqs_x
        lengths_x = [len(s) for s in seqs_x]   
        if len(lengths_x) < 1  :
            return None, None

    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x)
    x = np.zeros(( n_samples, opt.sent_len)).astype('int32')
    for idx, s_x in enumerate(seqs_x):
        if is_add_GO:
            x[idx, 0] = 1 # GO symbol
            x[idx, 1:lengths_x[idx]+1] = s_x
        else:
            x[idx, :lengths_x[idx]] = s_x
    return x   
    



def restore_from_save(t_vars, sess, opt, prefix = 'd_', load_path = None):
    if not load_path:
        load_path = opt.load_path
        
    if opt.load_from_pretrain:

        save_keys = tensors_key_in_file(load_path)
        #print(save_keys.keys()) 
        ss = set([var.name[2:][:-2] for var in t_vars])&set([s[2:] for s in save_keys.keys()])
        cc = {var.name[2:][:-2]:var for var in t_vars}
        ss_right_shape = set([s for s in ss if cc[s].get_shape() == save_keys[prefix+s]])  # only restore variables with correct shape
        ss_wrong_shape = ss - ss_right_shape
        cc2 = {prefix+ var.name[2:][:-2]:var for var in t_vars if var.name[2:][:-2] in ss_right_shape}  # name in file -> var
        loader = tf.train.Saver(var_list=cc2)
        loader.restore(sess, load_path)
        print("Loading variables from '%s'." % load_path)
        print("Loaded variables:"+str(ss_right_shape))
        print("Mis-shaped variables:"+str(ss_wrong_shape))
    else:
        save_keys = tensors_key_in_file(load_path)
        ss = [var for var in t_vars if var.name[:-2] in save_keys.keys()]
        ss_right_shape = [var.name for var in ss if var.get_shape() == save_keys[var.name[:-2]]]
        ss_wrong_shape = set([v.name for v in ss]) - set(ss_right_shape)
        #ss = [var for var in ss if 'OptimizeLoss' not in var]
        loader = tf.train.Saver(var_list= [var for var in t_vars if var.name in ss_right_shape])
        loader.restore(sess, load_path)
        print("Loading variables from '%s'." % load_path)
        print("Loaded variables:"+str(ss_right_shape))
        print("Mis-shaped variables:"+str(ss_wrong_shape))
    
    
_buckets = [(60,60)]    
    
def read_data(source_path, target_path, opt):
    """
    From tensorflow tutorial translate.py
    Read data from source and target files and put into buckets.
    Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

    Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()            
            counter = 0
            while source and target and (not opt.max_train_data_size or counter < opt.max_train_data_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(data_utils.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if opt.minlen <len(source_ids) < min(source_size, opt.maxlen) and opt.minlen <len(target_ids) < min(target_size, opt.maxlen):
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
            
            
            
    return data_set    
    
def read_pair_data_full(src_f, tgt_f, dic_f, train_prop = 0.9, max_num=None, rev_src=False, rev_tgt = False, is_text_src = False, is_text_tgt = False, p_f = '../data/', from_p = True):
    #train, val = [], []
    if from_p:
        p_f = src_f[:-3] + str(max_num) + '.p'
        if os.path.exists(p_f):
            with open(p_f, 'rb') as pfile:
                train, val, test, wordtoix, ixtoword = cPickle.load(pfile)
            return train, val, test, wordtoix, ixtoword


    wordtoix, ixtoword = {}, {}
    print "Start reading dic file . . ."
    if os.path.exists(dic_f):
        print("loading Dictionary")
        counter=0
        with codecs.open(dic_f,"r",'utf-8') as f:
            s=f.readline()
            while s:
                s=s.rstrip('\n').rstrip("\r")
                #print("s==",s)
                wordtoix[s]=counter
                ixtoword[counter]=s
                counter+=1
                s=f.readline()
    def shift_id(x):
        return x
    src, tgt = [], []
    print "Start reading src file . . ."
    with codecs.open(src_f,"r",'utf-8') as f:
        line = f.readline().rstrip("\n").rstrip("\r")
        count, max_l = 0, 0
        #max_length_fact=0
        while line and (not max_num or count<max_num):
            count+=1
            if is_text_src:
                tokens=[wordtoix[x] if x in wordtoix else dp.UNK_ID for x in line.split()]
            else:
                tokens=[shift_id(int(x)) for x in line.split()]
            max_l = max(max_l, len(tokens))
            if not rev_src: # reverse source
                src.append(tokens)
            else :
                src.append(tokens[::-1])
            #pdb.set_trace()
            line = f.readline().rstrip("\n").rstrip("\r")
            if np.mod(count,100000)==0:
                print count
    print "Source cnt: " + str(count) + " maxLen: " + str(max_l)

    print "Start reading tgt file . . ."
    with codecs.open(tgt_f,"r",'utf-8') as f:
        line = f.readline().rstrip("\n").rstrip("\r")
        count = 0
        #max_length_fact=0
        while line and (not max_num or count<max_num):
            count+=1
            if is_text_tgt:
                tokens=[wordtoix[x] if x in wordtoix else dp.UNK_ID for x in line.split()]
            else:
                tokens=[shift_id(int(x)) for x in line.split()]
            if not rev_tgt: # reverse source
                tgt.append(tokens)
            else :
                tgt.append(tokens[::-1])
            line = f.readline().rstrip("\n").rstrip("\r")
            if np.mod(count,100000)==0:
                print count
    print "Target cnt: " + str(count) + " maxLen: " + str(max_l)

    assert(len(src)==len(tgt))
    all_pairs = np.array(zip(*[tgt, src]))
    if not train_prop:
        train , val, test = all_pairs, [], []
    else:
        idx = np.random.choice(len(all_pairs), int(np.floor(train_prop*len(all_pairs))))
        rem_idx = np.array(list(set(range(len(all_pairs)))-set(idx)))
        #v_idx = np.random.choice(rem_idx, int(np.floor(0.5*len(rem_idx))))
        v_idx = np.random.choice(rem_idx, len(rem_idx)-2000)
        t_idx = np.array(list(set(rem_idx)-set(v_idx)))
        #pdb.set_trace()
        train, val, test = all_pairs[idx], all_pairs[v_idx], all_pairs[t_idx]
    if from_p:
        with open(p_f, 'wb') as pfile:
            cPickle.dump([train, val, test, wordtoix, ixtoword], pfile)


        #print(counter)
    #pdb.set_trace()
    return train, val, test, wordtoix, ixtoword
    
def read_test(test_file, wordtoix):
    print "Start reading test file . . ."
    test = []
    with codecs.open(test_file,"r",'utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip("\n").rstrip("\r").split('\t')
            conv = []
            for l in line:
                sent=[wordtoix[x] if x in wordtoix else dp.UNK_ID for x in l.split()] + [2]
                conv.append(sent)
                # bp()
            test.append(conv)
    return test 

def tensors_key_in_file(file_name):
    """Return tensors key in a checkpoint file.
    Args:
    file_name: Name of the checkpoint file.
    """
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        return reader.get_variable_to_shape_map()
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        return None

     
def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    # if (minibatch_start != n):
    #     # Make a minibatch out of what is left
    #     minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)
    
    
# def normalizing_L1(x, axis):
#     norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis, keep_dims=True))
#     normalized = x / (norm)
#     return normalized   
    
def normalizing(x, axis):    
    norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis, keep_dims=True))
    normalized = x / (norm)   
    return normalized

def normalizing_sum(x, axis):
    # sum(x) == 1
    sum_prob = tf.reduce_sum(x, axis=axis, keep_dims=True)
    normalized = x / sum_prob
    return normalized
    
def _p(pp, name):
    return '%s_%s' % (pp, name)

def dropout(X, trng, p=0.):
    if p != 0:
        retain_prob = 1 - p
        X = X / retain_prob * trng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
    return X

""" used for initialization of the parameters. """

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)
    
def uniform_weight(nin,nout=None, scale=0.05):
    if nout == None:
        nout = nin
    W = np.random.uniform(low=-scale, high=scale, size=(nin, nout))
    return W.astype(config.floatX)
    
def normal_weight(nin,nout=None, scale=0.05):
    if nout == None:
        nout = nin
    W = np.random.randn(nin, nout) * scale
    return W.astype(config.floatX)
    
def zero_bias(ndim):
    b = np.zeros((ndim,))
    return b.astype(config.floatX)

"""auxiliary function for KDE"""
def log_mean_exp(A,b,sigma):
    a=-0.5*((A-theano.tensor.tile(b,[A.shape[0],1]))**2).sum(1)/(sigma**2)
    max_=a.max()
    return max_+theano.tensor.log(theano.tensor.exp(a-theano.tensor.tile(max_,a.shape[0])).mean())

'''calculate KDE'''
def cal_nkde(X,mu,sigma):
    s1,updates=theano.scan(lambda i,s: s+log_mean_exp(mu,X[i,:],sigma), sequences=[theano.tensor.arange(X.shape[0])],outputs_info=[np.asarray(0.,dtype="float32")])
    E=s1[-1]
    Z=mu.shape[0]*theano.tensor.log(sigma*np.sqrt(np.pi*2))
    return (Z-E)/mu.shape[0]


def cal_relevance(generated, reference, embedding): # embedding V* E
    generated = [[g] for g in generated]
    reference = [[s] for s in reference]


    #bp()
    relevance_score = [0.0,0.0,0.0]
    relevance_score[0] = greedy_match(reference, generated, embedding)
    relevance_score[1] = average_score(reference, generated, embedding)
    relevance_score[2] = extrema_score(reference, generated, embedding)
    return relevance_score  

    


def cal_BLEU(generated, reference, is_corpus = False):
    #print 'in BLEU score calculation'
    #the maximum is bigram, so assign the weight into 2 half.
    BLEUscore = [0.0,0.0,0.0]
    for idx, g in enumerate(generated):
        if is_corpus:
            score, scores = Bleu(4).compute_score(reference, {0: [g]})
        else:
            score, scores = Bleu(4).compute_score({0: [reference[0][idx]]} , {0: [g]})
        #print g, score
        for i, s in zip([0,1,2],score[1:]):
            BLEUscore[i]+=s
        #BLEUscore += nltk.translate.bleu_score.sentence_bleu(reference, g, weight)
    BLEUscore[0] = BLEUscore[0]/len(generated)
    BLEUscore[1] = BLEUscore[1]/len(generated)
    BLEUscore[2] = BLEUscore[2]/len(generated)
    return BLEUscore
    
def cal_BLEU_4(generated, reference, is_corpus = False):
    #print 'in BLEU score calculation'
    #the maximum is bigram, so assign the weight into 2 half.
    BLEUscore = [0.0,0.0,0.0,0.0]
    for idx, g in enumerate(generated):
        if is_corpus:
            score, scores = Bleu(4).compute_score(reference, {0: [g]})
        else:
            score, scores = Bleu(4).compute_score({0: [reference[0][idx]]} , {0: [g]})
        #print g, score
        for i, s in zip([0,1,2,3],score):
            BLEUscore[i]+=s
        #BLEUscore += nltk.translate.bleu_score.sentence_bleu(reference, g, weight)
    BLEUscore[0] = BLEUscore[0]/len(generated)
    BLEUscore[1] = BLEUscore[1]/len(generated)
    BLEUscore[2] = BLEUscore[2]/len(generated)
    BLEUscore[3] = BLEUscore[3]/len(generated)
    return BLEUscore

def cal_BLEU_4_nltk(generated, reference, is_corpus = False):
    #print 'in BLEU score calculation'
    #the maximum is bigram, so assign the weight into 2 half.
    reference = [[s] for s in reference]
    #bp()
    chencherry = SmoothingFunction()
    # Note: please keep smoothing turned on, because there is a bug in NLTK without smoothing (see below).
    if is_corpus:
        return nltk.translate.bleu_score.corpus_bleu(reference, generated, smoothing_function=chencherry.method2) # smoothing options: 0-7
    else:
        return np.mean([nltk.translate.bleu_score.sentence_bleu(r, g, smoothing_function=chencherry.method2) for r,g in zip(reference, generated)]) # smoothing options: 0-7


def cal_entropy(generated):
    #print 'in BLEU score calculation'
    #the maximum is bigram, so assign the weight into 2 half.
    etp_score = [0.0,0.0,0.0,0.0]
    div_score = [0.0,0.0,0.0,0.0]
    counter = [defaultdict(int),defaultdict(int),defaultdict(int),defaultdict(int)]
    for gg in generated:
        g = gg.rstrip('2').split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) +1e-10
        for v in counter[n].values():
            etp_score[n] += - (v+0.0) /total * (np.log(v+0.0) - np.log(total))
        div_score[n] = (len(counter[n].values())+0.0) /total
    return etp_score, div_score 
 
def prepare_for_bleu(sentence):
    sent=[x for x in sentence if x!=0]
    while len(sent)<4:
        sent.append(0)
    #sent = ' '.join([ixtoword[x] for x in sent])
    sent = ' '.join([str(x) for x in sent])
    return sent
    

def _clip_gradients_seperate_norm(grads_and_vars, clip_gradients):
    """Clips gradients by global norm."""
    gradients, variables = zip(*grads_and_vars)
    clipped_gradients = [clip_ops.clip_by_norm(grad, clip_gradients) for grad in gradients]
    return list(zip(clipped_gradients, variables))


def binary_round(x):
    """
    Rounds a tensor whose values are in [0,1] to a tensor with values in {0, 1},
    using the straight through estimator for the gradient.
    """
    g = tf.get_default_graph()

    with ops.name_scope("BinaryRound") as name:
        with g.gradient_override_map({"Round": "Identity"}):
            return tf.round(x, name=name)



@tf.RegisterGradient("CustomGrad")
def _const_mul_grad(unused_op, grad):
    return  grad/1e4

def one_hot_round(x):
    """
    Rounds a tensor whose values are in [0,1] to a tensor with values in {0, 1},
    using the straight through estimator for the gradient.
    """      
    g = tf.get_default_graph()

    with g.gradient_override_map({"Log": "Identity"}):  
        x = tf.log(x)
    x = 1e4 * x
    with g.gradient_override_map({"Identity": "CustomGrad"}):
        x = tf.identity(x, name="Identity")
    with g.gradient_override_map({"Softmax": "Identity"}):
        x = tf.nn.softmax(x)
    with g.gradient_override_map({"Round": "Identity"}):
        return tf.round(x) # B L V

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def reshaping(x, opt, gen_turn = None):
    if gen_turn==None: gen_turn = opt.num_turn-opt.n_context
    x = np.array(x)
    dim = x.shape
    x = np.reshape(x, [dim[0]/opt.batch_size/(gen_turn), (gen_turn), opt.batch_size, -1])  
    x = np.transpose(x, (0,2,1,3))
    return np.squeeze(x.reshape([dim[0],-1]))
