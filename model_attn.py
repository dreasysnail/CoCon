import tensorflow as tf
import numpy as np
import math, json
from utils import normalizing

def adam(params, grads, lr, schedule, t_total, b1=0.9, b2=0.999, e=1e-8, l2=0, vector_l2=False, max_grad_norm=-1, **kwargs):
    """
    adam with weight decay fix
    """
    t = tf.Variable(0, dtype=tf.float32, trainable=False)
    tt = t+1
    updates = [t.assign(tt)]
    if max_grad_norm > 0:
        grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
    for p, g in zip(params, grads):
        if p is None or g is None:
            print("can't train", p.name, g)
        else:
            if isinstance(g, tf.IndexedSlices):
                g = tf.convert_to_tensor(g)
            m = tf.Variable(p*0, dtype=tf.float32, trainable=False)
            v = tf.Variable(p*0, dtype=tf.float32, trainable=False)
            lrt = lr*tf.sqrt(1-b2**tt)/(1-b1**tt)
            lrt *= schedule(t/t_total)
            mt = b1*m + (1-b1)*g
            vt = b2*v + (1-b2)*g*g
            if (len(p.get_shape()) > 1 or vector_l2) and l2 > 0:
                pt = p - lrt * (mt / (tf.sqrt(vt) + e) + l2*p)
            else:
                pt = p - lrt * (mt / (tf.sqrt(vt) + e))
            updates.extend([m.assign(mt), v.assign(vt), p.assign(pt)])
    return tf.group(*updates)


def shape_list(x):
    """
    deal with dynamic shape in tensorflow cleanly
    """
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]


def gelu(x):
    return 0.5*x*(1+tf.tanh(math.sqrt(2/math.pi)*(x+0.044715*tf.pow(x, 3))))

def swish(x):
    return x*tf.nn.sigmoid(x)

opt_fns = {
    'adam':adam,
}

act_fns = {
    'relu':tf.nn.relu,
    'swish':swish,
    'gelu':gelu
}


def _norm(x, g=None, b=None, e=1e-5, axis=[1]):
    u = tf.reduce_mean(x, axis=axis, keep_dims=True)
    s = tf.reduce_mean(tf.square(x-u), axis=axis, keep_dims=True)
    x = (x - u) * tf.rsqrt(s + e)
    if g is not None and b is not None:
        x = x*g + b
    return x

def norm(x, scope, axis=[-1], trainable=False):
    with tf.variable_scope(scope):
        n_state = shape_list(x)[-1]
        g = tf.get_variable("g", [n_state], initializer=tf.constant_initializer(1), trainable=trainable)
        b = tf.get_variable("b", [n_state], initializer=tf.constant_initializer(0), trainable=trainable)
        return _norm(x, g, b, axis=axis)

def dropout(x, pdrop, train):
    if train and pdrop > 0:
        x = tf.nn.dropout(x, 1-pdrop)
    return x

def mask_attn_weights(w):
    n = shape_list(w)[-1]
    b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)
    b = tf.reshape(b, [1, 1, n, n])
    w = w*b + -1e9*(1-b)
    return w

def _attn(q, k, v, attn_pdrop, train=False, scale=False):
    w = tf.matmul(q, k)

    if scale:
        n_state = shape_list(v)[-1]
        w = w*tf.rsqrt(tf.cast(n_state, tf.float32))

    w = mask_attn_weights(w)
    w = tf.nn.softmax(w)

    w = dropout(w, attn_pdrop, train)

    a = tf.matmul(w, v)
    return a

def split_states(x, n):
    x_shape = shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1]+[n, m//n]
    return tf.reshape(x, new_x_shape)

def merge_states(x):
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2]+[np.prod(x_shape[-2:])]
    return tf.reshape(x, new_x_shape)

def split_heads(x, n, k=False):
    if k:
        return tf.transpose(split_states(x, n), [0, 2, 3, 1])
    else:
        return tf.transpose(split_states(x, n), [0, 2, 1, 3])

def merge_heads(x):
    return merge_states(tf.transpose(x, [0, 2, 1, 3]))

def conv1d(x, scope, nf, rf, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), pad='VALID', train=False, trainable=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [rf, nx, nf], initializer=w_init, trainable=trainable)
        b = tf.get_variable("b", [nf], initializer=b_init, trainable=trainable)
        if rf == 1: #faster 1x1 conv
            c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, shape_list(x)[:-1]+[nf])
        else: #was used to train LM
            c = tf.nn.conv1d(x, w, stride=1, padding=pad)+b
        return c

def attn(x, scope, n_state, opt, train=False, scale=False, trainable=False):
    n_head = opt.n_head
    #print('n_state = %i, n_head = %i'%(n_state, n_head))        # error: n_state = 1, n_head = 5

    assert n_state%n_head==0
    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3, 1, train=train, trainable=trainable)
        q, k, v = tf.split(c, 3, 2)
        q = split_heads(q, n_head)
        k = split_heads(k, n_head, k=True)
        v = split_heads(v, n_head)
        a = _attn(q, k, v, opt.attn_pdrop, train=train, scale=scale)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state, 1, train=train, trainable=trainable)
        a = dropout(a, opt.resid_pdrop, train)
        return a

def mlp(x, scope, n_state, opt, train=False, trainable=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        act = act_fns[opt.afn]
        h = act(conv1d(x, 'c_fc', n_state, 1, train=train, trainable=trainable))
        h2 = conv1d(h, 'c_proj', nx, 1, train=train, trainable=trainable)
        h2 = dropout(h2, opt.resid_pdrop, train)
        return h2

def block(x, scope, opt, train=False, scale=False, trainable=False):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        a = attn(x, 'attn', nx, opt, train=train, scale=scale, trainable=trainable)
        n = norm(x+a, 'ln_1', trainable=trainable)
        m = mlp(n, 'mlp', nx*4, opt, train=train, trainable=trainable)
        h = norm(n+m, 'ln_2', trainable=trainable)
        return h



#def conv_model_3layer(X, opt, prefix = '', is_reuse= None, num_outputs = None, is_train = True, multiplier = 2):
def self_attn(x_seq, opt, prefix='', is_reuse=None, is_train=True):
    # x_seq = [opt.batch_size, opt.sent_len]
    # note the order of these lines (e.g. pe first, then we) matters

    trainable = opt.attn_trainable

    with tf.variable_scope('attn_model', reuse=is_reuse):
        pe = tf.get_variable("pe", [opt.maxlen, opt.embed_size], initializer=tf.random_normal_initializer(stddev=0.02))#, trainable=trainable)
        pe = dropout(pe, opt.embd_pdrop, is_train)

        we = tf.get_variable("we", [opt.n_words, opt.embed_size], initializer=tf.random_normal_initializer(stddev=0.02))#, trainable=trainable)
        we = dropout(we, opt.embd_pdrop, is_train)
        
        h = tf.gather(we, x_seq) + tf.expand_dims(pe, 0)

        print('h init',h.shape)
        for layer in range(opt.n_block):
            h = block(h, 'h%d'%layer, opt, train=is_train, scale=True, trainable=trainable)

        print('h after blocks',h.shape)
        
        end_h = tf.reshape(h, [-1, opt.embed_size])
        print('end_h init',end_h.shape)
        pool_idx = tf.cast(
            tf.argmax(
                tf.cast(
                    tf.equal(x_seq, opt.end_id), 
                    tf.float32
                    ), 
                1), 
            tf.int32)
        
        end_h = tf.gather(end_h, tf.range(shape_list(x_seq)[0], dtype=tf.int32)*opt.sent_len + pool_idx)
        print('end_h final',end_h.shape)

        end_logits = tf.layers.dense(end_h, opt.n_hid)#, activation=tf.tanh)

        print('end_logits',end_logits.shape)
        return end_logits




def find_trainable_variables(key):
    #return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ".*{}.*".format(key))
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, ".*{}.*".format(key))
    




def load_openAI_params(opt):

    FLD_OPENAI = opt.data_dir + '/attn_model'
    shapes = json.load(open(FLD_OPENAI + '/params_shapes.json'))
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [np.load(FLD_OPENAI + '/params_{}.npy'.format(n)) for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]

    # 0-th: positional embed
    # 1-th: word embed
    # 2+(j-1)*12 ~ 2+j*12: block-j params

    init_params[0] = init_params[0][:opt.maxlen, :]
    init_params[1] = get_word_embed(init_params[1])
    return init_params


def get_word_embed(raw):
    n_embd = raw.shape[1]
    embed = []
    for line in open('final_vocab.tsv'):    # word \t our_ix \t openAI_ix
        k, _, ix = line.strip('\n').strip('\r').split('\t')
        if ix == '?':
            e = (np.random.randn(1, n_embd)*0.02).astype(np.float32)
        else:
            ix = int(ix)
            e = raw[ix: ix+1, :]
        embed.append(e)
    return np.concatenate(embed, axis=0)




if __name__ == '__main__':
    class OPT:
        layer = 3
        n_head = 5
        attn_pdrop = 0.1
        resid_pdrop = 0.1
        afn ='gelu'

    opt = OPT()
    n_state = opt.n_head * 2
    x = tf.get_variable("my_variable", [1, 2, n_state])
    h = self_attn(x, opt)