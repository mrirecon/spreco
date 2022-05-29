from spreco.model.utils import *

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tf_slim import add_arg_scope
import numpy as np


@add_arg_scope
def dense(x_, num_units, nonlinearity=None, init_scale=1., counters={}, init=False, use_bias=True, ema=None, **kwargs):
    ''' fully connected layer '''
    if 'scope' in kwargs.keys():
        name = get_name(kwargs['scope'], counters)
    else:
        name = get_name('dense', counters)

    if 'debug' in kwargs.keys():
        if kwargs['debug']:
            print(name)
    stop_grad = False 
    if 'stop_grad' in kwargs.keys():
        stop_grad = kwargs['stop_grad']

    with tf.variable_scope(name):
        V = get_variable('V', stop_grad, shape=[int(x_.get_shape()[1]),num_units], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        g = get_variable('g', stop_grad, shape=[num_units], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
        if use_bias:
            b = get_variable('b', stop_grad, shape=[num_units], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.), trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        # https://arxiv.org/pdf/1602.07868.pdf
        x = tf.matmul(x_, V)

        scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))

        if use_bias:
            x = tf.reshape(scaler, [1, num_units]) * x + tf.reshape(b, [1, num_units])

        if init: # normalize x
            m_init, v_init = tf.nn.moments(x, [0])
            scale_init = init_scale/tf.sqrt(v_init + 1e-10)
            if use_bias:
                with tf.control_dependencies([g.assign(g*scale_init), b.assign_add(-m_init*scale_init)]):
                    x = tf.matmul(x_, V)
                    scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
                    x = tf.reshape(scaler, [1, num_units]) * x + tf.reshape(b, [1, num_units])
            else:
                with tf.control_dependencies([g.assign(g*scale_init)]):
                    x = tf.matmul(x_, V)
                    scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
                    x = tf.reshape(scaler, [1, num_units]) * x

        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)

        return x

@add_arg_scope
def dense_plus(x_, h, num_units, nr_classes, nonlinearity=None, init_scale=1., counters={}, init=False, use_bias=True, ema=None, **kwargs):
    ''' fully connected layer '''
    if 'scope' in kwargs.keys():
        name = get_name(kwargs['scope'], counters)
    else:
        name = get_name('dense', counters)

    if 'debug' in kwargs.keys():
        if kwargs['debug']:
            print(name)
    stop_grad = False 
    if 'stop_grad' in kwargs.keys():
        stop_grad = kwargs['stop_grad']

    with tf.variable_scope(name):
        V = get_variable('V', stop_grad, shape=[nr_classes, int(x_.get_shape()[1]),num_units], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        g = get_variable('g', stop_grad, shape=[nr_classes, num_units], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
        if use_bias:
            b = get_variable('b', stop_grad, shape=[nr_classes, num_units], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.), trainable=True)

        V_h = tf.gather(V, h, axis=0)
        g_h = tf.gather(g, h, axis=0)
        if use_bias:
            b_h =tf.gather(b, h, axis=0)

        # use weight normalization (Salimans & Kingma, 2016)
        x = tf.einsum('ij,ijk->ik',x_, V_h)

        scaler = g_h / tf.sqrt(tf.reduce_sum(tf.square(V_h), [1]))

        if use_bias:
            x = scaler * x + b_h

        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)

        return x


@add_arg_scope
def conv2d_plus(x_, num_filters, filter_size=[3,3], stride=[1,1], pad='SAME', nonlinearity=None, init_scale=1., counters={}, init=False, ema=None, bias=True, weight_norm=False, **kwargs):
    ''' convolutional layer '''
    if 'scope' in kwargs.keys():
        name = get_name(kwargs['scope'], counters) # this is the scope defined by args
    else:
        name = get_name('conv2d', counters) # this is default scope named with conv2d
    
    if 'debug' in kwargs.keys():
        if kwargs['debug']:
            print(name)
    stop_grad = False 
    if 'stop_grad' in kwargs.keys():
        stop_grad = kwargs['stop_grad']

    dilation = 1
    if 'dilation' in kwargs.keys():
        dilation = kwargs['dilation']

    with tf.variable_scope(name):
        V = get_variable('V', stop_grad, shape=filter_size+[int(x_.get_shape()[-1]),num_filters], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.05), trainable=True)

        if weight_norm:
            g = get_variable('g', stop_grad, shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
            # use weight normalization (Salimans & Kingma, 2016)
            W = tf.reshape(g, [1, 1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1, 2])
        else:
            W = V
        
        # calculate convolutional layer output
        x = tf.nn.conv2d(x_, W, [1] + stride + [1], pad, dilations=dilation)

        if bias:
            b = get_variable('b', stop_grad, shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(0.), trainable=True)
            x = tf.nn.bias_add(x, b)

        if init and weight_norm:  # normalize x
            m_init, v_init = tf.nn.moments(x, [0,1,2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
                # x = tf.identity(x)
                W = tf.reshape(g, [1, 1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1, 2])
                x = tf.nn.conv2d(x_, W, [1] + stride + [1], pad, dilations=dilation)
                if bias:
                    x = tf.nn.bias_add(x, b)

        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)

        return x

@add_arg_scope
def conv2d(x_, num_filters, filter_size=[3,3], stride=[1,1], pad='SAME', nonlinearity=None, init_scale=1., counters={}, init=False, ema=None, bias=True, **kwargs):
    ''' convolutional layer '''
    if 'scope' in kwargs.keys():
        name = get_name(kwargs['scope'], counters) # this is the scope defined by args
    else:
        name = get_name('conv2d', counters) # this is default scope named with conv2d
    
    if 'debug' in kwargs.keys():
        if kwargs['debug']:
            print(name)
    stop_grad = False 
    if 'stop_grad' in kwargs.keys():
        stop_grad = kwargs['stop_grad']

    dilation = 1
    if 'dilation' in kwargs.keys():
        dilation = kwargs['dilation']

    with tf.variable_scope(name):
        V = get_variable('V', stop_grad, shape=filter_size+[int(x_.get_shape()[-1]),num_filters], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        g = get_variable('g', stop_grad, shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
        b = get_variable('b', stop_grad, shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(0.), trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        W = tf.reshape(g, [1, 1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1, 2])
        
        # calculate convolutional layer output
        x = tf.nn.conv2d(x_, W, [1] + stride + [1], pad, dilations=dilation)
        if bias:
            x = tf.nn.bias_add(x, b)

        if init:  # normalize x
            m_init, v_init = tf.nn.moments(x, [0,1,2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
                # x = tf.identity(x)
                W = tf.reshape(g, [1, 1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1, 2])
                x = tf.nn.conv2d(x_, W, [1] + stride + [1], pad, dilations=dilation)
                if bias:
                    x = tf.nn.bias_add(x, b)

        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)

        return x

@add_arg_scope
def deconv2d(x_, num_filters, filter_size=[3,3], stride=[1,1], pad='SAME', nonlinearity=None, init_scale=1., counters={}, init=False, ema=None, bias = True, **kwargs):
    ''' transposed convolutional layer '''
    if 'scope' in kwargs.keys():
        name = get_name(kwargs['scope'], counters)
    else:
        name = get_name('deconv2d', counters)

    if 'debug' in kwargs.keys():
        if kwargs['debug']:
            print(name)

    stop_grad = False 
    if 'stop_grad' in kwargs.keys():
        stop_grad = kwargs['stop_grad']

    xs = int_shape(x_)
    if pad=='SAME':
        target_shape = [xs[0], xs[1]*stride[0], xs[2]*stride[1], num_filters]
    else:
        target_shape = [xs[0], xs[1]*stride[0] + filter_size[0]-1, xs[2]*stride[1] + filter_size[1]-1, num_filters]
    with tf.variable_scope(name):
        V = get_variable('V', stop_grad, shape=filter_size+[num_filters,int(x_.get_shape()[-1])], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        g = get_variable('g', stop_grad, shape=[num_filters], dtype=tf.float32,
                                initializer=tf.constant_initializer(1.), trainable=True)
        b = get_variable('b', stop_grad, shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(0.), trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        W = tf.reshape(g, [1, 1, num_filters, 1]) * tf.nn.l2_normalize(V, [0, 1, 3])

        # calculate convolutional layer output
        x = tf.nn.conv2d_transpose(x_, W, target_shape, [1] + stride + [1], padding=pad)
        if bias:
            x = tf.nn.bias_add(x, b)

        if init:  # normalize x
            m_init, v_init = tf.nn.moments(x, [0,1,2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
                # x = tf.identity(x)
                W = tf.reshape(g, [1, 1, num_filters, 1]) * tf.nn.l2_normalize(V, [0, 1, 3])
                x = tf.nn.conv2d_transpose(x_, W, target_shape, [1] + stride + [1], padding=pad)
                if bias:
                    x = tf.nn.bias_add(x, b)

        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)

        return x


@add_arg_scope
def nin(x, num_units, use_bias=True, nonlinearity=None, **kwargs):
    """ a network in network layer (1x1 CONV) """
    s = int_shape(x)
    if None in s:
        x = tf.reshape(x, [-1,s[-1]])  # 03.09 11:18
    else:
        x = tf.reshape(x, [np.prod(s[:-1]),s[-1]])

    x = dense(x, num_units, use_bias=use_bias, **kwargs)

    if nonlinearity is not None:
        x = nonlinearity(x)

    if None in s:
        if len(s) == 2:
            shape = [-1] + [num_units]
        else: 
            shape = [-1] +  list(map(int, s[1:-1])) + [num_units]
        out = tf.reshape(x, shape)
    else:
        out = tf.reshape(x, s[:-1]+[num_units])
    return out

@add_arg_scope
def emb(t, embedding_size=256, scale=1.0, counters={}, **kwargs):
    """
    gaussian fourier embedding for t
    # https://www.cs.cmu.edu/~schneide/DougalRandomFeatures_UAI2015.pdf
    # https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf
    """

    if 'scope' in kwargs.keys():
        name = get_name(kwargs['scope'], counters)
    else:
        name = get_name('emb', counters)

    if 'debug' in kwargs.keys():
        if kwargs['debug']:
            print(name)

    with tf.variable_scope(name):

        W = get_variable('W', stop_grad=True, shape=[embedding_size], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=scale), trainable=True)
        t_proj = t[:, None]*W[None, :] * 2 * 3.1415926
        
        return tf.concat([tf.math.sin(t_proj), tf.math.cos(t_proj)], axis=-1)

@add_arg_scope
def embed_t(t, embedding_size=256, scale=1.0, counters={}, **kwargs):
    """
    gaussian fourier embedding for t
    # https://www.cs.cmu.edu/~schneide/DougalRandomFeatures_UAI2015.pdf
    # https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf
    """
    name = get_name('embed_t', counters)
    W = get_variable(name, stop_grad=True, shape=[embedding_size], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=scale), trainable=True)
    t_proj = t[:, None]*W[None, :] * 2 * 3.1415926
    return tf.concat([tf.math.sin(t_proj), tf.math.cos(t_proj)], axis=-1)

@add_arg_scope
def self_attention(x, qk_chns, v_chns, **kwargs):
    """
    Non local neural networks
    https://arxiv.org/pdf/1711.07971.pdf
    """
    shape = int_shape(x)
    if None in shape:
        flatten_shape = (-1, shape[1]*shape[2], shape[3])
        out_shape = (-1, shape[1], shape[2], v_chns)
    else:
        flatten_shape = (shape[0], shape[1]*shape[2], shape[3])
        out_shape = (shape[0], shape[1], shape[2], v_chns)

    query_conv = tf.reshape(nin(x, qk_chns, nonlinearity=None, scope='global_attention'), flatten_shape)
    key_conv   = tf.reshape(nin(x, qk_chns, nonlinearity=None, scope='global_attention'), flatten_shape)
    value_conv = tf.reshape(nin(x, v_chns, nonlinearity=None, scope='global_attention'), flatten_shape)

    correlation = tf.einsum("bnf,bjf->bnj", query_conv, key_conv)
    attention_map = tf.nn.softmax(correlation, axis=-1)
    out = tf.einsum("bnf,bnj->bjf", value_conv, attention_map)
    out = nin(out, v_chns, scope='global_attention')

    if shape[-1] != v_chns:
        x = nin(x, v_chns, nonlinearity=None, scope='global_attention')
        
    return tf.reshape(out, out_shape) + x


## normalization modules
@add_arg_scope
def batch_normalization(x, is_training=True, counters={}, **kwargs):
    
    if 'scope' in kwargs.keys():
        name = get_name(kwargs['scope'], counters)
    else:
        name = get_name('batch_norm', counters)

    if 'debug' in kwargs.keys():
        if kwargs['debug']:
            print(name)

    with tf.variable_scope(name):
        return tf.layers.batch_normalization(x, is_training)

@add_arg_scope
def cond_instance_norm_plus(x, h, nr_classes, counters={}, **kwargs):
    """
    Adjusted conditional instance normalization arXiv:1907.05600

    y is the index of classes
    """
    if 'scope' in kwargs.keys():
        name = get_name(kwargs['scope'], counters)
    else:
        name = get_name('cond_instance_norm_plus', counters)

    stop_grad = False
    if 'stop_grad' in kwargs.keys():
        stop_grad = kwargs['stop_grad']

    in_shape = int_shape(x)

    with tf.variable_scope(name):
        gamma = get_variable(name+'_gamma', stop_grad, shape=[nr_classes, in_shape[-1]], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
        beta = get_variable(name+'_beta', stop_grad, shape=[nr_classes, in_shape[-1]], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.), trainable=True)
        alpha = get_variable(name+'_alpha', stop_grad, shape=[nr_classes, in_shape[-1]], dtype=tf.float32,
                                initializer=tf.constant_initializer(1.), trainable=True)
        mean, variance = tf.nn.moments(x, [1, 2], keepdims=True)
        nx = tf.nn.batch_normalization(x, mean, variance, offset=None, scale=None, variance_epsilon=1e-12, name='x_norm')
        cm, cvar = tf.nn.moments(mean, [-1], keepdims=True)
        adjusted_mean = tf.nn.batch_normalization(mean, cm, cvar, offset=None, scale=None, variance_epsilon=1e-12, name='adjust_norm')
        offset = tf.expand_dims(tf.expand_dims(tf.gather(beta, h, axis=0),1),1)
        scale = tf.expand_dims(tf.expand_dims(tf.gather(gamma, h, axis=0),1),1)
        out = nx + adjusted_mean*tf.expand_dims(tf.expand_dims(tf.gather(alpha, h, axis=0),1),1)
        #out = out*scale + offset
        out = tf.nn.batch_normalization(out, tf.zeros_like(mean), tf.ones_like(variance), offset=offset, scale=scale, variance_epsilon=1e-12, name='instancenorm') 
        
        return out

@add_arg_scope
def group_norm(x, groups=32, counters={}, **kwargs):

    if 'scope' in kwargs.keys():
        name = get_name(kwargs['scope'], counters)
    else:
        name = get_name('group_norm', counters)
    
    in_shape = int_shape(x)

    stop_grad = False
    if 'stop_grad' in kwargs.keys():
        stop_grad = kwargs['stop_grad']

    channels = in_shape[-1]

    if groups > channels:
        raise ValueError('Invalid groups %d for %d channels' % (groups, channels))

    new_shape = in_shape[:-1] + [groups, channels//groups]
    x = tf.reshape(x, new_shape)

    params_shape = [1, 1, 1, groups, channels//groups]

    with tf.variable_scope(name):

        gamma = get_variable(name+'_gamma', stop_grad, shape=params_shape, dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
        beta = get_variable(name+'_beta', stop_grad, shape=params_shape, dtype=tf.float32,
                                initializer=tf.constant_initializer(0.), trainable=True)
        mean, variance = tf.nn.moments(x, [1,2,4], keepdims=True)

        x = tf.nn.batch_normalization(x, mean, variance, offset = beta, scale=gamma, variance_epsilon=1e-12, name='group_norm')
    
    out = tf.reshape(x, in_shape)

    return out

@add_arg_scope
def instance_norm_plus(x, counters={}, **kwargs):
    """
    Adjusted conditional instance normalization arXiv:1907.05600

    y is the index of classes
    """
    if 'scope' in kwargs.keys():
        name = get_name(kwargs['scope'], counters)
    else:
        name = get_name('instance_norm_plus', counters)

    stop_grad = False
    if 'stop_grad' in kwargs.keys():
        stop_grad = kwargs['stop_grad']

    in_shape = int_shape(x)

    with tf.variable_scope(name):
        gamma = get_variable(name+'_gamma', stop_grad, shape=[in_shape[-1]], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
        beta = get_variable(name+'_beta', stop_grad, shape=[in_shape[-1]], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.), trainable=True)
        alpha = get_variable(name+'_alpha', stop_grad, shape=[in_shape[-1]], dtype=tf.float32,
                                initializer=tf.constant_initializer(1.), trainable=True)
        mean, variance = tf.nn.moments(x, [1, 2], keepdims=True)
        nx = tf.nn.batch_normalization(x, mean, variance, offset=None, scale=None, variance_epsilon=1e-12, name='x_norm')
        cm, cvar = tf.nn.moments(mean, [-1], keepdims=True)
        adjusted_mean = tf.nn.batch_normalization(mean, cm, cvar, offset=None, scale=None, variance_epsilon=1e-12, name='adjust_norm')
        offset = tf.expand_dims(tf.expand_dims(tf.expand_dims(beta, 0),0),0)
        scale = tf.expand_dims(tf.expand_dims(tf.expand_dims(gamma, 0),0),0)
        out = nx + adjusted_mean*tf.expand_dims(tf.expand_dims(tf.expand_dims(alpha, 0), 0), 0)
        #out = out*scale + offset
        out = tf.nn.batch_normalization(out, tf.zeros_like(mean), tf.ones_like(variance), offset=offset, scale=scale, variance_epsilon=1e-12, name='instancenorm') 
        
        return out

@add_arg_scope
def instance_norm(x, counters={}, **kwargs):
    """
    Adjusted conditional instance normalization arXiv:1907.05600

    y is the index of classes
    """
    if 'scope' in kwargs.keys():
        name = get_name(kwargs['scope'], counters)
    else:
        name = get_name('instance_norm_plus', counters)

    stop_grad = False
    if 'stop_grad' in kwargs.keys():
        stop_grad = kwargs['stop_grad']

    in_shape = int_shape(x)

    with tf.variable_scope(name):
        gamma = get_variable(name+'_gamma', stop_grad, shape=[in_shape[-1]], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
        beta = get_variable(name+'_beta', stop_grad, shape=[in_shape[-1]], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.), trainable=True)
        mean, variance = tf.nn.moments(x, [1, 2], keepdims=True)
        offset = tf.expand_dims(tf.expand_dims(tf.expand_dims(beta, 0),0),0)
        scale = tf.expand_dims(tf.expand_dims(tf.expand_dims(gamma, 0),0),0)
        out = tf.nn.batch_normalization(x, mean, variance, offset=offset, scale=scale, variance_epsilon=1e-12, name='instance_norm') 
        
        return out

@add_arg_scope
def layer_norm(x, counters={}, **kwargs):

    if 'scope' in kwargs.keys():
        name = get_name(kwargs['scope'], counters)
    else:
        name = get_name('layer_norm', counters)

    stop_grad = False
    if 'stop_grad' in kwargs.keys():
        stop_grad = kwargs['stop_grad']

    in_shape = int_shape(x)
    with tf.variable_scope(name):
        gamma = get_variable(name+'_gamma', stop_grad, shape=[in_shape[-1]], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
        beta = get_variable(name+'_beta', stop_grad, shape=[in_shape[-1]], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.), trainable=True)
        mean, variance = tf.nn.moments(x, [1,2,3], keepdims=True)
        out = tf.nn.batch_normalization(x, mean, variance, offset=beta, scale=gamma, variance_epsilon=1e-12, name='layer_norm')
        return out


##### PIXELCNN++ #####
@add_arg_scope
def gated_resnet(x, a=None, h=None, nonlinearity=concat_elu, conv=conv2d, init=False, counters={}, ema=None, dropout_p=0., **kwargs):
    xs = int_shape(x)
    num_filters = xs[-1]
    
    c1 = conv(nonlinearity(x), num_filters)
    if a is not None: # add short-cut connection if auxiliary input 'a' is given
        c1 += nin(nonlinearity(a), num_filters)
    c1 = nonlinearity(c1)
    if dropout_p > 0:
        c1 = tf.nn.dropout(c1, keep_prob=1. - dropout_p)
    
    c2 = conv(c1, num_filters * 2, init_scale=0.1)

    # add projection of h vector if included: conditional generation
    stop_grad = False 
    if 'stop_grad' in kwargs.keys():
        stop_grad = kwargs['stop_grad']

    if h is not None:
        with tf.variable_scope(get_name('conditional_weights', counters)):
            hw = get_variable('hw', stop_grad, ema, shape=[int_shape(h)[-1], 2 * num_filters], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        if init:
            hw = hw.initialized_value()
        c2 += tf.reshape(tf.matmul(h, hw), [xs[0], 1, 1, 2 * num_filters])
    

    a, b = tf.split(c2, 2, 3)
    c3 = a * tf.nn.sigmoid(b)
    
    return x + c3


@add_arg_scope
def down_shifted_conv2d(x, num_filters, filter_size=[2,3], stride=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[filter_size[0]-1,0], [int((filter_size[1]-1)/2),int((filter_size[1]-1)/2)],[0,0]])
    return conv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)

@add_arg_scope
def down_shifted_deconv2d(x, num_filters, filter_size=[2,3], stride=[1,1], **kwargs):
    x = deconv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)
    xs = int_shape(x)
    return x[:,:(xs[1]-filter_size[0]+1),int((filter_size[1]-1)/2):(xs[2]-int((filter_size[1]-1)/2)),:]

@add_arg_scope
def down_right_shifted_conv2d(x, num_filters, filter_size=[2,2], stride=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[filter_size[0]-1, 0], [filter_size[1]-1, 0],[0,0]])
    return conv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)

@add_arg_scope
def down_right_shifted_deconv2d(x, num_filters, filter_size=[2,2], stride=[1,1], **kwargs):
    x = deconv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)
    xs = int_shape(x)
    return x[:,:(xs[1]-filter_size[0]+1):,:(xs[2]-filter_size[1]+1),:]
