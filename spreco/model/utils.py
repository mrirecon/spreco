import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

def int_shape(x):
    shape = x.get_shape()
    if None in shape:
        return shape
    else:
        return list(map(int, shape))

def get_variable(name, stop_grad=False, ema=None, **kwargs):
    """ utility to get variable"""

    if stop_grad:
        v = tf.get_variable(name, **kwargs)
        v = tf.stop_gradient(v)
    
    else:
        v = tf.get_variable(name, **kwargs)
    
    if ema is not None:
        v = ema.average(v)
        
    return v

def get_name(layer_name, counters):
    """ utility to keep track of layer names"""

    if not layer_name in counters:
        counters[layer_name] = 0

    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name

def down_shift(x):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0],1,xs[2],xs[3]]), x[:,:xs[1]-1,:,:]],1)

def right_shift(x):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0],xs[1],1,xs[3]]), x[:,:,:xs[2]-1,:]],2)

def concat_elu(x):
    """
    like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU
    """
    axis = len(x.get_shape())-1
    return tf.nn.elu(tf.concat([x, -x], axis))

def get_nonlinearity(strs):

    if strs == 'relu':
        return tf.nn.relu
    elif strs == 'elu':
        return tf.nn.elu
    elif strs == 'swish':
        return tf.nn.swish
    elif strs == 'softplus':
        return tf.nn.softplus
    else:
        raise NotImplementedError('nonlinearity function does not exist!')
