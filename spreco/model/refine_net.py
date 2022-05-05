from spreco.model import nn, utils

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tf_slim import add_arg_scope
from tf_slim import arg_scope

@add_arg_scope
def cond_crp_block(x, h, nr_filters, nr_stages, nonlinearity, normalizer, **kwargs):
    """
    chained residual pool block
    """
    x = nonlinearity(x)
    path = x
    if nr_filters is None:
        nr_filters = nn.int_shape(x)[-1]
    for _ in range(nr_stages):
        path = normalizer(path, h)
        path = tf.nn.avg_pool2d(path, ksize=[5,5], strides=1, padding="SAME") #avg_pool2d
        path = nn.conv2d_plus(path, nr_filters, nonlinearity=None, bias=False, scope='cond_crp')  # don't need bias 
        x = path + x
    return x

@add_arg_scope
def cond_rcu_block(x, h, nr_filters, nr_resnet, nr_stages, nonlinearity, normalizer, **kwargs):
    """
    residual convolution unit
    """
    if nr_filters is None:
        nr_filters = nn.int_shape(x)[-1]
    for _ in range(nr_resnet):
        residual = x
        for _ in range(nr_stages):
            x = normalizer(x, h)
            x = nonlinearity(x)
            x = nn.conv2d_plus(x, nr_filters, nonlinearity=None, bias=False, scope='cond_rcu')  # don't need bias
        x += residual
    return x

@add_arg_scope
def cond_msf_block(blocks, h, nr_filters, out_shape, normalizer, **kwargs):
    """
    multi-resolution fusion
    blocks -> a list or tuple of blocks passed to msf 
    out_shape -> [batch_size, dim_0, dim_1, 2]
    """
    sums = []
    
    for i in range(len(blocks)):
        xl_out = normalizer(blocks[i], h)
        if nr_filters is None:
            nr_filters = nn.int_shape(blocks[i])[-1]
        xl_out = nn.conv2d_plus(xl_out, nr_filters, nonlinearity=None, scope='cond_msf')
        xl_out = tf.image.resize(xl_out, out_shape, method='bilinear')
        sums.append(xl_out)
    return tf.reduce_sum(sums, axis=0)


@add_arg_scope
def cond_refine_block(blocks, h, nr_filters, out_shape, nonlinearity, normalizer, end=False, **kwargs):
    """
    refine block
    """
    outs = []

    for i in range(len(blocks)):
        outs.append(cond_rcu_block(blocks[i], h, nr_filters=None, nr_resnet=2, nr_stages=2, nonlinearity=nonlinearity, normalizer=normalizer))
    
    if len(blocks) > 1:
        y = cond_msf_block(outs, h, nr_filters=nr_filters, out_shape=out_shape, normalizer=normalizer)
    else:
        y = outs[0]
    
    y = cond_crp_block(y, h, nr_filters=None, nr_stages=2, nonlinearity=nonlinearity, normalizer=normalizer)
    y = cond_rcu_block(y, h, nr_filters, nr_resnet=3 if end else 1, nr_stages=2, nonlinearity=nonlinearity, normalizer=normalizer)
    
    return y

@add_arg_scope
def cond_res_block(x, h, out_filters, nonlinearity, normalizer, rescale=False, **kwargs):
    """
    resnet block
    out_filters is output_dims/feature
    """
    in_filters = nn.int_shape(x)[-1]
    x_skip = x
    x = normalizer(x, h) 
    x = nonlinearity(x)
    if rescale:
        x = nn.conv2d_plus(x, in_filters, nonlinearity=None, scope='cond_res', **kwargs)
    else:
        x = nn.conv2d_plus(x, out_filters, nonlinearity=None, scope='cond_res', **kwargs)

    x = normalizer(x, h)
    x = nonlinearity(x)

    x = nn.conv2d_plus(x, out_filters, nonlinearity=None, scope='cond_res', **kwargs)
    if 'dilation' not in kwargs.keys() and rescale:
        x = tf.nn.avg_pool2d(x, ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')
    
    if out_filters == in_filters and not rescale:
        shortcut = x_skip
    else:
        if 'dilation' not in kwargs.keys():
            shortcut = nn.conv2d_plus(x_skip, out_filters, filter_size=[1,1])
            shortcut = tf.nn.avg_pool2d(shortcut, ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')
        else:
            shortcut = nn.conv2d_plus(x_skip, out_filters)

    return shortcut + x

class cond_refine_net():

    def __init__(self, config, chns=2, normalizer=nn.cond_instance_norm_plus):
        self.chns          = chns
        self.nr_filters    = config['nr_filters']
        self.nr_classes    = config['nr_levels']
        self.nonlinearity  = utils.get_nonlinearity(config['nonlinearity'])
        self.normalizer    = normalizer
        self.counters      = {}
        self.affine_x      = config['affine_x']
        self.forward       = tf.make_template('forward', self.body)

    def body(self, x, h):
        """
        multi level refine net conditional on y
        """
        if self.affine_x:
            x = 2*x - 1

        with arg_scope([nn.conv2d_plus, cond_refine_block, cond_crp_block, cond_rcu_block, cond_msf_block, cond_res_block, nn.cond_instance_norm_plus],
                                 nonlinearity=self.nonlinearity, counters=self.counters, normalizer=self.normalizer, nr_classes=self.nr_classes):

            x_level_0 = nn.conv2d_plus(x, num_filters=1*self.nr_filters, nonlinearity=None)
            x_level_1_0 = cond_res_block(x_level_0, h, out_filters=1*self.nr_filters, rescale=False)
            x_level_1_1 = cond_res_block(x_level_1_0, h, out_filters=1*self.nr_filters, rescale=False)
            x_level_2_0 = cond_res_block(x_level_1_1, h, out_filters=2*self.nr_filters, rescale=True)
            x_level_2_1 = cond_res_block(x_level_2_0, h, out_filters=2*self.nr_filters, rescale=False)
            x_level_3_0 = cond_res_block(x_level_2_1, h, out_filters=2*self.nr_filters, rescale=True, dilation=2)
            x_level_3_1 = cond_res_block(x_level_3_0, h, out_filters=2*self.nr_filters, rescale=False, dilation=2)
            x_level_4_0 = cond_res_block(x_level_3_1, h, out_filters=2*self.nr_filters, rescale=True, dilation=4)
            x_level_4_1 = cond_res_block(x_level_4_0, h, out_filters=2*self.nr_filters, rescale=False, dilation=4)

            refine_0 = cond_refine_block([x_level_4_1], h, nr_filters=2*self.nr_filters, out_shape=nn.int_shape(x_level_4_1)[1:3])
            refine_1 = cond_refine_block([x_level_3_1, refine_0], h, nr_filters=2*self.nr_filters, out_shape=nn.int_shape(x_level_3_1)[1:3])
            refine_2 = cond_refine_block([x_level_2_1, refine_1], h, nr_filters=1*self.nr_filters, out_shape=nn.int_shape(x_level_2_1)[1:3])
            refine_3 = cond_refine_block([x_level_1_1, refine_2], h, nr_filters=1*self.nr_filters, out_shape=nn.int_shape(x_level_1_1)[1:3], end=True)
            

            out = self.normalizer(refine_3, h)
            out = self.nonlinearity(out)
            out = nn.conv2d_plus(out, num_filters=self.chns, nonlinearity=None)
            
            self.counters = {} # reset counters

            return out