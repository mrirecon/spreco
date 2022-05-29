from spreco.model.utils import int_shape

import numpy as np 
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keepdims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x-m2), axis))

def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis, keepdims=True)
    return x - m - tf.log(tf.reduce_sum(tf.exp(x-m), axis, keepdims=True))

def discretized_mix_logistic_loss(x,l,itg_interval=255.0, sum_all=True):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    xs = int_shape(x) # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = int_shape(l) # predicted distribution, e.g. (B,32,32,100)
    nr_mix = int(ls[-1] / 10) # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:,:,:,:nr_mix]
    l = tf.reshape(l[:,:,:,nr_mix:], xs + [nr_mix*3])
    means = l[:,:,:,:,:nr_mix]
    log_scales = tf.maximum(l[:,:,:,:,nr_mix:2*nr_mix], -7.)
    coeffs = tf.nn.tanh(l[:,:,:,:,2*nr_mix:3*nr_mix])
    x = tf.reshape(x, xs + [1]) + tf.zeros(xs + [nr_mix]) # here and below: getting the means and adjusting them based on preceding sub-pixels
    m2 = tf.reshape(means[:,:,:,1,:] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :], [xs[0],xs[1],xs[2],1,nr_mix])
    m3 = tf.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0],xs[1],xs[2],1,nr_mix])
    means = tf.concat([tf.reshape(means[:,:,:,0,:], [xs[0],xs[1],xs[2],1,nr_mix]), m2, m3],3)
    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1./itg_interval)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1./itg_interval)
    cdf_min = tf.nn.sigmoid(min_in)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in) # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in) # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2.*tf.nn.softplus(mid_in) # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    log_probs = tf.where(x < -0.999, log_cdf_plus, tf.where(x > 0.999, log_one_minus_cdf_min, tf.where(cdf_delta > 1e-5, tf.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))))

    log_probs = tf.reduce_sum(log_probs,3) + log_prob_from_logits(logit_probs)
    if sum_all:
        return -tf.reduce_sum(log_sum_exp(log_probs))
    else:
        return -tf.reduce_sum(log_sum_exp(log_probs),[1,2])

def sample_from_discretized_mix_logistic(l,nr_mix):
    ls = int_shape(l)
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix*3])
    # sample mixture indicator from softmax
    sel = tf.one_hot(tf.argmax(logit_probs - tf.log(-tf.log(tf.random_uniform(logit_probs.get_shape(), minval=1e-5, maxval=1. - 1e-5))), 3), depth=nr_mix, dtype=tf.float32)
    sel = tf.reshape(sel, xs[:-1] + [1,nr_mix])
    # select logistic parameters
    means = tf.reduce_sum(l[:,:,:,:,:nr_mix]*sel,4)
    log_scales = tf.maximum(tf.reduce_sum(l[:,:,:,:,nr_mix:2*nr_mix]*sel,4), -7.)
    coeffs = tf.reduce_sum(tf.nn.tanh(l[:,:,:,:,2*nr_mix:3*nr_mix])*sel,4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = tf.random_uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)
    x = means + tf.exp(log_scales)*(tf.log(u) - tf.log(1. - u))
    x0 = tf.minimum(tf.maximum(x[:,:,:,0], -1.), 1.)
    x1 = tf.minimum(tf.maximum(x[:,:,:,1] + coeffs[:,:,:,0]*x0, -1.), 1.)
    x2 = tf.minimum(tf.maximum(x[:,:,:,2] + coeffs[:,:,:,1]*x0 + coeffs[:,:,:,2]*x1, -1.), 1.)
    return tf.concat([tf.reshape(x0,xs[:-1]+[1]), tf.reshape(x1,xs[:-1]+[1]), tf.reshape(x2,xs[:-1]+[1])],3)


def discretized_mix_logistic_loss_2(x, l, itg_interval=255.0, sum_all=True, return_prob=False):
    """
    1. log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval 
    2. this loss assume that real part is linearly dependent on the imaginary part
    """
    xs = int_shape(x) 
    ls = int_shape(l) 

    # unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 6) 
    logit_probs = l[:,:,:,:nr_mix]
    l = tf.reshape(l[:,:,:,nr_mix:], xs[:-1] + [5,nr_mix])
    means = l[:,:,:,:2,:]
    log_scales = tf.maximum(l[:,:,:,2:4,:], -7.)
    coeffs = tf.nn.tanh(l[:,:,:,4:,:])

    # getting the means and adjusting them based on preceding sub-pixels
    x = tf.reshape(x, xs + [1]) + tf.zeros(xs + [nr_mix]) 
    m2 = tf.reshape(means[:,:,:,1,:] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :], [xs[0],xs[1],xs[2],1,nr_mix])
    means = tf.concat([tf.reshape(means[:,:,:,0,:], [xs[0],xs[1],xs[2],1,nr_mix]), m2],3)
    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1./itg_interval)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1./itg_interval)
    cdf_min = tf.nn.sigmoid(min_in)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in) # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in) # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2.*tf.nn.softplus(mid_in) # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    log_probs = tf.where(x < -0.999, log_cdf_plus, tf.where(x > 0.999, log_one_minus_cdf_min, tf.where(cdf_delta > 1e-5, tf.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(itg_interval/2.))))

    log_probs = tf.reduce_sum(log_probs,3) + log_prob_from_logits(logit_probs)
    if sum_all:
        if return_prob:
            print(log_sum_exp(log_probs).get_shape().as_list())
            return -tf.reduce_sum(log_sum_exp(log_probs)), log_sum_exp(log_probs)
        else:
            return -tf.reduce_sum(log_sum_exp(log_probs))
    else:
        return -tf.reduce_sum(log_sum_exp(log_probs),[1,2])

def sample_from_discretized_mix_logistic_2(l,nr_mix):
    ls = int_shape(l)
    xs = ls[:-1] + [2]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:,:,:,nr_mix:], xs[:-1] + [5,nr_mix])
    # sample mixture indicator from softmax
    sel = tf.one_hot(tf.argmax(logit_probs - tf.log(-tf.log(tf.random_uniform(logit_probs.get_shape(), minval=1e-5, maxval=1. - 1e-5))), 3), depth=nr_mix, dtype=tf.float32)
    sel = tf.reshape(sel, xs[:-1] + [1,nr_mix])
    # select logistic parameters
    means = tf.reduce_sum(l[:,:,:,:2,:]*sel,4)
    log_scales = tf.maximum(tf.reduce_sum(l[:,:,:,2:4,:]*sel,4), -7.)
    coeffs = tf.reduce_sum(tf.nn.tanh(l[:,:,:,4:,:])*sel,4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = tf.random_uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)
    x = means + tf.exp(log_scales)*(tf.log(u) - tf.log(1. - u))
    x0 = tf.minimum(tf.maximum(x[:,:,:,0], -1.), 1.)
    x1 = tf.minimum(tf.maximum(x[:,:,:,1] + coeffs[:,:,:,0]*x0, -1.), 1.)
    #x2 = tf.minimum(tf.maximum(x[:,:,:,2] + coeffs[:,:,:,1]*x0 + coeffs[:,:,:,2]*x1, -1.), 1.)
    return tf.concat([tf.reshape(x0,xs[:-1]+[1]), tf.reshape(x1,xs[:-1]+[1])],3)


def discretized_mix_logistic_loss_1(x,l, itg_interval=255.0, sum_all=True,return_prob=False):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    xs = int_shape(x) # true image
    ls = int_shape(l) # predicted distribution
    nr_mix = int(ls[-1] / 3) # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:,:,:,:nr_mix]
    
    l = tf.reshape(l[:,:,:,nr_mix:], xs[:-1] + [2,nr_mix])
    means = l[:,:,:,:1,:]
    log_scales = tf.maximum(l[:,:,:,1:2,:], -7.)
    
    x = tf.reshape(x, xs + [1]) + tf.zeros(xs + [nr_mix]) # here and below: getting the means and adjusting them based on preceding sub-pixels    
    
    means = tf.reshape(means[:,:,:,0,:], [xs[0],xs[1],xs[2],1,nr_mix])
    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1./itg_interval)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1./itg_interval)
    cdf_min = tf.nn.sigmoid(min_in)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in) # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in) # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2.*tf.nn.softplus(mid_in) # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    log_probs = tf.where(x < -0.999, log_cdf_plus, tf.where(x > 0.999, log_one_minus_cdf_min, tf.where(cdf_delta > 1e-5, tf.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(itg_interval/2.))))

    log_probs = tf.reduce_sum(log_probs,3) + log_prob_from_logits(logit_probs)
    if sum_all:
        if return_prob:
            print(log_sum_exp(log_probs).get_shape().as_list())
            return -tf.reduce_sum(log_sum_exp(log_probs)), log_sum_exp(log_probs)
        else:
            return -tf.reduce_sum(log_sum_exp(log_probs))
    else:
        return -tf.reduce_sum(log_sum_exp(log_probs),[1,2])


def sample_from_discretized_mix_logistic_1(l,nr_mix):
    ls = int_shape(l)
    xs = ls[:-1] + [1]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix] # pi
    l = tf.reshape(l[:,:,:,nr_mix:], xs[:-1] + [2,nr_mix])

    # sample mixture indicator from softmax
    sel = tf.one_hot(tf.argmax(logit_probs - tf.log(-tf.log(tf.random_uniform(logit_probs.get_shape(), minval=1e-5, maxval=1. - 1e-5))), 3), depth=nr_mix, dtype=tf.float32)
    sel = tf.reshape(sel, xs[:-1] + [1,nr_mix])

    # select logistic parameters
    means = tf.reduce_sum(l[:,:,:,:1,:]*sel,4)
    log_scales = tf.maximum(tf.reduce_sum(l[:,:,:,1:,:]*sel,4), -7.)
    
    # we don't actually round to the nearest 8bit value when sampling
    u = tf.random_uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)
    x = means + tf.exp(log_scales)*(tf.log(u) - tf.log(1. - u))

    x0 = tf.minimum(tf.maximum(x[:,:,:,0], -1.), 1.)
    
    return tf.reshape(x0, xs)

def discretized_mix_logistic_loss_idp_2(x, l, itg_interval=255.0, sum_all=True, return_prob=False):
    """
    1. this loss assume that real part and the imaginary part are independent of each other
    """
    
    xs = int_shape(x)
    ls = int_shape(l)

    # unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 5) 
    logit_probs = l[:,:,:,:nr_mix]
    l = tf.reshape(l[:,:,:,nr_mix:], xs[:-1] + [4,nr_mix])
    means = l[:,:,:,:2,:]
    log_scales = tf.maximum(l[:,:,:,2:4,:], -7.)

    # adjusting means
    x = tf.reshape(x, xs + [1]) + tf.zeros(xs + [nr_mix]) 
    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1./itg_interval)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1./itg_interval)
    cdf_min = tf.nn.sigmoid(min_in)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in) # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in) # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2.*tf.nn.softplus(mid_in) # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    log_probs = tf.where(x < -0.999, log_cdf_plus, tf.where(x > 0.999, log_one_minus_cdf_min, tf.where(cdf_delta > 1e-5, tf.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(itg_interval/2.))))

    log_probs = tf.reduce_sum(log_probs,3) + log_prob_from_logits(logit_probs)
    if sum_all:
        if return_prob:
            print(log_sum_exp(log_probs).get_shape().as_list())
            return -tf.reduce_sum(log_sum_exp(log_probs)), log_sum_exp(log_probs)
        else:
            return -tf.reduce_sum(log_sum_exp(log_probs))
    else:
        return -tf.reduce_sum(log_sum_exp(log_probs),[1,2])


def discretized_mix_logistic_loss_bivariate(x, l, itg_interval=255.0, sum_all=True, return_prob=False):
    """log-likelihood for mixture of discretized bivariate logistics"""
    return None
