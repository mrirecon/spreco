from spreco.common import utils
from spreco.common import cu_ops as ops
from spreco.common.sampling_pattern import gen_mask

from spreco.model.sde import sde, posterior_sampler
from functools import partial

import argparse
import h5py
import numpy as np
import cupy as cp
import os
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def load_kspace(path):

    fs=h5py.File(path, 'r')

    kspace=fs['kspace']
    kspace = np.transpose(kspace, [2,3,1,0])
    kspace = (kspace[0::2, ...]+kspace[1::2, ...])/2.

    if 'mask' in fs.keys():
        mask = np.array(fs['mask'])
        mm   = np.array([mask for _ in range(kspace.shape[1])])
    else:
        mm = None

    if 'reconstruction_rss' in fs.keys():
        rss = np.array(fs['reconstruction_rss'])
    else:
        rss = None

    fs.close()

    return kspace, mm, rss

def recon(config, kspace, mask):

    model_config = utils.load_config(os.path.join(config['model_folder'], 'config.yaml'))
    np.random.seed(model_config['seed'])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_id']

    ## network
    x          = tf.placeholder(tf.float32, shape=[None]+model_config['input_shape']) 
    t          = tf.placeholder(tf.float32, shape=[None]) 
    ins_sde    = sde(model_config)
    _          = ins_sde.net.forward(x, t)
    all_params = tf.trainable_variables()
    saver      = tf.train.Saver()
    config_ = tf.ConfigProto()
    config_.gpu_options.allow_growth=True
    sess       = tf.Session(config=config_)

    sess.run(tf.global_variables_initializer())
    saver.restore(sess, os.path.join(config['model_folder'], config['model_name']))

    ins_sampler     = posterior_sampler(ins_sde, 
                              steps=config['c_steps'],
                              target_snr=config['target_snr'],
                              nr_samples = config['nr_samples'],
                              burn_in=config['burn_in'],
                              burn_t=config['burn_t'],
                              disable_z=config['disable_z'])

    nx, ny, _ = kspace.shape
    und_kspace = kspace*mask[..., np.newaxis]

    coilsen = np.squeeze(utils.bart(1, 'ecalib -m1 -r20 -c0.0001', und_kspace[np.newaxis, ...]))
    coilsen = cp.asarray(coilsen)
    und_kspace = cp.asarray(und_kspace)

    coil_comb = ops.AT_cart(kspace, coilsen, np.ones_like(mask), (nx,ny), center=True)

    zero_filled = ops.AT_cart(und_kspace, coilsen, cp.asarray(mask), (nx,ny), center=True)
    zero_filled = utils.float2cplx(utils.normalize_with_max(zero_filled)) # [-1, 1]
    grad_params = {'coilsen': coilsen[np.newaxis, ...], 'mask': cp.asarray(mask[np.newaxis, ...]), 'shape': (nx, ny), 'center': True}
    AHA         = partial(ops.AHA, **grad_params)

    image = ins_sampler.conditional_ancestral_sampler(x, t, sess, AHA, zero_filled[np.newaxis, ...], config['s_stepsize'], st=config['st'])
    samples = utils.float2cplx(image[-1])
    mmse = np.mean(samples, axis=0)
    var  = np.std(samples, axis=0)
    zero_fill = cp.asnumpy(zero_filled)
    coil_comb = cp.asnumpy(coil_comb)

    return mmse, var, zero_fill, coil_comb

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', metavar='path', default='/home/gluo/workspace/sampling_posterior/revision/fastmri.yaml', help='')
    parser.add_argument('--h5path', metavar='path', default='/home/ague/data/gluo/nyu_dataset/multicoil_val/file_brain_AXFLAIR_200_6002536.h5', help='')
    parser.add_argument('--workspace', metavar='path', default='/home/gluo/workspace', help='')
    parser.add_argument('--gpu_id', metavar='gpu_id', default='0', help='')

    args   = parser.parse_args()

    config = utils.load_config(args.config)
    config['gpu_id'] = args.gpu_id

    kspace, mask, official_rss = load_kspace(args.h5path)
    if mask is None:
        m = gen_mask(kspace.shape[1], config['factor'])
        mask   = np.array([m for _ in range(kspace.shape[0])])

    path, _  = os.path.splitext(args.h5path)
    filename = os.path.basename(path)
    save_path=os.path.join(args.workspace, filename)
    utils.create_folder(save_path, time=False)

    mmses = []
    vars  = []
    zero_fills=[]
    coil_combs=[]
    nr_slice = kspace.shape[-1]
    
    for i in range(nr_slice):
        mmse, var, zero_fill, coil_comb = recon(config, kspace[...,i], mask)
        tf.reset_default_graph()
        mmses.append(mmse)
        vars.append(var)
        zero_fills.append(zero_fill)
        coil_combs.append(coil_comb)

    utils.writecfl(save_path+'/mmse', np.array(mmses))
    utils.writecfl(save_path+'/var', np.array(vars))
    utils.writecfl(save_path+'/zero_filled', np.array(zero_fills))
    utils.writecfl(save_path+'/official_rss', np.array(official_rss))
    utils.writecfl(save_path+'/coil_combs', np.array(coil_combs))
    utils.writecfl(save_path+'/mask', mask)
    utils.save_config(config, save_path)