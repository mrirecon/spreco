from spreco.common import utils, ops
from spreco.model.sde import sde, posterior_sampler

import argparse
import os
from functools import partial
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

def recon(config, workspace, kspace_path, mask_path):

    if not os.path.exists(workspace):
        os.makedirs(workspace)

    model_config = utils.load_config(os.path.join(config['model_folder'], 'config.yaml'))
    np.random.seed(model_config['seed'])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_id']

    kspace = utils.readcfl(kspace_path)
    mask = utils.readcfl(mask_path)
    nx, ny, _ = kspace.shape

    coilsen = np.squeeze(utils.bart(1, 'ecalib -m1 -r20 -c0.0001', kspace[np.newaxis, ...]))
    #coilsen = np.squeeze(utils.bart(1, 'caldir 20', kspace[np.newaxis, ...]))
    coilsen = np.squeeze(coilsen)
    zero_filled = ops.AT_cart(kspace, coilsen, mask, (nx,ny), center=True)

    ## network
    x          = tf.placeholder(tf.float32, shape=[None]+model_config['input_shape']) 
    t          = tf.placeholder(tf.float32, shape=[None]) 
    ins_sde    = sde(model_config)
    _          = ins_sde.net.forward(x, t)
    all_params = tf.trainable_variables()
    saver      = tf.train.Saver()
    sess       = tf.Session()

    sess.run(tf.global_variables_initializer())
    saver.restore(sess, os.path.join(config['model_folder'], config['model_name']))

    ins_sampler     = posterior_sampler(ins_sde, 
                              steps=config['c_steps'],
                              target_snr=config['target_snr'],
                              nr_samples = config['nr_samples'],
                              burn_in=config['burn_in'],
                              burn_t=config['burn_t'])
    
    zero_filled = utils.float2cplx(utils.normalize_with_max(zero_filled)) # [-1, 1]
    grad_params = {'coilsen': coilsen[np.newaxis, ...], 'mask': mask[np.newaxis, ...], 'shape': (nx, ny), 'center': True}
    AHA         = partial(ops.AHA, **grad_params)

    image = ins_sampler.conditional_ancestral_sampler(x, t, sess, AHA, zero_filled[np.newaxis, ...], config['s_stepsize'], st=config['st'])

    mmse = np.mean(image[-1], axis=0)
    var  = np.std(image[-1], axis=0)
    tmp = os.path.split(os.path.splitext(kspace_path)[0])[-1]
    idx = tmp.split('_')[-1]

    utils.writecfl(workspace+'/mmse_%s'%idx, utils.float2cplx(mmse))
    utils.writecfl(workspace+'/var_%s'%idx, utils.float2cplx(var))
    utils.writecfl(workspace+'/zero_filled_%s'%idx, zero_filled)

def metric(file1, file2):
    img1 = abs(utils.readcfl(file1))
    img2 = abs(utils.readcfl(file2))
    img1_normalized = img1/np.linalg.norm(img1)
    img2_normalized = img2/np.linalg.norm(img2)
    print("psnr: %f, ssim: %f"%(utils.psnr(img1_normalized, img2_normalized), utils.ssim(img1_normalized, img2_normalized)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="benchmark")
    parser.add_argument('--config', metavar='path', default='/home/gluo/lite_prior/recon/benchmark.yaml', help='path of config file')
    parser.add_argument('--workspace', metavar='path', default='/home/gluo/workspace/sampling_posterior/benchmark', help='path of config file')
    parser.add_argument('--kspace', metavar='path', default='/home/gluo/workspace/sampling_posterior/benchmark', help='path of config file')
    parser.add_argument('--mask', metavar='path', default='/home/gluo/workspace/sampling_posterior/benchmark', help='path of config file')
    parser.add_argument('-t', '--metric', default=False, action='store_true', help='compute metrics')
    parser.add_argument('--mmse', metavar='path', default='/home/gluo/workspace/sampling_posterior/benchmark', help='path of config file')
    parser.add_argument('--ground', metavar='path', default='/home/gluo/workspace/sampling_posterior/benchmark', help='path of config file')
    args = parser.parse_args()

    if args.metric:
        metric(args.mmse, args.ground)
    else:
        config = utils.load_config(args.config)
        utils.save_config(config, args.workspace)
        recon(config, args.workspace, args.kspace, args.mask)
