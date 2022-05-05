from spreco.model.sde import sde, posterior_sampler
from spreco.common import utils, ops, sampling_pattern

import argparse
import os
from functools import partial
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

def main(config_path):

    config = utils.load_config(config_path)
    model_config = utils.load_config(os.path.join(config['model_folder'], 'config.yaml'))
    np.random.seed(model_config['seed'])
    os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_id']

    def prepare_simu(config, mask=None):
        
        kspace = np.squeeze(np.load(config['ksp_path'])['kspace'])

        nx, ny, _ = kspace.shape
        coilsen = np.squeeze(utils.bart(1, 'ecalib -m1 -r20 -c0.001', kspace[np.newaxis, ...]))
        img_shape = [nx, ny]
        std_coils = ops.mifft2(kspace, img_shape)

        rss = np.sum(np.multiply(std_coils, np.squeeze(np.conj(coilsen))), axis=2)

        if mask is None:
            if not config['poisson']:
                mask = sampling_pattern.gen_mask_2D(nx, ny, center_r = config['cal'], undersampling = config['sampling_rate'])
            else:
                mask = utils.bart(1, 'poisson -Y %d -Z %d -y %f -z %f -s 1234 -v -C %d'%(nx, ny, config['fx'], config['fy'], config['cal']))
                mask = np.squeeze(mask)

        und_ksp = kspace*abs(mask[..., np.newaxis])

        coilsen = np.squeeze(utils.bart(1, 'ecalib -m1 -r20 -c0.001', kspace[np.newaxis, ...]))
        coilsen = np.squeeze(coilsen)
        x_ = ops.AT_cart(und_ksp, coilsen, mask, img_shape)

        return x_, mask, coilsen, (nx, ny), rss

    ## data consistency
    zero_filled, mask, coilsen, shape, rss = prepare_simu(config)
    zero_filled = utils.float2cplx(utils.normalize_with_max(zero_filled)) # [-1, 1]

    grad_params = {'coilsen': coilsen[np.newaxis, ...], 'mask': mask[np.newaxis, ...], 'shape': shape, 'center': False}
    AHA         = partial(ops.AHA, **grad_params)

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
                              burn_t=config['burn_t'],
                              map_end = False if 'map_end' not in config.keys() else config['map_end'],
                              last_iteration = 5 if 'last_iteration' not in config.keys() else config['last_iteration'], 
                              last_step_factor= 1 if 'last_step_factor' not in config.keys() else config['last_step_factor'],
                              disable_z = False if 'disable_z' not in config.keys() else config['disable_z'])

    image = ins_sampler.conditional_ancestral_sampler(x, t, sess, AHA, zero_filled[np.newaxis, ...], config['s_stepsize'], st=config['st'])

    if config['burn_in']:
        idx = int(ins_sampler.sde.N*config['burn_t']*config['c_steps']) - config['c_steps']
        image = np.array(image[-idx:])
    else:
        image           = np.array(image)

    log_path = utils.create_folder(config['workspace'])
    utils.writecfl(log_path+'/image', utils.float2cplx(image))
    utils.writecfl(log_path+'/rss', rss)
    utils.writecfl(log_path+'/zero_filled', zero_filled)
    utils.writecfl(log_path+'/mask', mask)
    utils.save_config(config, log_path)

    # this is only for map_end experiment
    if 'compute_residual' in config.keys():
        
        if config['compute_residual']:
            samples = image[-1]
            for count, sample in enumerate(samples):
                sample = utils.float2cplx(sample)[np.newaxis,...]
                residual = np.squeeze(AHA(sample))-zero_filled
                utils.log_to(log_path+'/info.yaml', ["sample_map %d, residual_norm: %f"%(count, np.linalg.norm(residual))])
            
            samples = image[-(ins_sampler.last_iteration+1)]
            for count, sample in enumerate(samples):
                sample = utils.float2cplx(sample)[np.newaxis,...]
                residual = np.squeeze(AHA(sample))-zero_filled
                utils.log_to(log_path+'/info.yaml', ["sample %d, residual_norm: %f"%(count, np.linalg.norm(residual))])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create Figure1')
    parser.add_argument('--config', metavar='path', default='/home/gluo/lite_prior/recon/sde_recon.yaml', help='path of config file')
    args = parser.parse_args()
    main(args.config)
