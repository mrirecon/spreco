from spreco.common import utils, ops, sampling_pattern
from spreco.model.ncsn import ncsn

import argparse
import os
import numpy as np
import tqdm
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

def main(config_path):

    config       = utils.load_config(config_path)
    model_config = utils.load_config(config['model_folder']+'/config.yaml')
    model_path   = os.path.join(config['model_folder'], config['model_name'])

    np.random.seed(model_config['seed'])
    os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_id']

    def prepare_simu(config, mask=None):

        kspace    = np.squeeze(np.load(config['ksp_path'])['kspace'])
        nx, ny, _ = kspace.shape

        coilsen   = np.squeeze(utils.bart(1, 'ecalib -m1 -r20 -c0.001', kspace[np.newaxis, ...]))
        std_coils = ops.mifft2(kspace, [nx, ny])
        rss       = np.sum(np.multiply(std_coils, np.squeeze(np.conj(coilsen))), axis=2)

        if mask is None:
            if not config['poisson']:
                mask = sampling_pattern.gen_mask_2D(nx, ny, center_r = config['cal'], undersampling = config['sampling_rate'])
            else:
                mask = utils.bart(1, 'poisson -Y %d -Z %d -y %f -z %f -s 1234 -v -C %d'%(nx, ny, config['fx'], config['fy'], config['cal']))
                mask = np.squeeze(mask)

        und_ksp = kspace*abs(mask[..., np.newaxis])
        coilsen = np.squeeze(utils.bart(1, 'ecalib -m1 -r20 -c0.001', kspace[np.newaxis, ...]))
        coilsen = np.squeeze(coilsen)
        x_ = ops.AT_cart(und_ksp, coilsen, mask, [nx, ny])

        return x_, mask, coilsen, (nx, ny), rss

    def ancestral_sampler(zero_filled, coilsen, mask, grad_logp, sigmas, shape, psnr, lamb=5, nr_samples=10, n_steps_each=50,  burn_in=False, burn_step=0, center=False, disable_z=False):

        images      = []
        scalar      = np.max(abs(zero_filled))
        zero_filled = zero_filled/scalar
        nx, ny      = shape

        if burn_in:
            burn_flag = True
            x_mod = np.random.rand(1,nx,ny,2)
        else:
            x_mod = np.random.rand(nr_samples,nx,ny,2)

        for i in tqdm.tqdm(range(len(sigmas)-1), desc='reconing'):

            sigma     = sigmas[i]
            adj_sigma = sigmas[i+1]
            diff2     = (sigma ** 2 - adj_sigma ** 2)
            tau       = np.sqrt(adj_sigma ** 2 * diff2 / (sigma ** 2))

            if burn_in and i<burn_step:
                z      = np.random.randn(1, nx, ny, 2) * sigma
                labels = [np.int32(i)]
            else:
                z      = np.random.randn(nr_samples, nx, ny, 2) * sigma
                labels = np.array([i]*nr_samples, dtype=np.int32)

            if not disable_z:
                noise_x_ = zero_filled - ops.AHA(utils.float2cplx(z), coilsen[np.newaxis, ...], mask[np.newaxis, ...], shape, axis=(1,2))
                noise_x_ = utils.cplx2float(noise_x_)
            else:
                noise_x_ = utils.cplx2float(zero_filled)

            for _ in range(n_steps_each):
                if burn_in and i > burn_step-1 and burn_flag:
                    print("burned")
                    x_mod = np.squeeze(np.array([x_mod for _ in  range(nr_samples)]))
                    burn_flag = False

                # x_k+1 <-- x_k + tau*score - lambda*std*AHA(x_k + psnr*tau*score) + lambda*std*noise_x score is grad_logp
                score = grad_logp(x_mod, labels)
                grad_data_fidelity = ops.AHA(utils.float2cplx(x_mod), coilsen[np.newaxis, ...], mask[np.newaxis, ...], shape, center=center)
                grad_data_fidelity = utils.cplx2float(grad_data_fidelity)
                noise = np.random.randn(*x_mod.shape) * tau
                x_mod = x_mod + psnr*diff2*score - tau*lamb*grad_data_fidelity + lamb*tau*noise_x_ + noise
                
                if False:
                    noise = np.random.randn(*x_mod.shape) * std
                    x_mod = x_mod + psnr*tau*score + noise
                    iterkspace = ops.A_cart(utils.float2cplx(x_mod), coilsen[np.newaxis, ...], 1-mask[np.newaxis, ...], shape, axis=(1,2))
                    x_mod = utils.cplx2float(ops.AT_cart(und_ksp+iterkspace, coilsen[np.newaxis, ...], np.ones_like(mask[np.newaxis, ...]), shape, axis=(1,2)))

                images.append(x_mod)
        return images


    def get_grad_logp(config, model_path):

        batch_size = None
        x = tf.placeholder(tf.float32, shape=[batch_size]+config['input_shape'])
        h = tf.placeholder(tf.int32, shape=[batch_size]) 
        ins_ncsn = ncsn(config)

        grad_op = ins_ncsn.net.forward(x,h)
        saver   = tf.train.Saver()
        sess    = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path)
        sigmas  = sess.run(ins_ncsn.sigmas)

        def grad_logp(x_in, label):
            return sess.run(grad_op, {x:x_in, h:label})

        return grad_logp, sigmas

    zero_filled, mask, coilsen, shape, rss = prepare_simu(config)
    grad_logp, sigmas = get_grad_logp(model_config, model_path)


    params = {'coilsen': coilsen,
             'mask': mask,
             'grad_logp': grad_logp,
             'sigmas': sigmas, 
             'shape': shape,
             'psnr':config['psnr'],
             'lamb': config['lamb'],
             'nr_samples': config['nr_samples'],
             'burn_in': config['burn_in'],
             'burn_step': config['burn_step'],
             'n_steps_each': config['n_steps_each'],
             'disable_z': config['disable_z']
    }

    images = ancestral_sampler(np.squeeze(zero_filled)[np.newaxis, ...], **params)

    if config['burn_in']:
        images = np.array(images[config['burn_step']*config['n_steps_each']:])
    else:
        images = np.array(images)

    log_path = utils.create_folder(config['workspace'])

    images_cplx = utils.float2cplx(images)
    mean = np.mean(images_cplx[-1,...], axis=0)

    utils.writecfl(log_path+'/rss', rss)
    utils.writecfl(log_path+'/image', images_cplx)
    utils.writecfl(log_path+'/mask', mask)
    utils.writecfl(log_path+'/zero_filled', zero_filled)
    utils.writecfl(log_path+'/mean', mean)
    utils.save_config(config, log_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', metavar='path', default='/home/gluo/lite_prior/recon/ncsn_recon_exp1.yaml', help='path of config file')
    args = parser.parse_args()
    main(args.config)