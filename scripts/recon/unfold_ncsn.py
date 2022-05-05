from spreco.common import utils, ops
from spreco.model.ncsn import ncsn

import argparse
import os
import numpy as np
import tqdm
from functools import partial
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

def main(config_path):

    config = utils.load_config(config_path)
    model_config = utils.load_config(config['model_folder']+'/config.yaml')
    model_path = os.path.join(config['model_folder'], config['model_name'])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_id']
    np.random.seed(model_config['seed'])

    def get_mask(shape, o, center=True):
        """
        0 odd
        1 even
        cemter set the central two lines 
        """
        mask = np.zeros(shape)
        if o == 0:
            mask[0::2, ...] = 1
        if o == 1:
            mask[1::2, ...] = 1
        if center:
            mask[127:128, ...] = 1
        return mask

    def prepare_simu(kspace_path, o, center, shift):
        """to simulate single coil acquire kspace"""

        kspace = np.squeeze(np.load(kspace_path)['kspace'])
        nx, ny, _ = kspace.shape

        coilsen   = np.squeeze(utils.bart(1, 'caldir 40', kspace[np.newaxis, ...]))
        img_shape = [nx, ny]
        std_coils = ops.mifft2(kspace, img_shape)
    
        rss = np.sum(np.multiply(std_coils, np.squeeze(np.conj(coilsen))), axis=2)
        rss = np.roll(rss, shift=shift, axis=0)
        utils.writecfl('/scratch/gluo/zero_filled', rss)
        ksp = ops.mfft2(rss, img_shape)

        mask  = get_mask(ksp.shape, o, center)
        kspx2 = ksp*mask  #x,y -> (0,1)
        x_    = ops.mifft2(kspx2, img_shape)

        

        def A_cart(img, mask, shape, axis=(0,1), center=False):
            kspace = ops.mfft2(img, shape, axis=axis, center=center)
            kspace = np.multiply(kspace, mask)
            return kspace
            
        def AT_cart(kspace, mask, shape, axis=(0,1), center=False):
            """
            adjoint cartesian AT
            
            """
            img = ops.mifft2(kspace*mask, shape, axis=axis, center=center)
            return img
        
        def AHA(img, mask, shape, axis=(0,1), center=False):
            tmp = A_cart(img, mask, shape, axis, center)
            ret = AT_cart(tmp, mask, shape, axis=axis, center=center)
            return ret

        params1 = {'mask': mask[np.newaxis, ...], 'shape': img_shape, 'axis': (1,2)}
        AHA    = partial(AHA, **params1)
        params2 = {'mask': mask, 'shape': img_shape, 'axis': (0,1)}
        A      = partial(A_cart, **params2)
        return x_[np.newaxis, ...], rss, ksp, mask, AHA, A
    
    def get_grad_logp():

        batch_size = None
        x = tf.placeholder(tf.float32, shape=[batch_size]+model_config['input_shape'])
        h = tf.placeholder(tf.int32, shape=[batch_size]) 
        ins_ncsn = ncsn(model_config)

        grad_op = ins_ncsn.net.forward(x, h)
        saver   = tf.train.Saver()
        sess    = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path)
        sigmas = sess.run(ins_ncsn.sigmas)

        def grad_logp(x_in, label):
            return sess.run(grad_op, {x:x_in, h:label})

        return grad_logp, sigmas


    def ancestral_sampler(zero_filled, AHA, grad_logp, sigmas, shape, lamb=5, nr_samples=10, n_steps_each=50, burn_in=False, burn_step=0):

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
            tau       = (sigma ** 2 - adj_sigma ** 2)
            std       = np.sqrt(adj_sigma ** 2 * tau / (sigma ** 2))

            if burn_in and i<burn_step:
                z      = np.random.randn(1, nx, ny, 2) * sigma
                labels = [np.int32(i)]
            else:
                z      = np.random.randn(nr_samples, nx, ny, 2) * sigma
                labels = np.array([i]*nr_samples, dtype=np.int32)

            noise_x_ = utils.cplx2float(zero_filled + AHA(utils.float2cplx(z)))

            for _ in range(n_steps_each):

                if burn_in and i > burn_step-1 and burn_flag:
                    print("burned")
                    x_mod = np.squeeze(np.array([x_mod for _ in  range(nr_samples)]))
                    burn_flag = False

                # x_k+1 <-- x_k + tau*score - lambda*std*AHA(x_k + tau*score) + lambda*std*noise_x score is grad_logp
                score  = grad_logp(x_mod, labels)
                grad_data_fidelity = AHA(utils.float2cplx(x_mod+tau*score))
                grad_data_fidelity = utils.cplx2float(grad_data_fidelity)
                noise = np.random.randn(*x_mod.shape)*std
                x_mod = x_mod + tau*score - std*lamb*grad_data_fidelity + lamb*std*noise_x_ + noise
                images.append(x_mod)

        return images, scalar

    def run_recon():

        x_, rss, kspace, mask, AHA, A = prepare_simu(config['ksp_path'], config['o'], config['center'], config['shift'])
        grad_logp, sigmas = get_grad_logp()
        params ={
         'lamb': config['lamb'],
         'nr_samples': config['nr_samples'],
         'burn_in': config['burn_in'],
         'burn_step': config['burn_step'],
         'n_steps_each': config['n_steps_each']
        }
        images, scalar = ancestral_sampler(x_, AHA, grad_logp, sigmas, x_.shape[1:], **params)
        return x_, images, rss, kspace, mask, A, scalar
    zero_filled, images, rss, kspace, mask, A, scalar = run_recon()

    if config['burn_in']:
        images = np.array(images[config['burn_step']*config['n_steps_each']:])
    else:
        images = np.array(images)

    images_cplx = utils.float2cplx(images)    
    images_cplx = utils.float2cplx(images)
    mean = np.mean(images_cplx[-1,...], axis=0)
    std  = np.std(images_cplx[-1,...], axis=0)

    log_path = utils.create_folder(config['workspace'])

    samples = images_cplx[-1]
    for count, sample in enumerate(samples):
        restore_ksp = np.squeeze(A(sample*scalar))
        residual = restore_ksp - kspace*mask
        utils.log_to(log_path+'/info.yaml', ["sample %d, residual_norm: %f"%(count, np.linalg.norm(residual))])
        utils.writecfl(log_path+'/restore_ksp_%d'%count, restore_ksp)

    utils.writecfl(log_path+'/image', images_cplx)
    utils.writecfl(log_path+'/zero_filled', zero_filled)
    utils.writecfl(log_path+'/rss', rss)
    utils.writecfl(log_path+'/kspace', kspace)
    utils.writecfl(log_path+'/mask', mask)
    utils.writecfl(log_path+'/mean', mean)
    utils.writecfl(log_path+'/std', std)
    utils.save_config(config, log_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Figure1')
    parser.add_argument('--config', metavar='path', default='/home/gluo/lite_prior/recon/unfold_ncsn.yaml', help='path of config file')
    args = parser.parse_args()
    main(args.config)
