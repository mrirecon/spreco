from spreco.common import utils, ops
from spreco.model.pixelcnn import pixelcnn

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

    os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_id']

    def prepare_simu(config):

        name, ext = os.path.splitext(config['ksp_path'])
        if ext == '.npz':
            kspace = np.squeeze(np.load(config['ksp_path'])['kspace'])
        elif ext == '.cfl':
            kspace = np.squeeze(utils.readcfl(name))
        else:
            raise TypeError('File type is not supported!')

        nx, ny, _ = kspace.shape

        coilsen   = np.squeeze(utils.bart(1, 'ecalib -m1 -r20 -c0.001', kspace[np.newaxis, ...]))
        std_coils = ops.mifft2(kspace, [nx, ny])
        rss       = np.sum(np.multiply(std_coils, np.squeeze(np.conj(coilsen))), axis=2)

        mask = utils.bart(1, 'poisson -Y %d -Z %d -y %f -z %f -s 1234 -v -C %d'%(nx, ny, config['fx'], config['fy'], config['cal']))
        mask = np.squeeze(mask)

        und_ksp = kspace*abs(mask[..., np.newaxis])
        coilsen = np.squeeze(utils.bart(1, 'ecalib -m1 -r20 -c0.001', kspace[np.newaxis, ...]))
        coilsen = np.squeeze(coilsen)
        x_ = ops.AT_cart(und_ksp, coilsen, mask, [nx, ny])

        l1_recon = utils.bart(1, 'pics -l1 -r 0.01', und_ksp[:,:, np.newaxis, :], coilsen[:, :, np.newaxis, :])

        return x_, mask, coilsen, (nx, ny), rss, l1_recon, und_ksp


    ins_pixelcnn = pixelcnn(model_config)
    ins_pixelcnn.prep(True, batch_size=1)
    saver   = tf.train.Saver()
    sess    = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_path)

    def logp_grads(x):
        return sess.run(ins_pixelcnn.grads, {ins_pixelcnn.x: x})

    def sense_with_prior(zero_filled, coilsen, mask, lamb, iterations):

        scalar      = np.max(abs(zero_filled))
        zero_filled = utils.cplx2float(zero_filled/scalar)
        img_k       = zero_filled

        images=[img_k]
        for itr in tqdm.tqdm(range(iterations)):
            grads = logp_grads(img_k)
            select = np.random.choice(2, zero_filled.shape, p=[config['dropout'],1 - config['dropout']])
            img_k = img_k - select*grads*lamb
            grad_data_fidelity = ops.AHA(utils.float2cplx(img_k), coilsen[np.newaxis, ...], mask[np.newaxis, ...], shape)
            grad_data_fidelity = utils.cplx2float(grad_data_fidelity)
            img_k = img_k + zero_filled - grad_data_fidelity
            maximum = np.max(abs(utils.float2cplx(img_k)))
            zero_filled = zero_filled / maximum
            images.append(img_k)
        return images

    def pocsense_with_prior(zero_filled, und_ksp, coilsen, mask, lamb, iterations):

        scalar      = np.max(abs(zero_filled))
        zero_filled = utils.cplx2float(zero_filled/scalar)
        img_k       = zero_filled
        und_ksp = und_ksp/scalar
        images=[img_k]
        for itr in tqdm.tqdm(range(iterations)):
            grads = logp_grads(img_k)
            select = np.random.choice(2, zero_filled.shape, p=[config['dropout'],1 - config['dropout']])
            img_k = img_k - select*grads*lamb
            
            #maximum = np.max(abs(utils.float2cplx(img_k)))

            iterkspace = ops.A_cart(utils.float2cplx(img_k), coilsen[np.newaxis, ...], 1-mask[np.newaxis, ...], shape, axis=(1,2))
            img_k = utils.cplx2float(ops.AT_cart(und_ksp+iterkspace, coilsen[np.newaxis, ...], np.ones_like(mask[np.newaxis, ...]), shape, axis=(1,2)))

#            zero_filled = zero_filled / maximum
            images.append(img_k)
        return images

    zero_filled, mask, coilsen, shape, rss, l1_recon, und_ksp = prepare_simu(config)

    params = {'coilsen': coilsen,
             'mask': mask,
             'lamb': config['lamb'],
             'iterations': config['iterations'],
    }

    #images = sense_with_prior(np.squeeze(zero_filled)[np.newaxis, ...], **params)
 
    images_pocsense = pocsense_with_prior(np.squeeze(zero_filled)[np.newaxis, ...], und_ksp[np.newaxis, ...], **params)

    log_path = utils.create_folder(config['workspace'])
    #images_cplx = utils.float2cplx(np.array(images))
    images_cplx = utils.float2cplx(np.array(images_pocsense))

    utils.writecfl(log_path+'/rss', rss)
    utils.writecfl(log_path+'/image', images_cplx)
    utils.writecfl(log_path+'/mask', mask)
    utils.writecfl(log_path+'/zero_filled', zero_filled)
    utils.writecfl(log_path+'/l1_recon', l1_recon)
    utils.save_config(config, log_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', metavar='path', default='/home/gluo/github/spreco/scripts/recon_pixelcnn.yaml', help='path of config file')
    args = parser.parse_args()
    main(args.config)