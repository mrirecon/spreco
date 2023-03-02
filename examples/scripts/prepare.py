from spreco.common import utils, ops, sampling_pattern

import argparse
import os
import numpy as np
import h5py

def main(h5_path, workspace, config):

    if not os.path.exists(workspace):
        os.makedirs(workspace)
    np.random.seed(1234)
    def read_ksp(filepath):
        try:
            fs = h5py.File(filepath, "r")
            kspaces = fs['kspace']
            kspaces = np.transpose(kspaces, [2,3,1,0])
        except:
            return None
        return np.split(kspaces, kspaces.shape[-1], axis=-1)

    kspaces = read_ksp(h5_path)

    for pos, ksp in enumerate(kspaces):

        kspace = np.squeeze(utils.bart(1, 'cc -p 10 -A',np.squeeze(ksp[0::2])[np.newaxis, ...]))
        nx, ny, _ = kspace.shape

        coilsen = np.squeeze(utils.bart(1, 'ecalib -r20 -m1 -c0.0001', kspace[np.newaxis, ...]))
        img_shape = [nx, ny]
        std_coils = ops.mifft2(kspace, img_shape, center=True)

        rss = np.sum(np.multiply(std_coils, np.squeeze(np.conj(coilsen))), axis=2)

        if config['poisson']:
            mask = np.squeeze(utils.bart(1, 'poisson -Y %d -Z %d -y %f -z %f -s %s -v -C %d'%(nx, ny, config['fx'], config['fy'], str(pos), config['calr'])))
        else:
            mask = sampling_pattern.gen_mask_2D(nx, ny, center_r = config['calr'], undersampling = config['ksp_rate'])

        und_ksp = kspace*abs(mask[..., np.newaxis])

        utils.writecfl(workspace+'/rss_%s'%pos, rss)
        utils.writecfl(workspace+'/und_ksp_%s'%pos, und_ksp)
        utils.writecfl(workspace+'/kspace_%s'%pos, kspace)
        utils.writecfl(workspace+'/mask_%s'%pos, mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preparation for benchmark")
    parser.add_argument('--workspace', metavar='path', default='/home/gluo/workspace/sampling_posterior/benchmark', help='path of config file')
    parser.add_argument('--h5_path', metavar='path', default='/home/gluo/lite_prior/recon/benchmark.yaml', help='path of config file')
    parser.add_argument('--config', metavar='path', default='/home/gluo/lite_prior/recon/benchmark.yaml', help='path of config file')
    parser.add_argument('--ksp_rate', metavar='path', type=float, default=0.2, help='path of config file')
    parser.add_argument('--calr', metavar='path', type=int, default=20, help='path of config file')
    args = parser.parse_args()
    main(args.h5_path, args.workspace, utils.load_config(args.config))