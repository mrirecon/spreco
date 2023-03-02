from spreco.common import utils, pipe
from spreco.workbench.trainer import trainer

import argparse
import os
import numpy as np

def main(args):

    if os.path.exists(args.config):
        config = utils.load_config(args.config)
    else:
        raise Exception('The specified config.yaml is not existed, please check!')

    if args.train:
        train_files = utils.find_files(config['train_data_path'], config['pattern'])
        test_files  = utils.find_files(config['test_data_path'], config['pattern'])

        def npz_loader(x):
            return utils.npz_loader(x, 'rss')

        def squeeze(x):
            return np.squeeze(x)

        def normalize(x):
            return utils.normalize_with_max(x)

        def slice_image(x):
            return utils.slice_image(x, config['input_shape'])

        parts_funcs = [[npz_loader, squeeze, normalize, slice_image]]

        train_pipe = pipe.create_pipe(parts_funcs,
                            files=train_files,
                            buffer_size=config['num_prepare'],
                            batch_size=config['batch_size']*config['nr_gpu'],
                            shape_info=[config['input_shape']], names=['inputs'])

        test_pipe  = pipe.create_pipe(parts_funcs, test_files,
                                    buffer_size=config['num_prepare'],
                                    batch_size = config['batch_size']*config['nr_gpu'],
                                    shape_info=[config['input_shape']], names=['inputs'])

        go = trainer(train_pipe, test_pipe, config)
        utils.log_to(os.path.join(go.log_path, 'config.yaml'), [utils.get_timestamp(), "The training is starting"], prefix="#")
        go.train()
        utils.log_to(os.path.join(go.log_path, 'config.yaml'), [utils.get_timestamp(), "The training is ending"], prefix="#")
        utils.color_print('TRAINING FINISHED')

    if args.export:
        go = trainer(None, None, config)
        go.export(args.model_path, args.save_folder, config['saved_name'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/home/gluo/github/spreco/config_exp/pixelcnn.yaml')
    parser.add_argument('-t', '--train', default=False, action='store_true', help='select train')
    parser.add_argument('-e', '--export', default=False, action='store_true', help='select export')
    parser.add_argument('-m', '--model_path', type=str, default='None', help='the path of model to be exported')
    parser.add_argument('-s', '--save_folder', type=str, default='None', help='folder for saving the exported model')
    
    args = parser.parse_args()
    main(args)