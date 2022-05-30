from spreco.common import utils,pipe
from spreco.common.options import parts
from spreco.workbench.trainer import trainer

import argparse
import os

def main(args):

    if os.path.exists(args.config):
        config = utils.load_config(args.config)
    else:
        raise Exception('The specified config.yaml is not existed, please check!')

    if args.train:

        train_files = utils.find_files(config['train_data_path'], config['pattern'])
        test_files  = utils.find_files(config['test_data_path'], config['pattern'])

        parts_funcs = parts.parse(config['parts'])

        ###
        train_pipe = pipe.create_pipe(parts_funcs,
                            files=train_files,
                            batch_size=config['batch_size']*config['nr_gpu'],
                            shape_info=[config['input_shape'], [1]], names=['inputs', 't'])

        test_pipe  = pipe.create_pipe(parts_funcs, test_files,
                                    batch_size = config['batch_size']*config['nr_gpu'],
                                    shape_info=[config['input_shape'], [1]], names=['inputs', 't'])

        go = trainer(train_pipe, test_pipe, config)
        utils.log_to(os.path.join(go.log_path, 'training files'), train_files, prefix="#")
        utils.log_to(os.path.join(go.log_path, 'config.yaml'), [utils.get_timestamp(), "The training is starting"], prefix="#")
        go.train()
        utils.log_to(os.path.join(go.log_path, 'config.yaml'), [utils.get_timestamp(), "The training is ending"], prefix="#")
        utils.color_print('TRAINING FINISHED')

    else:
        go = trainer(None, None, config)
        go.export(args.model_path, args.save_folder, config['saved_name'])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/home/gluo/spreco/config_exp/sde_refinenet.yaml')
    parser.add_argument('-t', '--train', default=False, action='store_true', help='select train')
    parser.add_argument('-e', '--export', default=False, action='store_true', help='select export')
    parser.add_argument('-m', '--model_path', type=str, default='None', help='the path of model to be exported')
    parser.add_argument('-s', '--save_folder', type=str, default='None', help='folder for saving the exported model')

    args = parser.parse_args()
    main(args)