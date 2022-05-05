from spreco.common import utils,pipe
from spreco.common.options import parts
from spreco.workbench.trainer import trainer

import argparse
import os

def main(args):

    if args.train:
        ## preparation
        if os.path.exists(args.config):
            config = utils.load_config(args.config)
        else:
            raise Exception('The specified config.yaml is not existed, please check!')

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
        utils.color_print('You select neither training nor exporting the model!!!')
        utils.color_print("To see help info python run_pixelcnn.py -h")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', default=True, action='store_true', help='select train')
    parser.add_argument('-i', '--inference', default=False, action='store_true', help='select inference')
    parser.add_argument('--config', type=str, default='/home/gluo/spreco/config_exp/sde_refinenet.yaml')
    parser.add_argument('-l', '--logdir', metavar='', type=str, default='/scratch/gluo/20210707-095003', help='please give logdir at the inference stage,')
    parser.add_argument('-e', '--s_epoch', metavar='', type=int, default=50, help='select epoch at the inference stage')

    args = parser.parse_args()
    main(args)