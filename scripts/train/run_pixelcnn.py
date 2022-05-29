from spreco.common import utils, pipe
from spreco.workbench.trainer import trainer
from spreco.model.logistic_loss import sample_from_discretized_mix_logistic_2

import os
import numpy as np

config_file = '/home/gluo/github/spreco/config_exp/pixelcnn.yaml'
config = utils.load_config(config_file)

train_files = utils.find_files(config['train_data_path'], config['pattern'])
test_files  = utils.find_files(config['test_data_path'], config['pattern'])

def npz_loader(x):
    return utils.npz_loader(x, 'rss')

def squeeze(x):
    return np.squeeze(x)

def normalize(x):
    return utils.normalize_with_max(x)

def slice_image(x):
    return utils.slice_image(x, [256, 256, 2])

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
