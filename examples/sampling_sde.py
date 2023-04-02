from spreco.common import utils
from spreco.sampler import sampler

import numpy as np
import os
import sys

eval_path = lambda pr, ch: os.path.join(pr, ch)

s_config = utils.load_config(sys.argv[1])

model_path = eval_path(s_config['log_folder'], s_config['model_name'])
config     = utils.load_config(s_config['log_folder']+'/config.yaml')
save_path  = utils.create_folder(eval_path(s_config['log_folder'], 'samples'))
utils.save_config(s_config, save_path)

a_sampler = sampler(config, target_snr=s_config['target_snr'], skip=s_config['skip'])
a_sampler.init_sampler(model_path, gpu_id=s_config['gpu_id'])

image_n, image  = a_sampler.pc_sampler(s_config['nr_samples'], s_config['steps_pc'])
utils.writecfl(eval_path(save_path, 'pc'), utils.float2cplx(np.array(image[-1])))
utils.writecfl(eval_path(save_path, 'pc'), utils.float2cplx(np.array(image_n[-1])))

image_n, image  = a_sampler.ancestral_sampler(s_config['nr_samples'], s_config['steps_an'])
utils.writecfl(eval_path(save_path, 'an'), utils.float2cplx(np.array(image[-1])))
utils.writecfl(eval_path(save_path, 'an'), utils.float2cplx(np.array(image_n[-1])))