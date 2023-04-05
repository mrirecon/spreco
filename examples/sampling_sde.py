from spreco.common import utils
from spreco.sampler import sampler

import numpy as np
import os
import sys

evalpath = lambda pr, ch: os.path.join(pr, ch)
savecfl  = lambda p, a: utils.writecfl(p, utils.float2cplx(a))

s_config = utils.load_config(sys.argv[1])

model_path = evalpath(s_config['log_folder'], s_config['model_name'])
config     = utils.load_config(s_config['log_folder']+'/config.yaml')
save_path  = utils.create_folder(evalpath(s_config['log_folder'], 'samples'))
utils.save_config(s_config, save_path)

def check_paras(m_config, n_config):
    """
    check if the sigma is acceptable for score network
    """
    
    
    if m_config['sigma_max'] >=  n_config['sigma_max']:
        m_config['sigma_max'] = n_config['sigma_max']
    else:
        raise ValueError("sampling config for sigma_max is wrong!")
    
    if m_config['sigma_min'] <=  n_config['sigma_min']:
        m_config['sigma_min'] = n_config['sigma_min']
    else:
        raise ValueError("sampling config for sigma_min is wrong!")

    m_config['N'] = n_config['N']

    return m_config

check_paras(config, s_config)

print("INFO -> sigma type: %s, sigma max: %.4f, simga min: %.4f, discrete steps: %d "
      %(s_config['sigma_type'], s_config['sigma_max'], s_config['sigma_min'], s_config['N']))

a_sampler = sampler(config, s_config['target_snr'], s_config['sigma_type'])
a_sampler.init_sampler(model_path, gpu_id=s_config['gpu_id'])

image_n, image  = a_sampler.pc_sampler(s_config['nr_samples'], s_config['steps_pc'])
savecfl(evalpath(save_path, 'pc'), image[-1])
savecfl(evalpath(save_path, 'pc_n'), image_n[-1])

image_n, image  = a_sampler.ancestral_sampler(s_config['nr_samples'], s_config['steps_an'])
savecfl(evalpath(save_path, 'an'), image[-1])
savecfl(evalpath(save_path, 'an_n'), image_n[-1])