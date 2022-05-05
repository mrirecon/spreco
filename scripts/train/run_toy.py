from spreco.model.toy import toy
from spreco.model.gmm import Gaussian_Mixture
from spreco.common import utils
from spreco.common.custom_adam import AdamOptimizer
import tensorflow as tf

import argparse
import os
import numpy as np
import time
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

def main(args):


    config  = utils.load_config(args.config)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_id']

    log_path   = utils.create_folder(config['log_folder'])
    utils.save_config(config, log_path)

    tf.random.set_seed(config['seed'])
    teacher   = Gaussian_Mixture(2)
    student   = toy(config)
    optimizer = AdamOptimizer(config['lr'], beta1=0.9, beta2=0.999)

    @tf.function
    def train_step(samples):
        with tf.GradientTape() as tape:
            loss = student.score_estimation(samples)
            grads = tape.gradient(loss, student.score_net.trainable_variables)
            optimizer.apply_gradients(zip(grads, student.score_net.trainable_variables))
        return loss

    steps = 0
    begin      = time.time()
    while steps < config['train_steps']:
        
        ts = tf.squeeze(teacher.sample(config['batch_size']))
        l = train_step(ts)

        if steps% config['save_interval'] == 0:
            info = "Step %d, time = %ds, train loss = %.4f" % (steps, time.time()-begin, l)
            begin=time.time() 
            print(info)
            student.score_net.save_weights(os.path.join(log_path, 'toy_'+str(steps)))
        steps = steps + 1


    right_bound = config['right_bound']
    left_bound  = config['left_bound'] 
    view_factor = config['view_factor']

    def savefig(samples, title, name):
        plt.axis('equal')
        plt.xlim(view_factor*left_bound, view_factor*right_bound)
        plt.ylim(view_factor*left_bound, view_factor*right_bound)
        plt.scatter(samples[..., 0], samples[..., 1],s=13, marker='x', c='tab:green')
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title(title)
        plt.tight_layout(pad=0.4, w_pad=0.5)
        plt.savefig(name)
        plt.clf()

    # samples from teacher
    samples = teacher.sample(128)
    savefig(samples, 'samples from teacher', os.path.join(log_path, 'samples_1'))
    np.savez(os.path.join(log_path, 'teacher'), samples=samples)

    # samples from langevin dynamics
    init = tf.random.uniform((128,2))*(right_bound - left_bound) + left_bound
    samples, s_t = student.langevin_dynamics(teacher.score, init)
    savefig(samples, 'samples from langevin dynamics', os.path.join(log_path,'samples_2'))
    np.savez(os.path.join(log_path, 'ld'), samples=s_t)

    # samples from annealed langevin dynamics
    init = tf.random.uniform((128,2))*(right_bound - left_bound) + left_bound
    samples, s_t = student.anneal_langevin_dynamics(teacher.score, init, student.sigmas)
    savefig(samples, 'samples from annealed langevin dynamics', os.path.join(log_path, 'samples_3'))
    np.savez(os.path.join(log_path, 'ald'), samples=s_t)

    # samples from langevin dynamics with learned score
    init = tf.random.uniform((128,2))*(right_bound - left_bound) + left_bound
    samples, s_t = student.langevin_dynamics(student.score_net, init)
    savefig(samples, 'samples from langevin dynamics with learned score', os.path.join(log_path, 'samples_4'))
    np.savez(os.path.join(log_path, 'ldl'), samples=s_t)

    # samples from posterior               p(y|x) -> y= Ax+z
    logp = Gaussian_Mixture(2, mix_prob=[0.05, 0.95])
    scale = 1.

    # 1
    init = tf.random.uniform((128,2))*(right_bound - left_bound) + left_bound
    def new_score(x, sigma):
        tmp = teacher.score(x, sigma)+logp.score(x, sigma)
        return tmp*scale
    samples, s_t = student.anneal_langevin_dynamics(new_score, init, student.sigmas, n_steps_each=100)
    savefig(samples, 'samples from posterior', os.path.join(log_path, 'samples_5'))
    np.savez(os.path.join(log_path, 'pald'), samples=s_t)

    # 2
    init = tf.random.uniform((128,2))*(right_bound - left_bound) + left_bound
    def new_score(x):
        tmp = student.score_net(x)+logp.score(x)
        return tmp*scale
    samples, s_t = student.langevin_dynamics(new_score, init)
    savefig(samples, 'samples from posterior', os.path.join(log_path, 'samples_6'))
    np.savez(os.path.join(log_path, 'plld'), samples=s_t)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/home/gluo/spreco/config_exp/toy.yaml')
    args = parser.parse_args()
    main(args)


