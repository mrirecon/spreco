from spreco.model.refine_net_sde import cond_refine_net_plus
from spreco.common.custom_adam import AdamOptimizer
from spreco.common import utils

import tqdm
import tensorflow.compat.v1 as tf
import numpy as np

class sde():
    """
    class for generative modeling through stochastic diffusion equations (by Yang Song)
    only variance exploding sde is implemented hereafter.
    """

    def __init__(self, config): #sigma_min=0.01, sigma_max=50., eps=1e-5, N=1000):
        """
        eps, the smallest time step to sample from
        """
        if config['net'] == 'refine':
            self.net = cond_refine_net_plus(config)
        else:
            raise ValueError('Method %s not recognized.'%config['net'])

        self.config    = config
        self.sigma_min = config['sigma_min']
        self.sigma_max = config['sigma_max']
        self.N         = config['N']
        self.T         = 1.
        self.eps       = config['eps']
        self.sigmas    = tf.exp(tf.linspace(tf.log(self.sigma_min), tf.log(self.sigma_max), self.N))

        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        ###
        self.x       = [tf.placeholder(tf.float32, 
                               shape=[self.config['batch_size']]+self.config['input_shape'], name="input_%d"%i
                               ) for i in range(self.config['nr_gpu'])]
        self.t            = [tf.placeholder(tf.float32,
                               shape=[self.config['batch_size']]) for _ in range(config['nr_gpu'])]
        self.ins_outs     = {'inputs': self.x, 't': self.t}
        self.seed         = config['seed']


    def sde(self, x, t):

        sigma     = self.sigma_min * (self.sigma_max / self.sigma_min)**t
        drift     = tf.zeros_like(x)

        diffusion = sigma * tf.sqrt(2 * (tf.log(self.sigma_max) - tf.log(self.sigma_min)))

        return drift, diffusion

    def marginal_prob(self, x, t):
        """
        sigma(t)
        """
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        return x, std

    def prior_sampling(self, shape):
        """
        x(T) ~ N(0, sigma_max)
        """
        return tf.random.normal(shape, seed=self.seed) * self.sigma_max

    def discretize(self, x, t):

        timestep  = tf.cast(t * (self.N - 1) / self.T, tf.int32)
        sigma     = tf.gather(self.sigmas, timestep)
        bools     = tf.equal(timestep, tf.zeros_like(t, tf.int32))
        adj_sigma = tf.where(bools, tf.zeros_like(timestep, tf.float32), tf.gather(self.sigmas, tf.cast(timestep-1, tf.int32)))

        f         = tf.zeros_like(x)
        g         = tf.sqrt(sigma ** 2 - adj_sigma ** 2)

        return f, g

    def reverse_sde(self, x, t):
        """
        create the drift and diffusion functions for the reverse SDE/ODE
        """
        drift, diffusion = self.sde(x, t)
        score = self.score(x, t)
        drift = drift - diffusion ** 2 * score

        return drift, diffusion

    def reverse_discrete(self, x, t):
        f, G  = self.discretize(x, t)
        rev_f = f - G ** 2 * self.score(x, t)
        rev_g = G 
        return rev_f, rev_g

    def score(self, x, t, continuous = True):

        if continuous:
            labels = self.marginal_prob(tf.zeros_like(x), t)[1]
        else:
            labels = self.T - t
            labels = labels*(sde.N - 1)
            labels = np.round(labels)
        
        score = self.net.forward(x, labels)

        return score

    def loss(self, x, t, continuous=True, likelihood_weighting=False):
        """
        """
        shape = x.shape        
        z = tf.random.normal(shape, seed=self.seed)

        mean, std    = self.marginal_prob(x, t)
        std          = std[:, tf.newaxis, tf.newaxis, tf.newaxis]
        perturbed_x  = mean +  std * z
        score        = self.score(perturbed_x, t, continuous)

        if not likelihood_weighting:

            tmp       = tf.reshape(tf.math.square(score * std + z), (shape[0], -1))
            if self.config['reduce_mean']:
                loss      = tf.reduce_mean(tmp, axis=-1)
            else:
                loss      = tf.reduce_sum(tmp, axis=-1)

        else:

            g2   = self.sde(tf.zeros_like(x), t)[1] ** 2
            if self.config['reduce_mean']:
                loss = tf.reduce_mean(tf.math.square(score, z * 1. / std), axis=-1)
            else:
                loss = tf.reduce_sum(tf.math.square(score, z * 1. / std), axis=-1)
            loss = loss * g2

        loss = tf.reduce_mean(loss)

        return loss
    
    def prep(self, export =True):
        """create training objective"""

        if not export:
            _          = self.loss(self.x[0], self.t[0])
            all_params = tf.trainable_variables()

            loss      = []
            grads     = []
            loss_test = []

            optimizer = AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999)

            for i in range(self.config['nr_gpu']):
                with tf.device('/gpu:%d'%i):
                    # train
                    loss.append(self.loss(self.x[i], self.t[i]))

                    gvs = optimizer.compute_gradients(loss[-1], all_params)
                    gvs = [(k, v) for (k, v) in gvs if k is not None]
                    grads.append(gvs)

                    # test
                    loss_test.append(self.loss(self.x[i], self.t[i]))
                
            with tf.device('/gpu:0'):
                for i in range(1, self.config['nr_gpu']):
                    loss[0] += loss[i]
                    loss_test[0] += loss_test[i]
                
            grads_avg = optimizer.average_gradients(grads)

            self.train_op = optimizer.apply_gradients(grads_avg)
            self.loss_train = loss[0]/self.config['nr_gpu']
            self.loss_test  = loss_test[0]/self.config['nr_gpu']
        else:
            _          = self.loss(self.x[0], self.t[0])
            all_params = tf.trainable_variables()


class posterior_sampler():

    def __init__(self, sde, steps, target_snr, nr_samples, burn_in, burn_t, map_end=False, last_iteration=100, last_step_factor=1, disable_z=False):
        self.sde        = sde
        self.steps      = steps
        self.target_snr = target_snr
        self.nr_samples = nr_samples
        self.burn_in    = burn_in
        self.burn_t     = burn_t
        self.burn_flag  = True
        self.map_end    = map_end
        self.last_iteration = last_iteration
        self.last_step_factor = last_step_factor
        self.disable_z        = disable_z


    def conditional_ancestral_sampler(self, x, t, sess, AHA, AHy, s_stepsize, st=100):
      
        @tf.function
        def get_noise_1(x, t):
            shape = x.shape
            shape = [1] + shape[1:]
            
            timestep  = tf.cast(t * (self.sde.N - 1) / self.sde.T, tf.int32)
            sigma     = tf.gather(self.sde.sigmas, timestep)
            bools     = tf.equal(timestep, tf.zeros_like(t, tf.int32))
            adj_sigma = tf.where(bools, tf.zeros_like(timestep, tf.float32), tf.gather(self.sde.sigmas, tf.cast(timestep-1, tf.int32)))
            tau = (sigma ** 2 - adj_sigma ** 2)
            std = tf.sqrt(adj_sigma ** 2 * tau / (sigma ** 2))
            tau = tau[:, tf.newaxis, tf.newaxis, tf.newaxis]
            std = std[:, tf.newaxis, tf.newaxis, tf.newaxis]

            noise = tf.random.normal(shape, seed=self.sde.seed)*std
            return tau, std, noise
        
        @tf.function
        def get_noise_2(x, t):
            shape = x.shape
            shape = [self.nr_samples] + shape[1:]
            timestep  = tf.cast(t * (self.sde.N - 1) / self.sde.T, tf.int32)
            sigma     = tf.gather(self.sde.sigmas, timestep)
            bools     = tf.equal(timestep, tf.zeros_like(t, tf.int32))
            adj_sigma = tf.where(bools, tf.zeros_like(timestep, tf.float32), tf.gather(self.sde.sigmas, tf.cast(timestep-1, tf.int32)))
            tau = (sigma ** 2 - adj_sigma ** 2)
            std = tf.sqrt(adj_sigma ** 2 * tau / (sigma ** 2))
            tau = tau[:, tf.newaxis, tf.newaxis, tf.newaxis]
            std = std[:, tf.newaxis, tf.newaxis, tf.newaxis]

            noise = tf.random.normal(shape, seed=self.sde.seed)*std
            return tau, std, noise


        score_op = self.sde.score(x, t)
        get_noise_op_1 = get_noise_1(x, t)
        get_noise_op_2 = get_noise_2(x, t)


        x_val     = np.random.rand(*(utils.cplx2float(AHy).shape))
        t_vals    = np.linspace(self.sde.T, self.sde.eps, self.sde.N)

        xs      = []


        if not self.burn_in:
            cur_samples = self.nr_samples
            x_val = np.concatenate([x_val]*cur_samples)
        else:
            cur_samples=1

        for t_i in tqdm.tqdm(t_vals[st:-1]):
            for _ in range(self.steps):

                if self.burn_in: # just one time
                    if t_i < self.burn_t and self.burn_flag:
                        print("burned")
                        cur_samples = self.nr_samples
                        x_val = np.concatenate([x_val]*cur_samples)
                        self.burn_flag = False

                if self.burn_in and t_i > self.burn_t:
                    tau, std, noise = sess.run(get_noise_op_1, feed_dict={x: x_val, t: [t_i]*cur_samples})
                else:
                    tau, std, noise = sess.run(get_noise_op_2, feed_dict={x: x_val, t: [t_i]*cur_samples})

                score =  self.target_snr*sess.run(score_op,  {x: x_val, t: [t_i]*cur_samples})
                
                float_shape = x_val.shape
                _, sigma = self.sde.marginal_prob(AHy, t_i)

                if not self.disable_z:
                    z = utils.float2cplx(sess.run(tf.random.normal(float_shape, seed=self.sde.seed)*sigma))
                    noisy_x = utils.cplx2float(AHy + np.squeeze(AHA(z)))
                else:
                    noisy_x = utils.cplx2float(AHy)

                grad_data_fidelity = AHA(utils.float2cplx(x_val))
                grad_data_fidelity = utils.cplx2float(grad_data_fidelity)

                x_val = x_val + tau*score - std*s_stepsize*grad_data_fidelity + s_stepsize*std*noisy_x + noise

                xs.append(x_val)

        if self.map_end:
            t_i = t_vals[-1]
            for _ in range(self.last_iteration):

                if self.burn_in and t_i > self.burn_t:
                    tau, _, noise = sess.run(get_noise_op_1, feed_dict={x: x_val, t: [t_i]*cur_samples})
                else:
                    tau, _, noise = sess.run(get_noise_op_2, feed_dict={x: x_val, t: [t_i]*cur_samples})

                score =  self.target_snr*sess.run(score_op,  {x: x_val, t: [t_i]*cur_samples})
                
                float_shape = x_val.shape
                _, sigma = self.sde.marginal_prob(AHy, t_i)
                z = utils.float2cplx(sess.run(tf.random.normal(float_shape, seed=self.sde.seed)*sigma))
                noisy_x = utils.cplx2float(AHy + np.squeeze(AHA(z)))

                grad_data_fidelity = AHA(utils.float2cplx(x_val+tau*score))
                grad_data_fidelity = utils.cplx2float(grad_data_fidelity)

                x_val = x_val + tau*score - std*s_stepsize*grad_data_fidelity*self.last_step_factor + s_stepsize*std*noisy_x*self.last_step_factor + noise

                xs.append(x_val)

        return xs