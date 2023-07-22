from spreco.common.options import MODELS

import os
import tensorflow.compat.v1 as tf
import numpy as np
import tqdm

class sampler():
    """
    sigma_type is for noise schedule
    N is the number of disrecte steps
    [config['sigma_min'], config['simga_max']] is the range for the score network can handle
    """

    def __init__(self, config, target_snr, sigma_type='exp', cond_func=None):
        """
        args for prepare network and computation graph
        """
        self.config      = config
        self.target_snr  = target_snr
        self.sigma_type  = sigma_type
        self.cond_func   = cond_func
    
    def predictor(self, x, t):
        """
        reverse diffusion
        """
        f, G    = self.reverse(x, t, self.sigma_type)
        z       = tf.random.normal(tf.shape(x))
        x_mean  = x - f 
        x       = x_mean + G * z
        return [x, x_mean]


    @staticmethod
    def norm(x, axis=(1,2,3)):
        return tf.sqrt(tf.reduce_sum(tf.square(x), axis, keepdims=True))

    def corrector(self, x, t):
        """
        langevin corrector
        """
        grad       = self.model.score(x, t, self.sigma_type)
        noise      = tf.random.normal(tf.shape(x))
        grad_norm  = self.norm(grad)
        noise_norm = self.norm(noise)
        step_size  = (self.target_snr * noise_norm / grad_norm) ** 2 * 2 * 1
        x_mean     = x + step_size * grad
        x          = x_mean + noise * tf.sqrt(step_size * 2)

        return [x, x_mean]

    def euler_update(self, x, t):
        drift, diffusion=self.reverse(x, t, self.sigma_type, ode=True)
        x_mean = x - drift
        x      = x_mean + tf.random.normal(tf.shape(x), seed=self.model.seed)*diffusion
        return x, x_mean
    
    def an_update(self, x, t):
        if self.model.type == 'DDPM':
            timestep  = tf.cast(t  * (self.model.N-1), tf.int32)
            sigma = tf.gather(self.model.sigmas(), timestep)
            score = self.model.score(x, t)
            x_mean = (x + sigma[:, None, None, None] * score) / tf.sqrt(1. - sigma)[:, None, None, None]
            x = x_mean + tf.sqrt(sigma)[:, None, None, None] * tf.random.normal(tf.shape(x), seed=self.model.seed)
        else:
            drift, diffusion=self.reverse(x, t, self.sigma_type)
            x_mean = x - drift
            x      = x_mean + tf.random.normal(tf.shape(x), seed=self.model.seed)*diffusion
        return x, x_mean
    
    def ode_update(self, x, t):
        
        drift, _ = self.reverse(x, t, self.sigma_type, True)
        x = x - drift

        return x

    def init_ops(self):
        self.corrector_op = self.corrector(self.model.x, self.model.t)
        self.predictor_op = self.predictor(self.model.x, self.model.t)
        self.euler_update_op = self.euler_update(self.model.x, self.model.t)
        self.an_update_op = self.an_update(self.model.x, self.model.t)
        self.sig_op = self.model.sde(self.model.x, self.model.t, self.sigma_type)[1]
        self.ode_update_op = self.ode_update(self.model.x, self.model.t)

    def get_shape(self, samples):
        return [samples] + self.model.x.shape[1:]
    
    def pc_sampler(self, nr_samples, innnersteps):

        t_vals    = np.linspace(self.model.T, self.model.eps, self.model.N)
        x_val     = self.sess.run(self.model.prior_sampling(self.get_shape(nr_samples)))


        xs        = []
        xs_mean   = []
        for t_i in tqdm.tqdm(t_vals):
            x_val, x_mean = self.sess.run(self.corrector_op, {self.model.x: x_val, self.model.t: [t_i for _ in range(nr_samples)]})
            for _ in range(innnersteps):
                x_val, x_mean = self.sess.run(self.predictor_op, {self.model.x: x_val, self.model.t: [t_i for _ in range(nr_samples)]})

            if self.cond_func is not None:
                sig = self.sess.run(self.sig_op, {self.model.t: [t_i]})

                x_val = self.cond_func(x_val, sig)
                x_mean = self.cond_func(x_mean, sig)

            xs.append(x_val)
            xs_mean.append(x_mean)
        
        return xs, xs_mean
    
    def euler_sampler(self, nr_samples, innnersteps=1):

        t_vals    = np.linspace(self.model.T, self.model.eps, self.model.N)
        x_val     = self.sess.run(self.model.prior_sampling(self.get_shape(nr_samples)))

        xs      = []
        xs_mean = []


        for t_i in tqdm.tqdm(t_vals):
            for _ in range(innnersteps):
                x_val, x_mean = self.sess.run(self.euler_update_op, {self.model.x: x_val, self.model.t: [t_i for _ in range(nr_samples)]})
            
                if self.cond_func is not None:
                    sig = self.sess.run(self.sig_op, {self.model.t: [t_i]})
                    x_val = self.cond_func(x_val, sig)

            xs.append(x_val)
            xs_mean.append(x_mean)

        return xs, xs_mean
    
    def ancestral_sampler(self, nr_samples, innnersteps):

        t_vals    = np.linspace(self.model.T, self.model.eps, self.model.N)
        x_val     = self.sess.run(self.model.prior_sampling(self.get_shape(nr_samples)))

        xs      = []
        xs_mean = []


        for t_i in tqdm.tqdm(t_vals):
            for _ in range(innnersteps):
                x_val, x_mean = self.sess.run(self.an_update_op, {self.model.x: x_val, self.model.t: [t_i for _ in range(nr_samples)]})
            
                if self.cond_func is not None:
                    sig = self.sess.run(self.sig_op, {self.model.t: [t_i]})
                    x_val = self.cond_func(x_val, sig)

            xs.append(x_val)
            xs_mean.append(x_mean)

        return xs, xs_mean
    
    def ode_solver(self, nr_samples):
        t_vals    = np.linspace(self.model.T, self.model.eps, self.model.N)
        x_val     = self.sess.run(self.model.prior_sampling(self.get_shape(nr_samples)))
        xs = []

        for t_i in tqdm.tqdm(t_vals):
            
            x_val = self.sess.run(self.ode_update_op, {self.model.x: x_val, self.model.t: [t_i for _ in range(nr_samples)]})
            xs.append(x_val)
        
        return xs

    def init_sampler(self, model_path, gpu_id=None, seed=None):
        
        if seed is not None:
            tf.random.set_random_seed(seed)
        else:
            tf.random.set_random_seed(self.config['seed'])

        if self.config['model'] == MODELS.NCSN:
            from spreco.model.ncsn import ncsn as selected_class

        elif self.config['model'] == MODELS.SDE:
            from spreco.model.sde import sde as selected_class
        
        elif self.config['model'] == MODELS.SDE2:
            from spreco.model.sde2 import sde as selected_class

        elif self.config['model'] == MODELS.PIXELCNN:
            from spreco.model.pixelcnn import pixelcnn as selected_class

        else:
            raise Exception("Currently, this model is not implemented!")

        self.model = selected_class(self.config)

        self.model.init(mode=1, batch_size=None)

        if self.model.continuous:
            self.reverse = self.model.reverse_sde
        else:
            self.reverse = self.model.reverse_discrete

        self.init_ops()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if gpu_id is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
        else: 
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

        saver      = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        saver.restore(self.sess, model_path)