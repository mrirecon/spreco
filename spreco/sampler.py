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
        f, G    = self.model.reverse_sde(x, t, self.sigma_type)
        f       = f
        G       = G
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

    def an_update(self, x, t):
        diffusion=self.model.sde(x,t, self.sigma_type)[1]
        x_mean = x + self.model.score(x, t, self.sigma_type)*diffusion**2
        ratio  = self.model.sigma_t(t - 0.5/self.model.N, self.sigma_type)/self.model.sigma_t(t + 0.5/self.model.N, self.sigma_type)
        std    = diffusion*ratio[:, tf.newaxis, tf.newaxis, tf.newaxis]
        x      = x_mean + tf.random.normal(tf.shape(x), seed=self.model.seed)*std
        return x, x_mean

    def init_ops(self):
        self.corrector_op = self.corrector(self.model.x, self.model.t)
        self.predictor_op = self.predictor(self.model.x, self.model.t)
        self.an_update_op = self.an_update(self.model.x, self.model.t)
        self.sig_op = self.model.sde(self.model.x, self.model.t, self.sigma_type)[1]

    def get_shape(self, samples):
        return [samples] + self.model.x.shape[1:]
    
    def pc_sampler(self, nr_samples, steps):

        t_vals    = np.linspace(self.model.T, self.model.eps, self.model.N)
        x_val     = self.sess.run(self.model.prior_sampling(self.get_shape(nr_samples)))


        xs        = []
        xs_mean   = []
        for t_i in tqdm.tqdm(t_vals):
            x_val, x_mean = self.sess.run(self.corrector_op, {self.model.x: x_val, self.model.t: [t_i for _ in range(nr_samples)]})
            for _ in range(steps):
                x_val, x_mean = self.sess.run(self.predictor_op, {self.model.x: x_val, self.model.t: [t_i for _ in range(nr_samples)]})

            if self.cond_func is not None:
                sig = self.sess.run(self.sig_op, {self.model.t: [t_i]})

                x_val = self.cond_func(x_val, sig)
                x_mean = self.cond_func(x_mean, sig)

            xs.append(x_val)
            xs_mean.append(x_mean)
        
        return xs, xs_mean
    
    def ancestral_sampler(self, nr_samples, steps):

        t_vals    = np.linspace(self.model.T, self.model.eps, self.model.N)
        x_val     = self.sess.run(self.model.prior_sampling(self.get_shape(nr_samples)))

        xs      = []
        xs_mean = []


        for t_i in tqdm.tqdm(t_vals):
            for _ in range(steps):
                x_val, x_mean = self.sess.run(self.an_update_op, {self.model.x: x_val, self.model.t: [t_i for _ in range(nr_samples)]})
            
                if self.cond_func is not None:
                    sig = self.sess.run(self.sig_op, {self.model.t: [t_i]})
                    x_val = self.cond_func(x_val, sig)

            xs.append(x_val)
            xs_mean.append(x_mean)

        return xs, xs_mean

    def init_sampler(self, model_path, gpu_id=None, seed=None):
        
        if seed is not None:
            tf.random.set_random_seed(seed)
        else:
            tf.random.set_random_seed(self.config['seed'])

        if self.config['model'] == MODELS.NCSN:
            from spreco.model.ncsn import ncsn as selected_class

        elif self.config['model'] == MODELS.SDE:
            from spreco.model.sde import sde as selected_class

        elif self.config['model'] == MODELS.PIXELCNN:
            from spreco.model.pixelcnn import pixelcnn as selected_class

        else:
            raise Exception("Currently, this model is not implemented!")

        self.model = selected_class(self.config)

        self.model.init(mode=1, batch_size=None)
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