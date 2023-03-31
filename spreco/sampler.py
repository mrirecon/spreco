from spreco.common.options import MODELS

import os
import tensorflow.compat.v1 as tf
import numpy as np
import tqdm

class sampler():
    """
    when using large skip, it doesn't work
    """

    def __init__(self, config, steps, target_snr, skip=1):
        self.config     = config
        self.steps      = steps
        self.target_snr = target_snr
        self.skip       = skip
    
    def predictor(self, x, t):
        """
        reverse diffusion
        """
        f, G    = self.model.reverse_sde(x, t)
        f       = self.skip * f
        G       = self.skip * G
        z       = tf.random.normal(tf.shape(x))
        x_mean  = x - f 
        x       = x_mean + G * z
        return [x, x_mean]

    def get_shape(self, samples):
        return [samples] + self.model.x.shape[1:]
    
    @staticmethod
    def norm(x, axis=(1,2,3)):
        return tf.sqrt(tf.reduce_sum(tf.square(x), axis))

    def corrector(self, x, t):
        """
        langevin corrector
        """
        for _ in range(self.steps):
            grad       = self.model.score(x, t)
            noise      = tf.random.normal(tf.shape(x))
            grad_norm  = tf.reduce_mean(self.norm(grad))
            noise_norm = tf.reduce_mean(self.norm(noise))
            step_size  = (self.target_snr * noise_norm / grad_norm) ** 2 * 2 * 1
            x_mean     = x + step_size * grad
            x          = x_mean + noise * tf.sqrt(step_size * 2)

        return [x, x_mean]

    def pc_sampler(self, nr_samples):

        x       = self.model.x
        t       = self.model.t

        x_val     = self.sess.run(self.model.prior_sampling(self.get_shape(nr_samples)))
        t_vals    = np.linspace(self.model.T, self.model.eps, self.model.N)

        corrector = self.corrector(x, t)
        predictor = self.predictor(x, t)
        xs        = []
        xs_mean   = []
        for t_i in tqdm.tqdm(t_vals[0::self.skip]):
            x_val, x_mean = self.sess.run(corrector, {x: x_val, t: [t_i for _ in range(nr_samples)]})
            x_val, x_mean = self.sess.run(predictor, {x: x_val, t: [t_i for _ in range(nr_samples)]})
            xs.append(x_val)
            xs_mean.append(x_mean)
        
        return xs, xs_mean


    def ancestral_sampler(self, nr_samples):

        x_val     = self.sess.run(self.model.prior_sampling(self.get_shape(nr_samples)))
        t_vals    = np.linspace(self.model.T, self.model.eps, self.model.N)

        xs      = []
        xs_mean = []

        def update(x, t):

            diffusion=self.model.sde(x,t)[1]*self.skip

            for _ in range(self.steps):
                x_mean = x + self.model.score(x, t)*diffusion**2
                ratio  = self.model.sigma_t(t - self.skip*0.5/self.model.N)/self.model.sigma_t(t + self.skip*0.5/self.model.N)
                std    = diffusion*ratio[:, tf.newaxis, tf.newaxis, tf.newaxis]
                x      = x_mean + tf.random.normal(tf.shape(x), seed=self.model.seed)*std
            return x, x_mean

        update_op = update(self.model.x, self.model.t)

        for t_i in tqdm.tqdm(t_vals[0::self.skip]):
            x_val, x_mean = self.sess.run(update_op, {self.model.x: x_val, self.model.t: [t_i for _ in range(nr_samples)]})
            xs.append(x_val)
            xs_mean.append(x_mean)

        return xs, xs_mean

    def init_model(self, model_path, gpu_id=None, seed=None):
        
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

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if gpu_id is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
        else: 
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

        saver      = tf.train.Saver()
        self.sess  = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        saver.restore(self.sess, model_path)