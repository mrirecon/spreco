import tensorflow.compat.v1 as tf
import numpy as np
import tqdm

class sampler():

    def __init__(self, sde, steps, target_snr):
        self.sde        = sde
        self.steps      = steps
        self.target_snr = target_snr
    
    def predictor(self, x, t):
        """
        reverse diffusion
        """
        f, G    = self.sde.reverse_sde(x, t)
        z       = tf.random.normal(x.shape)
        x_mean  = x - f 
        x       = x_mean + G * z
        return [x, x_mean]


    def corrector(self, x, t):
        """
        langevin corrector
        """
        
        for i in range(self.steps):
            shape      = x.shape
            grad       = self.sde.score(x, t)
            noise      = tf.random.normal(shape)
            grad_norm  = tf.reduce_mean(tf.norm(tf.reshape(grad, [shape[0], -1]), axis=-1))
            noise_norm = tf.reduce_mean(tf.norm(tf.reshape(noise, [shape[0], -1]), axis=-1))
            step_size  = (self.target_snr * noise_norm / grad_norm) ** 2 * 2 * 1
            x_mean     = x + step_size * grad
            x          = x_mean + noise * tf.sqrt(step_size * 2)

        return [x, x_mean]

    def pc_sampler(self, x, t, sess, skip=1):

        x_val     = sess.run(self.sde.prior_sampling(x.shape))
        t_vals    = np.linspace(self.sde.T, self.sde.eps, self.sde.N)

        corrector = self.corrector(x, t)
        predictor = self.predictor(x, t)
        xs        = []
        xs_mean   = []
        for t_i in tqdm.tqdm(t_vals[0::skip]):
            x_val, x_mean = sess.run(corrector, {x: x_val, t: [t_i]})
            x_val, x_mean = sess.run(predictor, {x: x_val, t: [t_i]})
            xs.append(x_val)
            xs_mean.append(x_mean)
        
        return xs, xs_mean


    def ancestral_sampler(self, x, t, sess):
        x_val     = sess.run(self.sde.prior_sampling(x.shape))
        t_vals    = np.linspace(self.sde.T, self.sde.eps, self.sde.N)

        xs      = []
        xs_mean = []

        def update(x, t):
            diffusion=self.sde.sde(x,t)[1]
            for _ in range(self.steps):
                x_mean    = x + self.sde.score(x, t)*diffusion**2
                std = diffusion#*self.sde.sigma_t(t-0.5/self.sde.N)/self.sde.sigma_t(t+0.5/self.sde.N)
                x = x_mean + tf.random.normal(x.shape, seed=self.sde.seed)*std
            return x, x_mean

        
        update_op = update(x, t)
        for t_i in tqdm.tqdm(t_vals):
            x_val, x_mean = sess.run(update_op, {x: x_val, t: [t_i]})
            xs.append(x_val)
            xs_mean.append(x_mean)

        return xs, xs_mean