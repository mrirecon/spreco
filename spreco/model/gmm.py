import tensorflow as tf
import numpy as np

class Gaussian_Mixture():
    
    def __init__ (self, dim, mix_prob=[0.7, 0.2, 0.1]):

        self.mix_prob = mix_prob
        self.means    = tf.stack([5*tf.ones(dim), -5*tf.ones(dim), [5,-5]], axis=0)
        self.sigma    = 1.

    def sample(self, nr_samples, sigma=1.):
        mix_idx = tf.random.categorical(tf.math.log([self.mix_prob]), nr_samples)
        means   = tf.gather(self.means, mix_idx)
        return tf.random.normal(means.shape)*sigma + means

    def log_prob(self, samples, sigma=1.):
        logps = []
        for i in range(len(self.mix_prob)):
            logps.append((-tf.reduce_sum((samples - self.means[i]) ** 2, axis=-1) / (2 * sigma ** 2) - 0.5 * tf.math.log(
                2 * np.pi * sigma ** 2)) + tf.math.log(self.mix_prob[i]))
        logp = tf.reduce_logsumexp(tf.stack(logps, axis=0), axis=0)
        return logp

    def score(self, samples, sigma=1.):
        with tf.GradientTape() as g:
            g.watch(samples)
            log_probs = tf.reduce_sum(self.log_prob(samples, sigma))
            return g.gradient(log_probs, samples)

