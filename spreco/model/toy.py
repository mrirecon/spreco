import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import numpy as np


class net(Model):
    def __init__(self, config):
        super(net, self).__init__()
        self.d1 = Dense(config['num_units'], activation=config['nonlinearity'])
        self.d2 = Dense(config['num_units'], activation=config['nonlinearity'])
        self.d3 = Dense(2)

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return x


class toy():
    """
    the toy model to demonstrate how samples are simulated from posterior
    """
    def __init__(self, config):

        self.config       = config

        self.sigma_min     = config['sigma_min']
        self.sigma_max     = config['sigma_max']
        self.nr_levels     = config['nr_levels']
        self.sigmas        = tf.exp(tf.linspace(tf.math.log(self.sigma_max), self.sigma_min, self.nr_levels))
        self.seed          = config['seed']
        self.score_net     = net(config)


    @staticmethod
    def langevin_dynamics(score, x_k, lr=0.1, step=1000):
        samples_t=[]
        for i in range(step):
            samples_t.append(x_k)
            current_lr = lr
            x_k = x_k + current_lr / 2 * score(x_k)
            x_k = x_k + tf.random.normal(x_k.shape) * np.sqrt(current_lr)
        samples_t.append(x_k)
        return x_k, samples_t

    @staticmethod
    def anneal_langevin_dynamics(score, x_k, sigmas, lr=0.1, n_steps_each=100):
        samples_t=[]
        for sigma in sigmas:
            for i in range(n_steps_each):
                samples_t.append(x_k)
                current_lr = lr * (sigma / sigmas[-1]) ** 2
                x_k = x_k + current_lr / 2 * score(x_k, sigma)
                x_k = x_k + tf.random.normal(x_k.shape) * np.sqrt(current_lr)
        samples_t.append(x_k)

        return x_k, samples_t

    def score_estimation(self, samples):

        vectors = tf.random.normal(samples.shape)

        with tf.GradientTape() as g:
            g.watch(samples)
            grad1 = self.score_net(samples)
            gradv = tf.reduce_sum(grad1 * vectors)
            grad2 = g.gradient(gradv, samples)

        grad1 = tf.reshape(grad1, (samples.shape[0], -1))

        loss1 = tf.reduce_sum(grad1 * grad1, axis=-1) / 2.
        loss2 = tf.reduce_sum(tf.reshape(vectors * grad2, (samples.shape[0], -1)), axis=-1)
        loss = loss1 + loss2

        return tf.reduce_mean(loss)
