from spreco.model.refine_net_sde import cond_refine_net_plus
from spreco.common.custom_adam import AdamOptimizer

import tensorflow.compat.v1 as tf


class sde():
    """
    class for a diffusion model formulated with stochastic diffusion equation
    Refs: arXiv 2006.11239 (Ho Jonathan); ICLR2021 (Song Yang)
    """

    def __init__(self, config): 

        self.config    = config
        self.sigma_min = config['sigma_min']
        self.sigma_max = config['sigma_max']
        self.N         = 100 if 'N' not in  config.keys() else config['N']
        self.T         = 1.
        self.eps       = 1.e-5

        if 'scale_out' in self.config.keys():
            self.net = cond_refine_net_plus(config, scale_out=self.config['scale_out'])
        else:
            self.net = cond_refine_net_plus(config)

        self.seed         = config['seed']

    def init_placeholder(self, mode=0, batch_size=None):

        if mode == 0:
            # training
            self.learning_rate  = tf.placeholder(tf.float32, shape=[])
            self.x              = [tf.placeholder(tf.float32, 
                                   shape=[self.config['batch_size']]+self.config['input_shape'], name="input_%d"%i
                                   ) for i in range(self.config['nr_gpu'])]
            self.t            = [tf.placeholder(tf.float32,
                                   shape=[self.config['batch_size']]) for _ in range(self.config['nr_gpu'])]
            self.ins_outs     = {'inputs': self.x, 't': self.t}

        if mode == 1:
            # inference
            self.x = tf.placeholder(tf.float32, shape=[batch_size]+self.config['input_shape'])
            self.t = tf.placeholder(tf.float32, shape=[batch_size])

        if mode == 2:
            # exporting
            self.x = tf.placeholder(tf.float32, shape=[batch_size]+self.config['input_shape'], name="input_0")
            self.t = tf.placeholder(tf.float32, shape=[batch_size], name="input_1")

    def prior_sampling(self, shape):
        """
        x(T) ~ N(0, sigma_max)
        """
        return tf.random.normal(shape, seed=self.seed) * self.sigma_max

    def sigma_t(self, t, typ='quad'):
        """
        noise schedule for t in (1, 0)
        """
        #
        if typ == "quad":
            sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * t**2
        elif typ == "exp":
            sigma = self.sigma_min * (self.sigma_max / self.sigma_min)**t
        elif typ == 'linear':
            sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * t
        else:
            raise TypeError("Check you the type of sigma_t!")
        return sigma
    
    def sigmas(self, typ='quad'):
        return self.sigma_t(tf.linspace(self.T, self.eps, self.N+1), typ)

    def sde(self, x, t, typ='quad'):

        drift     = tf.zeros_like(x)
        diffusion = tf.sqrt(tf.gradients(tf.math.square(self.sigma_t(t, typ)), t)[0]/self.N)

        return drift, diffusion[:, tf.newaxis, tf.newaxis, tf.newaxis]

    def reverse_sde(self, x, t, typ='quad'):
        
        drift, diffusion = self.sde(x, t, typ)
        score = self.score(x, t, typ)
        drift = drift - diffusion ** 2 * score

        return drift, diffusion

    def discretize(self, x, t, typ='quad'):

        timestep  = tf.cast((1. - t / self.T) * self.N, tf.int32)
        sigma     = tf.gather(self.sigmas(typ), timestep)
        adj_sigma = tf.gather(self.sigmas(typ), timestep + 1)

        f         = tf.zeros_like(x)
        g         = tf.sqrt(sigma ** 2 - adj_sigma ** 2)

        return f, g[:, tf.newaxis, tf.newaxis, tf.newaxis]

    def reverse_discrete(self, x, t, typ='quad'):
        f, G  = self.discretize(x, t, typ)
        rev_f = f - G ** 2 * self.score(x, t, typ)
        rev_g = G 
        return rev_f, rev_g

    def score(self, x_t, t, typ='quad'):
        return self.net.forward(x_t, self.sigma_t(t, typ))

    def loss(self, x, t, likelihood_weighting=False):
        """
        x is perturb image
        t is the sigma used to perturb image
        """

        z = tf.random.normal(tf.shape(x), seed=self.seed)

        std       = t[:, tf.newaxis, tf.newaxis, tf.newaxis]
        x_t       = x +  std * z
        score     = self.net.forward(x_t, t)

        reduce    = lambda tmp: tf.reduce_mean(tmp, axis=[1,2,3]) if self.config['reduce_mean'] else tf.reduce_sum(tmp, axis=[1,2,3])

        if not likelihood_weighting:
            loss = reduce(tf.math.square(score * std + z))
        else:
            loss = reduce(tf.math.square(score + z/std))
            loss = loss * std ** 2

        loss = tf.reduce_mean(loss)

        return loss
    
    def init(self, mode=0, batch_size=None):

        self.init_placeholder(mode, batch_size)

        if mode == 0:

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

        elif mode == 1:
            _  = self.loss(self.x, self.t)

        elif mode == 2:
            _  = self.loss(self.x, self.t)
            diffusion=self.sde(self.x, self.t, self.config['sigma_type'])[1]
            self.default_out = self.score(self.x, self.t, self.config['sigma_type']) * diffusion**2

        else:
            raise ValueError("Value for mode selection is wrong, only 0,1,2 are valid.")
