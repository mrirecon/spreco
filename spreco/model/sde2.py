from spreco.model.unet import unet
from spreco.model.refine_net_sde import cond_refine_net_plus
from spreco.common.custom_adam import AdamOptimizer

import tensorflow.compat.v1 as tf


class sde():
    """
    Variance preserved difussion
    Refs: arXiv 2006.11239 (Ho Jonathan) --> DDPM;
    """

    def __init__(self, config):

        self.config   = config
        self.beta_max = config['beta_max']
        self.beta_min = config['beta_min']

        self.seed     = config['seed']
        self.N        = 1000 if 'N' not in  config.keys() else config['N']
        self.T        = 1.
        self.eps      = 1.e-5

        if config['net'] == 'refine':
            self.net      = cond_refine_net_plus(config, chns=config['data_chns'], scale_out=False)
        else:
            self.net      = unet(config, chns=config['data_chns'])
            
        self.type     = 'DDPM'
    
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

    def beta_t(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def sde(self, x, t, typ=None):
        beta = self.beta_t(t)[:, None, None, None]
        drift = - 0.5 * beta * x / self.N
        diffusion = tf.sqrt(beta/self.N)
        return drift, diffusion

    def reverse_sde(self, x, t, typ=None, ode=False):
        drift, diffusion = self.sde(x, t, typ)
        
        if ode:
            score = 0.5*self.score(x, t, typ)
        else:
            score = self.score(x, t, typ)
        drift = drift - diffusion ** 2 * score
        return drift, diffusion

    def score(self, x_t, t, typ=None):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        std = tf.sqrt(-tf.math.expm1(2. * log_mean_coeff))
        return self.net.forward(x_t, t)


    def prior_sampling(self, shape, seed=None):
        """
        x(T) ~ N(0, 1)
        """
        return tf.random.normal(shape, seed=seed) 

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean = tf.exp(log_mean_coeff[:, tf.newaxis, tf.newaxis, tf.newaxis]) * x
        std = tf.sqrt(-tf.math.expm1(2. * log_mean_coeff))
        return mean, std

    def loss(self, x, t):
        """
        """
        z = tf.random.normal(tf.shape(x))

        mean, std = self.marginal_prob(x, t)

        x_t  = mean + std[:, tf.newaxis, tf.newaxis, tf.newaxis]*z

        score = self.net.forward(x_t, t)

        reduce    = lambda tmp: tf.reduce_mean(tmp, axis=[1,2,3]) if self.config['reduce_mean'] else tf.reduce_sum(tmp, axis=[1,2,3])

        l = reduce(tf.math.square(score*std[:, tf.newaxis, tf.newaxis, tf.newaxis] + z))

        l = tf.reduce_mean(l)

        return l

    def init(self, mode=0, batch_size=None, **kwargs):

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
            self.net.dropout = 0.0
            _  = self.loss(self.x, self.t)

        elif mode == 2:

            self.net.dropout = 0.0

            if 'sigma_type' in kwargs.keys() and kwargs['sigma_type'] is not None:
                self.config['sigma_type'] = kwargs['sigma_type']
            else:
                self.config['sigma_type'] = 'linear'

            print('INFO -> Exporting diffusion model with %s noise schedule'%self.config['sigma_type'])

            if 'default_out' in kwargs.keys() and kwargs['default_out'] == False:
                print("INFO -> Customizing tf inputs and outputs")
                
            else:
                self.x = tf.placeholder(tf.float32, shape=[batch_size]+self.config['input_shape'], name="input_0")
                self.t = tf.placeholder(tf.float32, shape=[batch_size], name="input_1")
                diffusion=self.sde(self.x, self.t, self.config['sigma_type'])[1]
                self.default_out = self.score(self.x, self.t, self.config['sigma_type']) * diffusion**2
                _  = self.loss(self.x, self.t)
