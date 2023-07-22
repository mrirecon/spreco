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
        self.sigma_max = config['sigma_max']
        self.sigma_min = config['sigma_min']

        self.seed     = config['seed']
        self.N        = 200 if 'N' not in  config.keys() else config['N']
        self.T        = 1.
        self.eps      = 1.e-5

        self.net        = cond_refine_net_plus(config, chns=config['data_chns'], scale_out=False)
        self.type       = 'DDPM'
        self.continuous = True if 'continuous' not in config.keys() else config['continuous'] # continuous sde take longer to be well-trained

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

    def sigma_t(self, t):
        return self.sigma_min + t * (self.sigma_max - self.sigma_min)

    def sigmas(self):
        return self.sigma_t(tf.linspace(self.eps, self.T, self.N))/self.N
    
    def alphas(self):
        return 1. - self.sigmas()
    
    def sqrt_alpha_cumprod(self):
        return tf.sqrt(tf.cumprod(self.alphas(), axis=0))

    def sqrt_1m_alphas_cumprod(self):
        return tf.sqrt(1. - tf.cumprod(self.alphas(), axis=0))

    def sde(self, x, t, typ=None):
        sigma = self.sigma_t(t)[:, None, None, None]
        drift = - 0.5 * sigma * x / self.N
        diffusion = tf.sqrt(sigma/self.N)
        return drift, diffusion

    def reverse_sde(self, x, t, typ=None, ode=False):
        drift, diffusion = self.sde(x, t, typ)
        
        if ode:
            score = 0.5*self.score(x, t, typ)
        else:
            score = self.score(x, t, typ)
        drift = drift - diffusion ** 2 * score
        return drift, diffusion

    def discretize(self, x, t):
        timestep  = tf.cast((t / self.T) * (self.N-1), tf.int32)
        sigma      = tf.gather(self.sigmas(), timestep)[:, tf.newaxis, tf.newaxis, tf.newaxis]
        alpha     = tf.gather(self.alphas(), timestep)[:, tf.newaxis, tf.newaxis, tf.newaxis]

        f         = tf.sqrt(alpha)*x - x
        g         = tf.sqrt(sigma)
        return f, g
    
    def reverse_discrete(self, x, t, typ=None, ode=False):
        f, G  = self.discretize(x, t)

        if ode:
            score = 0.5 * self.score(x, t)
        else:
            score = self.score(x, t)

        return f - G ** 2 * score, G

    def score(self, x_t, t, typ=None):

        if self.continuous:
            log_mean_coeff = -0.25 * t ** 2 * (self.sigma_max - self.sigma_min) - 0.5 * t * self.sigma_min
            std = tf.sqrt(-tf.math.expm1(2. * log_mean_coeff))
        else:
            timestep  = tf.cast(( t / self.T) * (self.N-1), tf.int32)
            std = tf.gather(self.sqrt_1m_alphas_cumprod(), timestep)
        return -self.net.forward(x_t, std)/std[:, tf.newaxis, tf.newaxis, tf.newaxis]


    def prior_sampling(self, shape, seed=None):
        """
        x(T) ~ N(0, 1)
        """
        return tf.random.normal(shape, seed=seed) 

    def marginal_prob(self, x, t):
        if self.continuous:

            log_mean_coeff = -0.25 * t ** 2 * (self.sigma_max - self.sigma_min) - 0.5 * t * self.sigma_min
            mean = tf.exp(log_mean_coeff[:, tf.newaxis, tf.newaxis, tf.newaxis])
            std = tf.sqrt(-tf.math.expm1(2. * log_mean_coeff))

        else:

            timestep  = tf.cast((t / self.T) * (self.N-1), tf.int32)
            mean = tf.gather(self.sqrt_alpha_cumprod(), timestep)[:, tf.newaxis, tf.newaxis, tf.newaxis]
            std = tf.gather(self.sqrt_1m_alphas_cumprod(), timestep)

        return mean, std

    def loss(self, x, t, weighting=False):
        """
        """
        z = tf.random.normal(tf.shape(x))

        mean, std = self.marginal_prob(x, t)

        x_t  = mean*x + std[:, tf.newaxis, tf.newaxis, tf.newaxis]*z

        z_theta = -self.net.forward(x_t, std)

        reduce    = lambda tmp: tf.reduce_mean(tmp, axis=[1,2,3]) if self.config['reduce_mean'] else tf.reduce_sum(tmp, axis=[1,2,3])
        if weighting:
            w = tf.sigmoid(mean/(std[:, tf.newaxis, tf.newaxis, tf.newaxis]**2))
        l = reduce(tf.math.square(z_theta + z) if not weighting else w * tf.math.square(z_theta + z))

        l = tf.reduce_mean(l)

        return l

    def init(self, mode=0, batch_size=None, **kwargs):

        self.init_placeholder(mode, batch_size)

        if mode == 0:

            _          = self.loss(self.x[0], self.t[0], weighting= False if 'loss_weight' not in self.config.keys() else self.config['loss_weight'])
            all_params = tf.trainable_variables()

            loss      = []
            grads     = []
            loss_test = []

            optimizer = AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999)

            for i in range(self.config['nr_gpu']):
                with tf.device('/gpu:%d'%i):
                    # train
                    loss.append(self.loss(self.x[i], self.t[i], weighting= False if 'loss_weight' not in self.config.keys() else self.config['loss_weight']))

                    gvs = optimizer.compute_gradients(loss[-1], all_params)
                    gvs = [(k, v) for (k, v) in gvs if k is not None]
                    grads.append(gvs)

                    # test
                    loss_test.append(self.loss(self.x[i], self.t[i], weighting= False if 'loss_weight' not in self.config.keys() else self.config['loss_weight']))

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
            _  = self.net.forward(self.x, self.t)

        elif mode == 2:

            self.net.dropout = 0.0

            if 'default_out' in kwargs.keys() and kwargs['default_out'] == False:
                print("INFO -> Customizing tf inputs and outputs")

            else:
                self.x = tf.placeholder(tf.float32, shape=[batch_size]+self.config['input_shape'], name="input_0")
                self.t = tf.placeholder(tf.float32, shape=[batch_size], name="input_1")
                diffusion=self.sde(self.x, self.t)[1]
                self.default_out = self.score(self.x, self.t) * diffusion**2
