from spreco.model import nn
from spreco.model.refine_net import cond_refine_net
from spreco.common.custom_adam import AdamOptimizer

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

class ncsn():
    """
    noise conditional score networks
    Generative modeling by estimating gradients of the data distribution arXiv: 1907.05600
    """
    def __init__(self, config):

        self.config       = config
        self.net          = cond_refine_net(config)
        self.begin_sigma  = config['begin_sigma']
        self.end_sigma    = config['end_sigma']
        self.nr_levels    = config['nr_levels']
        self.anneal_power = config['anneal_power']
        self.sigmas       = tf.exp(tf.linspace(tf.log(self.begin_sigma), tf.log(self.end_sigma), self.nr_levels))
        self.seed         = config['seed']
    

    def init_placeholder(self, mode=0):

        if mode == 0:
            # train
            self.learning_rate  = tf.placeholder(tf.float32, shape=[])
            self.x              = [tf.placeholder(tf.float32, 
                               shape=[self.config['batch_size']]+self.config['input_shape'], name="input_%d"%i
                               ) for i in range(self.config['nr_gpu'])]
            self.t            = [tf.placeholder(tf.int32,
                               shape=[self.config['batch_size']]) for _ in range(self.config['nr_gpu'])]
            self.ins_outs     = {'inputs': self.x, 't': self.t}

        elif mode == 1:
            # inference
            self.x = tf.placeholder(tf.float32, shape=[self.config['batch_size']]+self.config['input_shape'])
            self.t = tf.placeholder(tf.int32, shape=[self.config['batch_size']])

        elif mode == 2:
            # export
            self.x = tf.placeholder(tf.float32, shape=[self.config['batch_size']]+self.config['input_shape'], name="input_0")
            self.t = tf.placeholder(tf.int32, shape=[self.config['batch_size']], name="input_1")

    def denoise_score_matching(self, x):

        shape     = nn.int_shape(x)
        perturb_x = x + tf.random_normal(shape, mean=0.0, stddev=self.begin_sigma, seed=self.seed)
        target    = -1/(self.begin_sigma**2) * (perturb_x - x)
        score     = self.net.forward(perturb_x)
        tmp       = tf.reshape(score-target, (shape[0], -1))

        loss      = 1/2 * tf.reduce_mean(tf.reduce_sum(tmp**2, axis=-1))
        return loss
    
    def anneal_denoise_score_matching(self, x, h):
        
        shape     = nn.int_shape(x)
        sigs      = tf.gather(self.sigmas, h)[:, tf.newaxis, tf.newaxis, tf.newaxis]
        perturb_x = x + tf.random_normal(shape, seed=self.seed)*sigs
        target    = -1./(sigs**2) * (perturb_x - x)
        score     = self.net.forward(perturb_x, h)
        tmp       = tf.reshape(score-target, (shape[0], -1))

        loss      = tf.reduce_mean(1/2. * (tf.reduce_sum(tmp**2, axis=-1))*tf.squeeze(sigs)**self.anneal_power)
        return loss
    
    def anneal_denoise_score_matching_v2(self, x, h):
        """
        condition term h is not required for the network
        """
        shape     = nn.int_shape(x)
        sigs      = tf.gather(self.sigmas, h)[:, tf.newaxis, tf.newaxis, tf.newaxis]
        perturb_x = x + tf.random_normal(shape, seed=self.seed)*sigs
        target    = -1./(sigs**2) * (perturb_x - x)
        score     = self.net.forward(perturb_x, [0]*shape[0])
        score     = score / sigs # Parameterize the NCSN with s(x, σ) = s(x)/σ 
        tmp       = tf.reshape(score-target, (shape[0], -1))

        loss      = tf.reduce_mean(1/2. * (tf.reduce_sum(tmp**2, axis=-1))*tf.squeeze(sigs)**self.anneal_power)
        return loss
    
    def init(self, mode=0):
        
        if mode == 0:

            self.init_placeholder(mode)
            _ = self.anneal_denoise_score_matching(self.x[0], self.t[0])
            all_params = tf.trainable_variables()

            loss      = []
            grads     = []
            loss_test = []

            optimizer = AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999)

            for i in range(self.config['nr_gpu']):
                with tf.device('/gpu:%d'%i):
                    # train
                    loss.append(self.anneal_denoise_score_matching(self.x[i], self.t[i]))

                    gvs = optimizer.compute_gradients(loss[-1], all_params)
                    gvs = [(k, v) for (k, v) in gvs if k is not None]
                    grads.append(gvs)

                    # test
                    loss_test.append(self.anneal_denoise_score_matching(self.x[i], self.t[i]))
            
            with tf.device('/gpu:0'):
                for i in range(1, self.config['nr_gpu']):
                    loss[0] += loss[i]
                    loss_test[0] += loss_test[i]
            
            grads_avg = optimizer.average_gradients(grads)

            self.train_op = optimizer.apply_gradients(grads_avg)
            self.loss_train = loss[0]/self.config['nr_gpu']
            self.loss_test = loss_test[0]/self.config['nr_gpu']

        elif mode == 1:
            self.init_placeholder(mode)
            _ = self.anneal_denoise_score_matching(self.x, self.t)

        elif mode == 2:
            self.init_placeholder(mode)
            self.output = tf.squeeze(self.net.forward(self.x, self.t), name='output_0')

        else:
            raise ValueError("Value for mode selection is wrong, only 0,1,2 are valid.")
