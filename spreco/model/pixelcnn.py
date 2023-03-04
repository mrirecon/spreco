from spreco.model import nn
from spreco.model.utils import concat_elu
from spreco.common.custom_adam import AdamOptimizer
from spreco.common.options import DATA_CHNS

import numpy as np
from tf_slim import arg_scope
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

class pixelcnn():
    
    def __init__(self, config):
        self.config         = config
        self.forward        = tf.make_template('forward', self.body)

    def init_placeholder(self, mode=0):

        if mode == 0:
            # training
            self.learning_rate = tf.placeholder(tf.float32, shape=[])
            self.x  = [tf.placeholder(tf.float32, 
                               shape=[self.config['batch_size']]+self.config['input_shape'], name="input_%d"%i
                               ) for i in range(self.config['nr_gpu'])]
            if self.config['conditional']:
                self.t = [tf.placeholder(tf.float32, shape=[self.config['batch_size']]) for _ in range(self.config['nr_gpu'])]
            else:
                self.t = [None for _ in range(self.config['nr_gpu'])]

            if self.config['conditional']:
                self.ins_outs     = {'inputs': self.x, 't': self.t}
            else:
                self.ins_outs     = {'inputs': self.x}

        elif mode == 1:
            # inferencing
            self.x  = tf.placeholder(tf.float32, shape=[self.config['batch_size']]+self.config['input_shape']) 

            if self.config['conditional']:
                self.t = tf.placeholder(tf.float32, shape=[self.config['batch_size']])
            else:
                self.t = None

        elif mode == 2:
            # exporting
            self.x  = tf.placeholder(tf.float32, shape=[self.config['batch_size']]+self.config['input_shape'], name='input_0')
            #self.x  = tf.transpose(self.x, [0, 2, 1, 3])
            if self.config['conditional']:
                self.t = tf.placeholder(tf.float32, shape=[self.config['batch_size']], name='input_1')
            else:
                self.t = None

    def init(self, mode=0):

        model_opt = {'nr_resnet': self.config['nr_resnet'],
                     'nr_filters': self.config['nr_filters'],
                     'nr_logistic_mix': self.config['nr_logistic_mix'],
                     'data_chns': self.config['data_chns'],
                     'rlt': self.config['rlt'],
                     'layer_norm': self.config['layer_norm']
                    }

        if self.config['data_chns'] == DATA_CHNS.CPLX:
            if self.config['rlt'] == 1:
                # assume the linear dependence between real and imaginary parts
                from spreco.model.logistic_loss import discretized_mix_logistic_loss_2 as loss_func
            elif self.config['rlt'] == 2:
                # assume independence between real and imaginary parts
                from spreco.model.logistic_loss import discretized_mix_logistic_loss_idp_2 as loss_func
            elif self.config['rlt'] == 3:
                # assume joint distribution
                from spreco.model.logistic_loss import discretized_mix_logistic_loss_bivariate as loss_func
            else:
                raise Exception("please check the assumption of relationship between real and imaginary")

        if self.config['data_chns'] == DATA_CHNS.MAG:
            from spreco.model.logistic_loss import discretized_mix_logistic_loss_1 as loss_func

        self.init_placeholder(mode)

        if mode == 0:
            init_pass = self.forward(self.x[0], init=True, dropout_p=self.config['dropout_rate'], **model_opt) # initialization of parameters
            all_params = tf.trainable_variables()

            loss      = []
            grads     = []
            loss_test = []
            optimizer = AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999)

            for i in range(self.config['nr_gpu']):
                with tf.device('/gpu:%d'%i):
                    # train
                    logits = self.forward(self.x[i], ema=None, dropout_p=self.config['dropout_rate'], **model_opt)
                    loss.append(loss_func(tf.stop_gradient(self.x[i]), logits, itg_interval=self.config['itg_interval']))
                    
                    gvs = optimizer.compute_gradients(loss[-1], all_params)
                    gvs = [(k, v) for (k, v) in gvs if k is not None]
                    grads.append(gvs)

                    # test
                    logits = self.forward(self.x[i], ema=None, dropout_p=0.0, **model_opt)
                    loss_test.append(loss_func(self.x[i], logits, itg_interval=self.config['itg_interval']))

            with tf.device('/gpu:0'):
                for i in range(1,self.config['nr_gpu']):
                    loss[0]      += loss[i]
                    loss_test[0] += loss_test[i]

            grads_avg = optimizer.average_gradients(grads)

            self.train_op    = optimizer.apply_gradients(grads_avg)
            self.loss_train  = loss[0]/(np.log(2.0)*np.prod(self.config['input_shape'])*self.config['batch_size']*self.config['nr_gpu'])
            self.loss_test   = loss_test[0]/(np.log(2.0)*np.prod(self.config['input_shape'])*self.config['batch_size']*self.config['nr_gpu'])
        
        elif mode == 1:

            init_pass = self.forward(self.x, init=True, dropout_p=0., **model_opt) 
            self.out = self.forward(self.x, dropout_p=0., **model_opt)

            loss = loss_func(self.x, self.out)
            self.loss = loss/(np.log(2.0)*np.prod(self.config['input_shape'])*1)

            self.grads = tf.squeeze(tf.gradients(self.loss, self.x), name='grad_0') 
        
        elif mode == 2:
            init_pass = self.forward(self.x, init=True, dropout_p=0., **model_opt) 
            self.out = self.forward(self.x, dropout_p=0., **model_opt)

            loss = loss_func(self.x, self.out)
            loss = loss/(np.log(2.0)*np.prod(self.config['input_shape'])*1)

            output = tf.identity(tf.stack([loss, tf.zeros_like(loss)], axis=-1), name='output_0')
            self.grads = tf.squeeze(tf.gradients(loss, self.x), name='grad_0') 
            #self.grads = tf.transpose(self.grads, [1, 0, 2], name='grad_0')
            grad_ys = tf.placeholder(tf.float32, shape=[2], name='grad_ys_0')


    @staticmethod
    def body(x, h=None, init=False, ema=None, layer_norm=False, dropout_p=0.5, nr_resnet=3, nr_filters=160, nr_logistic_mix=10, resnet_nonlinearity='concat_elu', data_chns=DATA_CHNS.CPLX, rlt=1):
        """
        We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
        a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
        of the x_out tensor describes the predictive distribution for the RGB at
        that position.
        'h' is an optional N x K matrix of values to condition our generative model on
        """

        if data_chns==DATA_CHNS.MAG:
            x = 2*x - 1   #TODO should be moved to data pipe

        counters = {}
        with arg_scope([nn.conv2d, nn.deconv2d, nn.gated_resnet, nn.dense, nn.layer_norm], counters=counters, init=init, ema=ema, dropout_p=dropout_p):

            # parse resnet nonlinearity argument 
            # TODO should be moved to model/utils.py
            if resnet_nonlinearity == 'concat_elu':
                resnet_nonlinearity = nn.concat_elu
            elif resnet_nonlinearity == 'elu':
                resnet_nonlinearity = tf.nn.elu
            elif resnet_nonlinearity == 'relu':
                resnet_nonlinearity = tf.nn.relu
            else:
                raise('resnet nonlinearity ' + resnet_nonlinearity + ' is not supported')

            with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity, h=h):

                # ////////// up pass through pixelCNN ////////
                xs = nn.int_shape(x)
                x_pad = tf.concat([x,tf.ones(xs[:-1]+[1])],3) # add channel of ones to distinguish image from padding later on
                u_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 3]))] # stream for pixels above
                ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1,3])) + \
                            nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2,1]))] # stream for up and to the left
                
                # 1
                if layer_norm:
                    u_list[-1] = nn.layer_norm(u_list[-1])
                    ul_list[-1] = nn.layer_norm(ul_list[-1])

                for rep in range(nr_resnet):
                
                    u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d,))
                    ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))

                u_list.append(nn.down_shifted_conv2d(u_list[-1], num_filters=nr_filters, stride=[2, 2]))
                ul_list.append(nn.down_right_shifted_conv2d(ul_list[-1], num_filters=nr_filters, stride=[2, 2]))

                # 2
                if layer_norm:
                    u_list[-1] = nn.layer_norm(u_list[-1])
                    ul_list[-1] = nn.layer_norm(ul_list[-1])

                for rep in range(nr_resnet):
                
                    u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
                    ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))

                u_list.append(nn.down_shifted_conv2d(u_list[-1], num_filters=nr_filters, stride=[2, 2]))
                ul_list.append(nn.down_right_shifted_conv2d(ul_list[-1], num_filters=nr_filters, stride=[2, 2]))

                # 3
                if layer_norm:
                    u_list[-1] = nn.layer_norm(u_list[-1])
                    ul_list[-1] = nn.layer_norm(ul_list[-1])

                for rep in range(nr_resnet):
                
                    u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
                    ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))

                # remember nodes
                for t in u_list+ul_list:
                    tf.add_to_collection('checkpoints', t)

                # /////// down pass ////////
                
                u = u_list.pop()
                ul = ul_list.pop()
                
                # 1
                if layer_norm:
                    u = nn.layer_norm(u)
                    ul = nn.layer_norm(ul)
                
                for rep in range(nr_resnet):
                
                    u = nn.gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d)
                    ul = nn.gated_resnet(ul, tf.concat([u, ul_list.pop()],3), conv=nn.down_right_shifted_conv2d)
                    tf.add_to_collection('checkpoints', u)
                    tf.add_to_collection('checkpoints', ul)
                

                u = nn.down_shifted_deconv2d(u, num_filters=nr_filters, stride=[2, 2])
                ul = nn.down_right_shifted_deconv2d(ul, num_filters=nr_filters, stride=[2, 2])
                
                # 2
                if layer_norm:
                    u = nn.layer_norm(u)
                    ul = nn.layer_norm(ul)

                for rep in range(nr_resnet+1):
                
                    u = nn.gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d)
                    ul = nn.gated_resnet(ul, tf.concat([u, ul_list.pop()],3), conv=nn.down_right_shifted_conv2d)
                    tf.add_to_collection('checkpoints', u)
                    tf.add_to_collection('checkpoints', ul)            

                u = nn.down_shifted_deconv2d(u, num_filters=nr_filters, stride=[2, 2])
                ul = nn.down_right_shifted_deconv2d(ul, num_filters=nr_filters, stride=[2, 2])
                

                #  3
                if layer_norm:
                    u = nn.layer_norm(u)
                    ul = nn.layer_norm(ul)

                for rep in range(nr_resnet+1):
                
                    u = nn.gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d)
                    ul = nn.gated_resnet(ul, tf.concat([u, ul_list.pop()],3), conv=nn.down_right_shifted_conv2d)
                    tf.add_to_collection('checkpoints', u)
                    tf.add_to_collection('checkpoints', ul)

                """
                nr logits 3, 6 or 10 depends on the number of image channel magnitude, real/image or red/green/blue
                """
                if data_chns == DATA_CHNS.RGB:
                    nr_logits = 10
                if data_chns == DATA_CHNS.CPLX:
                    if rlt == 1:
                        nr_logits = 6
                    elif rlt == 2:
                        nr_logits = 5
                    elif rlt == 3:
                        nr_logits = 5
                    else:
                        raise Exception("rlt shoulde be 1 or 2 or 3.")
                if data_chns == DATA_CHNS.MAG:
                    nr_logits = 3
                    
                x_out = nn.nin(tf.nn.elu(ul),nr_logits*nr_logistic_mix) 

                assert len(u_list) == 0
                assert len(ul_list) == 0

                return x_out
