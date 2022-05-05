import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from common.options import MODELS
from common import utils
from common.logger import logger

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import numpy as np
import time

class trainer():
    """
    """

    def __init__(self, train_pipe, test_pipe, config):

        self.train_pipe = train_pipe
        self.test_pipe  = test_pipe
        self.config     = config

        if train_pipe is not None:
            self.log_path   = utils.create_folder(config['log_folder'])
            utils.save_config(config, self.log_path)
            self.logger = logger(self.log_path)

        self.model      = None
        self.sess       = None
        self.pred_y     = None

        self.global_step = 0

    
    def get_model(self, export):
        """
        TODO: put this function into options
        """
        
        if self.config['model'] == MODELS.NCSN:
            tf.random.set_random_seed(self.config['seed'])
            from model.ncsn import ncsn as selected_class

        elif self.config['model'] == MODELS.SDE:
            from model.sde import sde as selected_class

        else:
            raise Exception("Currently, this model is not implemented!")

        self.model = selected_class(self.config)
        self.model.prep(export)

    def mk_feed_dict(self, pipe, reps=None):
        """"utility to make feed dict"""
        if reps is None:
            reps = self.config['nr_gpu']

        keys = self.model.ins_outs.keys()
        elm = pipe.get_next()
        feed_dict ={}

        for key in keys:
            tmp = np.split(elm[key], self.config['nr_gpu'])
            feed_dict.update({self.model.ins_outs[key][i]: tmp[i] for i in range(reps)})

        return feed_dict

    def train_loop(self):

        # ready to go
        init_op    = tf.global_variables_initializer()
        saver      = tf.train.Saver(max_to_keep=self.config['max_keep'])
        gpu_config = tf.ConfigProto(allow_soft_placement=True)
        sess       = tf.Session(config=gpu_config)
        sess.run(init_op)

        utils.print_parameters(self.log_path+'/layer_info')
        train_loss = []
        test_loss  = []
        begin      = time.time()

        while self.train_pipe.get_epoch() < self.config['max_epochs']:

            feed_dict = self.mk_feed_dict(self.train_pipe)

            learning_rate = utils.get_lr(self.global_step, self.config['lr'],
                                         warmup_steps=None if 'warmup_steps' not in self.config.keys() else self.config['warmup_steps'],
                                         hidden_size=None if 'hidden_size' not in self.config.keys() else self.config['hidden_size'])

            feed_dict.update({self.model.learning_rate: learning_rate})

            l, _      = sess.run([self.model.loss_train, self.model.train_op], feed_dict=feed_dict)
            self.global_step = self.global_step + 1

            kvs = {}
            kvs['train_loss'] = l
            kvs['learning_rate'] = learning_rate
            self.logger.writekvs(kvs)

            train_loss.append(l)

            if self.train_pipe.check_epoch():

                # one epoch is finised
                self.train_pipe.update_epoch()
                if self.train_pipe.get_epoch() % self.config["save_interval"] == 0:
                    saver.save(sess, os.path.join(self.log_path, self.config['saved_name']+'_'+str(self.train_pipe.get_epoch())))

                info = "Epochs %d, time = %ds, train loss = %.4f" % (self.train_pipe.get_epoch(), time.time()-begin, np.mean(train_loss))
                if self.config['print_loss']:
                    print(info)
                utils.log_to(os.path.join(self.log_path, 'loss'), [info], prefix="->")

                begin      = time.time()
                train_loss = []

                # run test
                if self.test_pipe is not None:
                    
                    while not self.test_pipe.check_epoch():
                        feed_dict = self.mk_feed_dict(self.test_pipe)
                        l         = sess.run(self.model.loss_test, feed_dict=feed_dict)

                        kvs = {}
                        kvs['test_loss'] = l
                        self.logger.writekvs(kvs)

                        test_loss.append(l)
                        
                    self.test_pipe.update_epoch()

                    info = "Epochs %d, time = %ds, test loss = %.4f" % (self.test_pipe.get_epoch(), time.time()-begin, np.mean(test_loss))
                    if self.config['print_loss']:
                        print(info)
                    else:
                        utils.log_to(os.path.join(self.log_path, 'loss'), [info], prefix="->")

                    test_loss  = []
                    begin      = time.time()

    def train(self):

        # set gpu
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config['gpu_id']

        self.train_pipe.start()
        if self.test_pipe is not None:
            self.test_pipe.start()
        self.get_model(export=False)
        self.train_loop()
        self.train_pipe.stop()
        if self.test_pipe is not None:
            self.test_pipe.stop()

    def export(self, model_path, export_path, name):
        
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config['gpu_id']

        self.get_model(export=True)
        saver, sess, gpu_id = self.model.restore(model_path)
        utils.export_model(saver, sess, export_path, name, gpu_id=gpu_id)