from spreco.common.options import MODELS
from spreco.common import utils
from spreco.common.logger import tb_logger
from spreco.common.utils import LambdaWarmUpCosineScheduler

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import os
import numpy as np
import time

class trainer():
    """
    mode, 0: train
    """

    def __init__(self, train_pipe, test_pipe, config, default_feed=True):

        self.train_pipe = train_pipe
        self.test_pipe  = test_pipe
        self.config     = config
        self.model      = None

        self.global_step  = 0
        self.epoch        = 0
        self.default_feed = default_feed

        self.log_path = utils.create_folder(config['log_folder'])
        self.logger   = tb_logger(self.log_path)

        utils.save_config(config, self.log_path)

    def init_model(self, feed_func=None, gpu_id=None):
        
        if self.config['model'] == MODELS.NCSN:
            tf.random.set_random_seed(self.config['seed'])
            from spreco.model.ncsn import ncsn as selected_class

        elif self.config['model'] == MODELS.SDE:
            from spreco.model.sde import sde as selected_class

        elif self.config['model'] == MODELS.SDE2:
            from spreco.model.sde2 import sde as selected_class

        elif self.config['model'] == MODELS.PIXELCNN:
            from spreco.model.pixelcnn import pixelcnn as selected_class

        else:
            raise Exception("Currently, this model is not implemented!")

        self.lr_scheduler = LambdaWarmUpCosineScheduler(self.config['lr_warm_up_steps'],
                                self.config['lr_min'], self.config['lr_max'], self.config['lr_start'], self.config['lr_max_decay_steps'])

        self.model = selected_class(self.config)

        self.model.init(mode=0)
        if not self.default_feed:
            self.feed_func=feed_func

        # set gpu
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if gpu_id is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
        else: 
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    def mk_feed_dict(self, batch, reps=None):
        """"utility to make feed dict"""
        if reps is None:
            reps = self.config['nr_gpu']

        keys = self.model.ins_outs.keys()
        feed_dict ={}

        for key in keys:
            tmp = np.split(batch[key], self.config['nr_gpu'])
            feed_dict.update({self.model.ins_outs[key][i]: tmp[i] for i in range(reps)})

        return feed_dict

    def run_op(self, sess, ops, loss, feed_dict, is_training):
        
        if is_training:
            if type(ops) == list:
                l = []
                for op, l_op in zip(ops, loss):
                    l_tmp, _ = sess.run([l_op, op], feed_dict=feed_dict)
                    l.append(l_tmp)

            else:
                l, _ = sess.run([loss, ops], feed_dict=feed_dict)

            return l

        else:
            if type(loss) == list:
                l = []
                for l_op in loss:
                    l_tmp = sess.run(l_op, feed_dict=feed_dict)
                    l.append(l_tmp)

            else:
                l = sess.run(loss, feed_dict=feed_dict)

            return l

    def log_tb(self, keys, values, epoch=None):
        kvs = {}
        for key, value in zip(keys, values):
            if type(value) == list:
                for i, a in enumerate(value):
                    key_ = "%s_%d"%(key,i)
                    kvs[key_] = a
            else:
                kvs[key] = value
        self.logger.writekvs(kvs, epoch)

    def log_info(self, epoch, t, loss_avg, printing, is_training=True):

        if is_training:
            info = "Epochs %d, time %ds, train loss: %s" % (epoch, t, ''.join(str(loss_avg)))
        else:
            info = "Epochs %d, time %ds, test loss: %s" % (epoch, t, ''.join(str(loss_avg)))

        if printing:
            print(info)
        utils.log_to(os.path.join(self.log_path, 'loss'), [info], prefix="->")

    def train_loop(self):

        # ready to go
        init_op    = tf.global_variables_initializer()
        saver      = tf.train.Saver(max_to_keep=self.config['max_keep'])
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth=True
        sess = tf.Session(config=gpu_config)
        sess.run(init_op)

        if "restore_path" in self.config.keys():
            saver.restore(sess, self.config['restore_path'])
            info = [utils.get_timestamp() + ", the training session was restored from %s"%self.config['restore_path']]
            utils.log_to(os.path.join(self.log_path, 'train_info'), info, prefix="")

        utils.print_parameters(self.log_path+'/layer_info')

        for epoch in range(self.config['max_epochs']):

            begin_t    = time.time()
            train_loss = []
            test_loss  = []

            #training 
            for batch in self.train_pipe:

                if self.default_feed:
                    feed_dict = self.mk_feed_dict(batch)
                else:
                    feed_dict = self.feed_func(batch)

                learning_rate = self.lr_scheduler(self.global_step)

                feed_dict.update({self.model.learning_rate: learning_rate})

                l = self.run_op(sess, self.model.train_op, self.model.loss_train, feed_dict, is_training=True)
                self.global_step = self.global_step + 1

                self.log_tb(['train_loss', 'learning_rate'], [l, learning_rate])
                train_loss.append(l)

            #testing
            for batch in self.test_pipe:

                if self.default_feed:
                    feed_dict = self.mk_feed_dict(batch)
                else:
                    feed_dict = self.feed_func(batch)

                l = self.run_op(sess, None, self.model.loss_test, feed_dict=feed_dict, is_training=False)
                test_loss.append(l)
            
            end_t = time.time()

            # save model
            self.epoch = self.epoch + 1
            if self.epoch % self.config["save_interval"] == 0:
                saver.save(sess, os.path.join(self.log_path, self.config['saved_name']+'_'+str(self.epoch)))

            # log info
            train_loss_avg = np.mean(train_loss, axis=0).tolist()
            self.log_info(epoch, end_t-begin_t, train_loss_avg, self.config['print_loss'])
            self.log_tb(['train_loss_avg'], [train_loss_avg], self.epoch)

            test_loss_avg = np.mean(test_loss, axis=0).tolist()
            self.log_info(epoch, end_t-begin_t, test_loss_avg, self.config['print_loss'], False)
            self.log_tb(['test_loss_avg'], [test_loss_avg], self.epoch)


    def train(self, feed_func=None):
        self.train_pipe.reset_state()
        self.test_pipe.reset_state()
        self.init_model(feed_func=feed_func, gpu_id=self.config['gpu_id'])
        self.train_loop()
