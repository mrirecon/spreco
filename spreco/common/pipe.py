from . import utils
import numpy as np
import os

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from multiprocessing import Queue, current_process, Process
from spreco.worker import worker

class data_pipe(object):
    """
    give part,
    create pipe, start, get

    parts may includes scale_func, noise_func, slice_func, normalize_func, file_loader, transform_func
    TODO: read data simultaneously
    """
    def __init__(self, parts, files, buffer_size, batch_size, shape_info, names=[None], need_sess=[False], config_path=[None], model_path=[None], parallel=False):

        self.parts        = parts     # a list contains functionalities for data preparation (normalization, augmentation)
        self.files        = files     # a list contains paths of training files
        self.queue        = Queue(buffer_size)
        self.names        = names     # names of streams [inputs, labels]
        self.rseed        = None
        self.batch_size   = batch_size
        self.shape_info   = shape_info
        self.epoch_hook   = Queue(1)
        self.epoch        = 0
        self.p_steams     = len(parts) # number of streams of preprocess function
        self.nr_files     = len(files)
        self.idx          = 0
        self.is_next_file = True
        self.local_random = np.random.RandomState(self.rseed)
        self.data_idx     = self.local_random.permutation(self.nr_files)
        self.lefted       = [np.zeros((None)) for _ in range(self.p_steams)]
        self.batch        = [np.zeros(self.shape_info[i]) for i in range(self.p_steams)]
        self.cur_remained = 0

        self.need_sess   = need_sess
        self.config_path = config_path
        self.model_path  = model_path
        self.worker      = [None for _ in range(len(need_sess))]
        self.parallel    = parallel

        for i, sess_flag in enumerate(need_sess):
            if sess_flag:
                self.worker[i] = worker(None, None, utils.load_config(config_path[i]))

    def get_epoch(self):
        return self.epoch
    
    def update_epoch(self):
        self.epoch = self.epoch_hook.get()

    def check_epoch(self):
        return self.epoch_hook.full()

    def fetch_single(self, idx):
        pieces = [np.zeros((None)) for _ in range(self.p_steams)]
        for i in range(self.p_steams):
            parts = self.parts[i]
            if self.parallel:
                flow  = self.files[self.data_idx[self.idx:self.idx+self.batch_size]]
            else:
                flow  = self.files[self.data_idx[idx]]
            for part in parts:
                if self.need_sess and 'need_sess' in part.__code__.co_varnames:
                    flow=flow
                else:
                    flow = part(flow)
            pieces[i] = flow
        return pieces

    def kernel(self, need_sess=False, gpu_id=None):

        for i, sess_flag in enumerate(self.need_sess):
            if sess_flag:
                self.worker[i].init_model(is_training=False, gpu_id=gpu_id)
                if self.model_path[i] is not None:
                    _, sess, _ = self.worker[i].restore(self.model_path[i])
                    self.worker[i].sess  = sess
                else:
                    self.worker[i].sess = tf.Session()

        while True:
            while not self.queue.full():

                if self.is_next_file:

                    if self.idx == self.nr_files:
                        self.idx = 0
                        self.epoch = self.epoch + 1
                        self.epoch_hook.put(self.epoch)
                        self.data_idx = self.local_random.permutation(self.nr_files)
                        utils.color_print("Files permuted", 'red')

                    pieces = [np.zeros((None)) for _ in range(self.p_steams)]
                    for i in range(self.p_steams):
                        parts = self.parts[i]
                        if self.parallel:
                            flow  = self.files[self.data_idx[self.idx:self.idx+self.batch_size]]
                        else:
                            flow  = self.files[self.data_idx[self.idx]]
                        for part in parts:
                            if self.need_sess and 'need_sess' in part.__code__.co_varnames:
                                flow = part(flow, self.worker[i])
                            else:
                                flow = part(flow)
                        pieces[i] = flow
                    
                    if self.parallel:
                        self.idx = self.idx + self.batch_size
                    else:
                        self.idx = self.idx + 1
                    

                    for i in range(self.p_steams):
                        if self.cur_remained == 0:
                            self.lefted = pieces  
                        else:
                            self.lefted[i] = np.concatenate([self.lefted[i], pieces[i]], axis=0)
                    self.cur_remained = self.check_remained(self.lefted)

                    if self.cur_remained > self.batch_size:
                        self.is_next_file = False
                        for i in range(self.p_steams):
                        
                            self.batch[i]  = self.lefted[i][0:self.batch_size, ...]
                            self.lefted[i] = self.lefted[i][self.batch_size:, ...]
                        
                        if self.names is not None:
                            tmp = {}
                            for i in range(self.p_steams):
                                tmp[self.names[i]] = self.batch[i]
                            self.queue.put(tmp)
                        else:
                            self.queue.put(self.batch)
                            
                        self.cur_remained = self.check_remained(self.lefted)
                        if self.cur_remained <= self.batch_size:
                            self.is_next_file = True
                
                else:
                    if self.cur_remained > self.batch_size:
                        self.is_next_file = False
                        for i in range(self.p_steams):
                        
                            self.batch[i]  = self.lefted[i][0:self.batch_size, ...]
                            self.lefted[i] = self.lefted[i][self.batch_size:, ...]
                        
                        if self.names is not None:
                            tmp = {}
                            for i in range(self.p_steams):
                                tmp[self.names[i]] = self.batch[i]
                            self.queue.put(tmp)
                        else:
                            self.queue.put(self.batch)
                            
                        self.cur_remained = self.check_remained(self.lefted)
                        if self.cur_remained <= self.batch_size:
                            self.is_next_file = True

    def check_remained(self, lefted, idx=0):
        """
        return the number of batches remained in the list "lefted"
        """
        if lefted[idx].shape == ():
            cur_remained = 0
        else:
            cur_remained = lefted[idx].shape[0]
        return cur_remained

    def get_next(self):
        """
        return one batch which is a numpy array
        """
        return self.queue.get()
    
    def start(self, gpu_id=None):
        self.proc = Process(target=self.kernel, args=(self.need_sess, gpu_id))
        self.proc.daemon = True
        self.proc.start()
    
    def stop(self):
        self.proc.terminate()

def create_pipe(parts_funcs, files, batch_size, shape_info, buffer_size=10, names=[None], need_sess=[False], config_path=[None], model_path=[None], parallel=False):
    pipe = data_pipe(parts=parts_funcs,
                       files=files,
                       buffer_size=buffer_size,
                       batch_size=batch_size,
                       shape_info=shape_info,
                       names=names,need_sess=need_sess,
                       config_path=config_path,
                       model_path=model_path, parallel=parallel)
    try:
        _ = pipe.fetch_single(0)
        return pipe
    except:
        raise Exception("Creating pipe failed, please check your parts funcs")
