from spreco.common import utils
from spreco.trainer import trainer
from spreco.dataflow.base import RNGDataFlow
from spreco.dataflow.common import BatchData
from spreco.dataflow.parallel_map import MultiThreadMapData


import os
import numpy as np
import argparse

class cfl_pipe(RNGDataFlow):

    def __init__(self, files, shuffle):
        self._size   = len(files)
        self.files   = files
        self.shuffle = shuffle
    
    def __len__(self):
        return len(self.files)
    
    def __iter__(self):
        idxs = np.arange(self._size)
        if self.shuffle:
            self.rng.shuffle(idxs)
        
        for idx in idxs:
            fname = self.files[idx]
            yield fname

def main(config_path):

    try:
        config = utils.load_config(config_path)
    except:
        raise Exception('Loading config file failed, please check if it exists.')

    try: 
        train_files = utils.read_filelist(config['train_list'])
    except:
        raise Exception("Load the list of train files failed, please check!")
    
    try:
        test_files = utils.read_filelist(config['test_list'])
    except:
        raise Exception("Load the list of test files failed, please check!")


    def load_file(x):
        """
        x     ---> file path
        imgs  ---> normalized images with shape (batch_size, x, y, 2) 
        """
        #path, ext = os.path.splitext(x)
        imgs = np.squeeze(np.load(x)['rss'])[np.newaxis, ...]
        imgs = imgs / np.max(np.abs(imgs), axis=(1,2), keepdims=True)
        imgs = utils.cplx2float(imgs)
        return imgs

    def flip_and_rotate(x, case=1):

        if case == 1:
            # flip leftside right
            x = x[:,:,::-1,...]
        elif case == 2:
            # flip upside down
            x = x[:,::-1,:,...]
        elif case == 3:
            # 
            x = x[:,:,::-1,...]
            x = x[:,::-1,:,...]
        elif case == 4:
            x = np.rot90(x, k=1, axes=(1,2))
        elif case == 5:
            x = np.rot90(x, k=1, axes=(1,2))
            x = x[:,:,::-1,...]
        elif case == 6:
            x = np.rot90(x, k=3, axes=(1,2))
        elif case == 7:
            x = np.rot90(x, k=3, axes=(1,2))
            x = x[:,:,::-1,...]
        elif case == 8:
            x = x
        else:
            raise Exception("check you the number of possible cases!")
        return x
    

    def aug_load_file(x):
        x = np.mean(load_file(x), axis=0, keepdims=True)
        x = np.squeeze(flip_and_rotate(x, 8))
        return x

    def randint(x, dtype='int32'):
        # x is a dummy arg
        return np.random.randint(0, config['nr_levels'], (1), dtype=dtype)

    def randfloat(x):
        # x is a dummy arg
        return np.random.uniform(config['sigma_min'], config['sigma_max'], size=(1))

    if config['model'] == 'NCSN':
        def map_f(x):
            d1 = aug_load_file(x)
            d2 = np.squeeze(randint(x))
            return {"inputs": d1, "t": d2}
    elif config['model'] == 'SDE':
        def map_f(x):
            d1 = aug_load_file(x)
            d2 = np.squeeze(randfloat(x))
            return {"inputs": d1, "t": d2}
    elif config['model'] == 'PIXELCNN':
        def map_f(x):
            d = aug_load_file(x)
            return {"inputs": d}
    else:
        raise Exception("Please select NCSN or SDE or PIXELCNN")

    nr_elm = config['batch_size']*config['nr_gpu']

    d1 = cfl_pipe(train_files, True)
    d1 = MultiThreadMapData(d1, num_thread=config['num_thread'], map_func=map_f,  buffer_size=nr_elm*10, strict=True)
    train_pipe = BatchData(d1, nr_elm, use_list=False)

    d2 = cfl_pipe(test_files, True)
    d2 = MultiThreadMapData(d2, num_thread=config['num_thread'], map_func=map_f,  buffer_size=nr_elm*10, strict=True)
    test_pipe = BatchData(d2, nr_elm, use_list=False)

    go = trainer(train_pipe, test_pipe, config)
    utils.log_to(os.path.join(go.log_path, 'train_info'), [utils.get_timestamp() + ", the training is starting"])
    go.train()
    utils.log_to(os.path.join(go.log_path, 'train_info'), [utils.get_timestamp() + ", the training is ending"])
    utils.color_print('TRAINING FINISHED')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', metavar='path', default='/home/gluo/spreco/tests/sde.yaml', help='')

    args = parser.parse_args()
    main(args.config)