import os
from datetime import datetime
from termcolor import colored
from contextlib import redirect_stdout

import tensorflow.compat.v1 as tf
import tempfile 
import yaml

import subprocess as sp
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def float2cplx(float_in):
    return np.array(float_in[...,0]+1.0j*float_in[...,1], dtype='complex64')

def cplx2float(cplx_in):
    return np.array(np.stack((cplx_in.real, cplx_in.imag), axis=-1), dtype='float32')

def log_to(file, vars, mode="a", clear=False, end='\n', prefix=None):
    if clear:
        open(file, "w").close()
    else:
        with open(file, mode) as f:
            with redirect_stdout(f):
                
                for var in vars:
                    if prefix is not None:
                        print(prefix, end='')
                    print(var, end=end)

def save_img(img, path, vmin=0., vmax=1., cmap='gray', interpolation=None):
    """
    print images to pdf and png without white margin

    Args:
    img: image arrays
    path: saving path
    """
    plt.imshow(img, cmap=cmap, interpolation=interpolation, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches = 0)
    plt.savefig(path+'.pdf', bbox_inches='tight', pad_inches = 0)
    plt.close()

def load_config(path):
    """
    load configuration defined with yaml file
    """
    with open(path, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    return config

def save_config(x,path):
    with open(os.path.join(path, 'config.yaml'), 'w') as yaml_file:
        yaml.dump(x, yaml_file, default_flow_style=False, sort_keys=False)
    
def create_folder(save_path, time=True):
    """
    create folder for logs
    """
    if time:
        log_path = os.path.join(save_path, datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        log_path = save_path

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    return log_path

def check_out(cmd, split=True):
    """ utility to check_out terminal command and return the output"""

    strs = sp.check_output(cmd, shell=True).decode()

    if split:
        split_strs = strs.split('\n')[:-1]
    return split_strs

def find_files(path, pattern):
    cmd = "find " + path + " -type f -name " + pattern
    return check_out(cmd)

def read_filelist(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f]
        return lines

def list_data_sys(path, suffix, ex_pattern=None):
    """
    list all the files under path with given suffix

    Args:

    path: the folder to be listed
    suffix: suffix of file to be listed
    ex_pattern: pattern for excluding files

    Returns:
    
    array of the files' path
    """

    cmd = 'ls '
    cmd = cmd + path + '/*.' + suffix

    if ex_pattern is not None:
        cmd = cmd + ' | ' + 'grep -v ' + ex_pattern
    
    strs = sp.check_output(cmd, shell=True).decode()
    files = strs.split('\n')[:-1]

    return files

def slice_image(inp, shape):
    """
    slice image into pieces

    Args:
    inp: (nx, ny, _)
    shape: 

    Returns:
    
    """
    nx = ny = None
    if len(inp.shape) == 3:
        nx, ny, _ = inp.shape
    elif len(inp.shape) == 2:
        nx, ny = inp.shape
    else:
        print(inp.shape)
        raise Exception("please check the shape of input")

    if len(shape) == 3:
        sx, sy, _ = shape
    elif len(shape) == 2:
        sx, sy = shape
    else:
        raise Exception("Please check the given shape")

    steps_x = int(np.ceil(float(nx)/sx))
    steps_y = int(np.ceil(float(ny)/sy))

    total = steps_x*steps_y
    pieces = np.zeros([total] + shape, dtype=inp.dtype)
     
    for x in range(steps_x):
        
        if x == (steps_x-1):
            bx = nx-sx
            ex = nx
        else:
            bx = x*sx
            ex = bx + sx

        for y in range(steps_y):

            if y == (steps_y-1):
                by = ny-sy
                ey = ny
            else:
                by = y*sy
                ey = by + sy

            pieces[x*steps_y+y, ...] = np.reshape(inp[bx:ex, by:ey], shape)
    return pieces

def npz_loader(filename, key='rss'):
    tmp = np.load(filename)
    if key not in tmp.keys():
        info = "File loading failed, key %s doesn't match or exist!"%key
        #color_print(info)
        raise Exception(info)
    else:
        image = tmp[key]
    return image

def normalize_with_max(x, axis=(0,1), data_chns='CPLX'):
    """
    x is complex value
    x = x/(max(abs(x)))
    """
    scalor = np.max(abs(x), axis)

    if data_chns == 'CPLX':
        normalized_x = cplx2float(x/scalor)
        return normalized_x
    
    if data_chns == 'MAG':
        normalized_x = abs(x/scalor)
        return normalized_x

def color_print(strs, color='red', bold=True):
    if bold:
        print(colored(strs, color, attrs=['bold']))
    else:
        print(colored(strs, color))

def readcfl(name):
    """
    read cfl file
    """

    # get dims from .hdr
    h = open(name + ".hdr", "r")
    h.readline() # skip
    l = h.readline()
    h.close()
    dims = [int(i) for i in l.split( )]

    # remove singleton dimensions from the end
    n = np.prod(dims)
    dims_prod = np.cumprod(dims)
    dims = dims[:np.searchsorted(dims_prod, n)+1]

    # load data and reshape into dims
    d = open(name + ".cfl", "r")
    a = np.fromfile(d, dtype=np.complex64, count=n)
    d.close()
    return a.reshape(dims, order='F') # column-major

def writecfl(name, array):
    """
    write cfl file
    """

    h = open(name + ".hdr", "w")
    h.write('# Dimensions\n')
    for i in (array.shape):
            h.write("%d " % i)
    h.write('\n')
    h.close()
    d = open(name + ".cfl", "w")
    array.T.astype(np.complex64).tofile(d) # tranpose for column-major order
    d.close()

def transform(inp):
    """
    function for data augmentation
    TODO: zoom in, zoom out
    """
    k = np.random.choice(12)
    if k == 0:
        return inp
    if k == 1:    
        return np.flipud(inp)
    if k == 2:
        return np.fliplr(inp)
    if k == 3:
        return np.rot90(inp, 1)
    if k == 4:
        return np.rot90(inp, 2)
    if k == 5:
        return np.rot90(inp, 3)
    if k == 6:
        return np.rot90(np.flipud(inp), 1)
    if k == 7:
        return np.rot90(np.flipud(inp), 2)
    if k == 8:
        return np.rot90(np.flipud(inp), 3)
    if k == 9:
        return np.rot90(np.fliplr(inp), 1)
    if k == 10:
        return np.rot90(np.fliplr(inp), 2)
    if k == 11:
        return np.rot90(np.fliplr(inp), 3)


def get_timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def export_model(saver, sess, path, name, as_text=False, gpu_id=None):
    """
    not needed will be delete in the future
    """
    saver.save(sess, os.path.join(path, name))
    tf.train.write_graph(sess.graph, path, name+'.pb', as_text)
    if gpu_id is not None:
        with open(os.path.join(path, name+'_gpu_id'), 'w+') as fs:
            for i in range(len(gpu_id)):
                fs.write(gpu_id[i])
                fs.write('\t')

def random_rescale(x, vmin=0.7, vmax=1.):
    return x*round(np.random.uniform(vmin,vmax),4)

def randint(x, nr_levels, dtype='int32'):
    # x is a dummy arg
    return np.random.randint(0, nr_levels, (1), dtype=dtype)

def randfloat(x, eps, T):
    # x is a dummy arg
    return np.random.uniform(eps, T, size=(1))

def to_dict(val, key='inputs'):
    return {key: val}

def gaussian_noise(shape, mu=0, sigma=1):
        return np.random.normal(mu, sigma, shape).astype(np.float32)
    
def gaussian_noise_bivariate(shape, mu=[0, 0], sigma=[[1, 0],[0, 1]]):
        tmp = np.random.multivariate_normal(mu, sigma, np.prod(shape)).astype(np.float32)
        return float2cplx(np.reshape(tmp, [shape[0], shape[1], 2]))

def affine(x, a, b):
    return (x-a)/b

def expand_dims(x, axis):
    return np.expand_dims(x, axis)

def bart(nargout, cmd, *args, return_str=False):
    """
    call bart from system command line
    """
    if type(nargout) != int or nargout < 0:
        print("Usage: bart(<nargout>, <command>, <arguements...>)")
        return None

    name = tempfile.NamedTemporaryFile().name

    nargin = len(args)
    infiles = [name + 'in' + str(idx) for idx in range(nargin)]
    in_str = ' '.join(infiles)

    for idx in range(nargin):
        writecfl(infiles[idx], args[idx])

    outfiles = [name + 'out' + str(idx) for idx in range(nargout)]
    out_str = ' '.join(outfiles)

    shell_str = 'bart ' + cmd + ' ' + in_str + ' ' + out_str
    print(shell_str)
    if not return_str:
        ERR = os.system(shell_str)
    else:
        try:
            strs = sp.check_output(shell_str, shell=True).decode()
            return strs
        except:
            ERR = True


    for elm in infiles:
        if os.path.isfile(elm + '.cfl'):
            os.remove(elm + '.cfl')
        if os.path.isfile(elm + '.hdr'):
            os.remove(elm + '.hdr')

    output = []
    for idx in range(nargout):
        elm = outfiles[idx]
        if not ERR:
            output.append(readcfl(elm))
        if os.path.isfile(elm + '.cfl'):
            os.remove(elm + '.cfl')
        if os.path.isfile(elm + '.hdr'):
            os.remove(elm + '.hdr')

    if ERR:
        print("Make sureyou install bart properly")
        raise Exception("Command exited with an error.")

    if nargout == 1:
        output = output[0]

    return output

def print_parameters(file, mode="a"):
    """
    print the trainable parameters info of tf graph
    """
    total_parameters = 0
    with open(file, mode) as f:
        with redirect_stdout(f):
            
            for variable in tf.trainable_variables():
                # shape is an array of tf.Dimension
                print("========layer=======")
                print(variable)
                shape = variable.get_shape()
                print("shape->",shape)
                print("shape->len",len(shape))
                variable_parameters = 1
                for dim in shape:
                    print("dim->", dim)
                    variable_parameters *= dim
                print(variable_parameters)
                total_parameters += variable_parameters
            print(total_parameters)

def norm_to_uint(inp, bit=8):
    maximum = np.max(inp)
    out = inp/maximum*np.power(2., bit)
    
    tp = np.uint8
    if bit == 16:
        tp= np.uint16
    if bit == 32:
        tp= np.uint16
    return out.astype(tp)

def psnr(img1, img2, bit=16, tobit=False):
    """
    calculate peak SNR, img1-true, img2-test
    """
    pixel_max = np.max(img2)
    
    if tobit:
        img1 = norm_to_uint(img1, bit)
        img2 = norm_to_uint(img2, bit)
        pixel_max = np.power(2., bit) - 1

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    pixel_max = pixel_max.astype(np.float64)

    return peak_signal_noise_ratio(img1, img2, data_range=pixel_max)


def ssim(img1, img2, bit=16, tobit=False):
    """
    calcualte similarity index between img1 and img2
    """
    
    scale = np.max(img2)
    if tobit:
        img1 = norm_to_uint(img1, bit)
        img2 = norm_to_uint(img2, bit)  
        scale = np.power(2., bit) - 1 
    img1.astype(np.float64)
    img2.astype(np.float64)
    scale.astype(np.float64)
    return structural_similarity(img1, img2, data_range=scale)

def get_lr(step, lr, warmup_steps=None, hidden_size=None):
    """
    not used, learning rate scheduler
    """
    if warmup_steps is not None and hidden_size is not None:
        lr_base = lr * 0.002 # for Adam correction
        ret = 5000. * hidden_size ** (-0.5) * \
          np.min([(step + 1) * warmup_steps ** (-1.5), (step + 1) ** (-0.5)])
        return ret * lr_base
    else:
        return lr

class LambdaWarmUpCosineScheduler:
    """
    note: use with a base_lr of 1.0
    """
    def __init__(self, warm_up_steps, lr_min, lr_max, lr_start, max_decay_steps, verbosity_interval=0):
        self.lr_warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay_steps = max_decay_steps
        self.last_lr = 0.
        self.verbosity_interval = verbosity_interval

    def schedule(self, n):
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: print(f"current step: {n}, recent lr-multiplier: {self.last_lr}")
        if n < self.lr_warm_up_steps:
            lr = (self.lr_max - self.lr_start) / self.lr_warm_up_steps * n + self.lr_start
            self.last_lr = lr
            return lr
        else:
            t = (n - self.lr_warm_up_steps) / (self.lr_max_decay_steps - self.lr_warm_up_steps)
            t = min(t, 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                    1 + np.cos(t * np.pi))
            self.last_lr = lr
            return lr

    def __call__(self, n):
        return self.schedule(n)

_RNG_SEED = None


def fix_rng_seed(seed):
    """
    Call this function at the beginning of program to fix rng seed within tensorpack.

    Args:
        seed (int):

    Note:
        See https://github.com/tensorpack/tensorpack/issues/196.

    Example:

        Fix random seed in both tensorpack and tensorflow.

    .. code-block:: python

            seed = 42
            utils.fix_rng_seed(seed)
            tesnorflow.set_random_seed(seed)
            # run trainer
    """
    global _RNG_SEED
    _RNG_SEED = int(seed)

def get_rng(obj=None):
    """
    Get a good RNG seeded with time, pid and the object.

    Args:
        obj: some object to use to generate random seed.
    Returns:
        np.random.RandomState: the RNG.
    """
    seed = (id(obj) + os.getpid() +
            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    if _RNG_SEED is not None:
        seed = _RNG_SEED
    return np.random.RandomState(seed)