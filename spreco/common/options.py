from . import utils
import numpy as np
from functools import partial
from copy import deepcopy

class DATA_CHNS():
    """

    """
    MAG  = 'MAG'  # for magnitude images
    CPLX = 'CPLX' # for images with real and imaginary channels 
    RGB  = 'RGB'  # for images with red, green, blue channnels

    @staticmethod
    def get():
        return [
            DATA_CHNS.MAG,
            DATA_CHNS.CPLX,
            DATA_CHNS.RGB
        ]

class MODELS():
    """

    """
    NCSN         = 'NCSN'
    SDE          = 'SDE'
    
    @staticmethod
    def get():
        return [
            MODELS.NCSN,
            MODELS.SDE
        ]


class parts():
    """
    class to get parts_func for building pipe
    """

    dict_func = {}
    
    dict_func['abs']         = abs
    dict_func['squeeze']     = np.squeeze
    dict_func['slice_image'] = utils.slice_image  # args -> (inp, shape)
    dict_func['npz_loader']  = utils.npz_loader   # args -> (filename, key='rss')
    dict_func['cplx2float']  = utils.cplx2float   # args -> (cplx_in)
    dict_func['float2cplx']  = utils.float2cplx   # args -> (float_in)
    dict_func['cfl_loader']  = utils.readcfl      # args -> (filename)
    dict_func['normalize_with_max'] = utils.normalize_with_max    # args -> (x, axis=(0,1), data_chns='CPLX')
    dict_func['transform']          = utils.transform             # args -> (inp)
    dict_func['random_rescale']     = utils.random_rescale        # args -> (x, vmin, vmax)
    dict_func['randint']            = utils.randint               # args -> (x, nr_levels, dtype)
    dict_func['randfloat']          = utils.randfloat             # args -> (x, lower=0, upper=1, eps, T) interval [eps, T)
    dict_func['affine']             = utils.affine                # args -> (x, a, b) 
    dict_func['expand_dims']        = utils.expand_dims           # args -> (x, axis) 

    @staticmethod
    def parse(parts_ll):
        cl = deepcopy(parts_ll)
        ret = []
        for parts_l in cl:
            l = []
            for part in parts_l:
                func_key = part.pop('func', None)

                if func_key is None:
                    raise Exception("check the definition of part in config file")
                else:
                    try:
                        func = parts.dict_func[func_key]
                    except:
                        raise Exception("the function %s is not defined"% func_key)

                l.append(partial(func, **part))
            ret.append(l)
        return ret