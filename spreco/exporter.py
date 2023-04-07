from spreco.common.options import MODELS
from spreco.common import utils
from spreco import bart_tf

import os
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

class exporter():
    """
    """

    def __init__(self, log, meta, path, name, sigma_type=None, default_out=True, **kwargs):
        self.log    = log
        self.meta   = meta
        self.path   = path
        self.name   = name
        self.sigma_type = sigma_type # for sde model
        self.init(default_out, kwargs)

    def init(self, default, kwargs):

        try:
            config = utils.load_config(os.path.join(self.log, 'config.yaml'))
        except:
            raise ValueError("Config file is missing!")
        
        if config['model'] == MODELS.NCSN:
            from spreco.model.ncsn import ncsn as selected_class

        elif config['model'] == MODELS.SDE:
            from spreco.model.sde import sde as selected_class

        elif config['model'] == MODELS.PIXELCNN:
            from spreco.model.pixelcnn import pixelcnn as selected_class

        else:
            raise Exception("Currently, this model is not implemented!")
        
        if 'sigma_max' in kwargs.keys() and kwargs['sigma_max'] <= config['sigma_max']:
            config['sigma_max'] = kwargs['sigma_max']
        else:
            raise ValueError('Sigma max is not acceptiable for score net!')
    
        if 'sigma_min' in kwargs.keys() and kwargs['sigma_min'] >= config['sigma_min']:
            config['sigma_min'] = kwargs['sigma_min']
        else:
            raise ValueError('Sigma min is not acceptiable for score net!')


        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
        
        self.model = selected_class(config)
        self.model.config['sigma_type'] = self.sigma_type

        self.model.init(mode=2, default_out=default)

        if default:
            self.restore()
            bart_tf.tf1_export_graph(self.path, session=self.sess,
                                name=self.name,
                                inputs=[self.model.x, self.model.t],
                                outputs=[self.model.default_out], attach_gradients=False)

    def restore(self):
        saver      = tf.train.Saver()
        iconfig = tf.ConfigProto()
        iconfig.gpu_options.allow_growth=True
        self.sess = tf.Session(config=iconfig)
        self.sess.run(tf.global_variables_initializer())
        saver.restore(self.sess, os.path.join(self.log, self.meta))

    def export(self, inputs, outputs):
        self.restore()
        bart_tf.tf1_export_graph(self.path, session=self.sess,
                                name=self.name,
                                inputs=inputs,
                                outputs=outputs, attach_gradients=False)