from spreco.common.options import MODELS
from spreco.common import utils
from spreco import bart_tf

import os
import tensorflow.compat.v1 as tf

class exporter():
    """
    """

    def __init__(self, log, meta, path, name, sigma_type=None, placeholder_name=None):
        self.log    = log
        self.meta   = meta
        self.path   = path
        self.name   = name
        self.sigma_type = sigma_type # for sde model
        self.placeholder_name = placeholder_name
        self.init_and_export()

    def init_and_export(self):

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

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

        self.model = selected_class(config)
        self.model.config['sigma_type'] = self.sigma_type

        self.model.init(mode=2, batch_size=None)

        saver      = tf.train.Saver()
        iconfig = tf.ConfigProto()
        iconfig.gpu_options.allow_growth=True
        sess = tf.Session(config=iconfig)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, os.path.join(self.log, self.meta))
        
        bart_tf.tf1_export_graph(self.path, session=sess,
                                name=self.name,
                                inputs=[self.model.x, self.model.t],
                                outputs=[self.model.default_out], attach_gradients=False)
