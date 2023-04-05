# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
import time
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tensorflow.core.util import event_pb2
import logging
import os
import os.path
import shutil
import sys
from datetime import datetime
from six.moves import input
from termcolor import colored

__all__ = ['set_logger_dir', 'auto_set_dir', 'get_logger_dir', 'tb_logger']

class tb_logger(object):
    """
    Dumps key/value pairs into TensorBoard's numeric format.

    Credit: OpenAI baselines. Licensed under MIT License.
    Copyright (c) 2017 OpenAI (http://openai.com)
    """

    def __init__(self, dir):
        os.makedirs(dir, exist_ok=True)
        self.dir = dir
        self.step = 1
        self.test_step = 1
        prefix = "events"
        path = osp.join(osp.abspath(dir), prefix)
        import tensorflow.compat.v1 as tf
        try:
            from tensorflow.python.client import _pywrap_events_writer as event_writer # v2
        except:
            #print("using tensorflow 1.x")
            from tensorflow.python import pywrap_tensorflow as event_writer# v1
        else:
            pass
            #print("using tensorflow 2.x")

        from tensorflow.core.util import event_pb2
        from tensorflow.python.util import compat

        self.tf = tf
        self.event_pb2 = event_pb2
        self.event_writer = event_writer
        self.writer = event_writer.EventsWriter(compat.as_bytes(path))

    def writekvs(self, kvs, epoch=None):
        def summary_val(k, v):
            kwargs = {"tag": k, "simple_value": float(v)}
            return self.tf.Summary.Value(**kwargs)

        summary = self.tf.Summary(value=[summary_val(k, v) for k, v in kvs.items()])
        event = self.event_pb2.Event(wall_time=time.time(), summary=summary)
        if epoch is not None:
            event.step = epoch
        else:
            event.step = self.step
        self.writer.WriteEvent(event)
        self.writer.Flush()
        if epoch is not None:
            pass
        else:
            self.step += 1

    def close(self):
        if self.writer:
            self.writer.Close()
            self.writer = None

    @staticmethod
    def make_image(tensor, scalar):
        """Convert an numpy representation image to Image protobuf"""
        from PIL import Image
        if len(tensor.shape) == 4:
            _, height, width, channel = tensor.shape
        elif len(tensor.shape) == 3:
            height, width, channel = tensor.shape
        elif len(tensor.shape) == 2:
            height, width = tensor.shape
            channel = 1
        tensor = (tensor*scalar).astype(np.uint8)
        image = Image.fromarray(tensor)
        import io
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height, width=width, colorspace=channel, encoded_image_string=image_string)

    def log_image(self, im, tag, scalar=1, step=0):
        im = self.make_image(im, scalar)

        summary = [tf.Summary.Value(tag=tag, image=im)]
        summary = tf.Summary(value=summary)
        event = event_pb2.Event(summary=summary)
        event.step = step
        self.writer.WriteEvent(event)
        self.writer.Flush()


class _MyFormatter(logging.Formatter):
    def format(self, record):
        date = colored('[%(asctime)s @%(filename)s:%(lineno)d]', 'green')
        msg = '%(message)s'
        if record.levelno == logging.WARNING:
            fmt = date + ' ' + colored('WRN', 'red', attrs=['blink']) + ' ' + msg
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            fmt = date + ' ' + colored('ERR', 'red', attrs=['blink', 'underline']) + ' ' + msg
        elif record.levelno == logging.DEBUG:
            fmt = date + ' ' + colored('DBG', 'yellow', attrs=['blink']) + ' ' + msg
        else:
            fmt = date + ' ' + msg
        if hasattr(self, '_style'):
            # Python3 compatibility
            self._style._fmt = fmt
        self._fmt = fmt
        return super(_MyFormatter, self).format(record)


def _getlogger():
    # this file is synced to "dataflow" package as well
    package_name = "dataflow" if __name__.startswith("dataflow") else "tensorpack"
    logger = logging.getLogger(package_name)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_MyFormatter(datefmt='%m%d %H:%M:%S'))
    logger.addHandler(handler)
    return logger


_logger = _getlogger()
_LOGGING_METHOD = ['info', 'warning', 'error', 'critical', 'exception', 'debug', 'setLevel', 'addFilter']
# export logger functions
for func in _LOGGING_METHOD:
    locals()[func] = getattr(_logger, func)
    __all__.append(func)
# 'warn' is deprecated in logging module
warn = _logger.warning
__all__.append('warn')


def _get_time_str():
    return datetime.now().strftime('%m%d-%H%M%S')


# globals: logger file and directory:
LOG_DIR = None
_FILE_HANDLER = None


def _set_file(path):
    global _FILE_HANDLER
    if os.path.isfile(path):
        backup_name = path + '.' + _get_time_str()
        shutil.move(path, backup_name)
        _logger.info("Existing log file '{}' backuped to '{}'".format(path, backup_name))  # noqa: F821
    hdl = logging.FileHandler(
        filename=path, encoding='utf-8', mode='w')
    hdl.setFormatter(_MyFormatter(datefmt='%m%d %H:%M:%S'))

    _FILE_HANDLER = hdl
    _logger.addHandler(hdl)
    _logger.info("Argv: " + ' '.join(sys.argv))


def set_logger_dir(dirname, action=None):
    """
    Set the directory for global logging.

    Args:
        dirname(str): log directory
        action(str): an action of ["k","d","q"] to be performed
            when the directory exists. Will ask user by default.

                "d": delete the directory. Note that the deletion may fail when
                the directory is used by tensorboard.

                "k": keep the directory. This is useful when you resume from a
                previous training and want the directory to look as if the
                training was not interrupted.
                Note that this option does not load old models or any other
                old states for you. It simply does nothing.

    """
    dirname = os.path.normpath(dirname)
    global LOG_DIR, _FILE_HANDLER
    if _FILE_HANDLER:
        # unload and close the old file handler, so that we may safely delete the logger directory
        _logger.removeHandler(_FILE_HANDLER)
        del _FILE_HANDLER

    def dir_nonempty(dirname):
        # If directory exists and nonempty (ignore hidden files), prompt for action
        return os.path.isdir(dirname) and len([x for x in os.listdir(dirname) if x[0] != '.'])

    if dir_nonempty(dirname):
        if not action:
            _logger.warning("""\
Log directory {} exists! Use 'd' to delete it. """.format(dirname))
            _logger.warning("""\
If you're resuming from a previous run, you can choose to keep it.
Press any other key to exit. """)
        while not action:
            action = input("Select Action: k (keep) / d (delete) / q (quit):").lower().strip()
        act = action
        if act == 'b':
            backup_name = dirname + _get_time_str()
            shutil.move(dirname, backup_name)
            info("Directory '{}' backuped to '{}'".format(dirname, backup_name))  # noqa: F821
        elif act == 'd':
            shutil.rmtree(dirname, ignore_errors=True)
            if dir_nonempty(dirname):
                shutil.rmtree(dirname, ignore_errors=False)
        elif act == 'n':
            dirname = dirname + _get_time_str()
            info("Use a new log directory {}".format(dirname))  # noqa: F821
        elif act == 'k':
            pass
        else:
            raise OSError("Directory {} exits!".format(dirname))
    LOG_DIR = dirname
    from .fs import mkdir_p
    mkdir_p(dirname)
    _set_file(os.path.join(dirname, 'log.log'))


def auto_set_dir(action=None, name=None):
    """
    Use :func:`logger.set_logger_dir` to set log directory to
    "./train_log/{scriptname}:{name}". "scriptname" is the name of the main python file currently running"""
    mod = sys.modules['__main__']
    basename = os.path.basename(mod.__file__)
    auto_dirname = os.path.join('train_log', basename[:basename.rfind('.')])
    if name:
        auto_dirname += '_%s' % name if os.name == 'nt' else ':%s' % name
    set_logger_dir(auto_dirname, action=action)


def get_logger_dir():
    """
    Returns:
        The logger directory, or None if not set.
        The directory is used for general logging, tensorboard events, checkpoints, etc.
    """
    return LOG_DIR
