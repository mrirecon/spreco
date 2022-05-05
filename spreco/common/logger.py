# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import time
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.core.util import event_pb2

class logger(object):
    """
    Dumps key/value pairs into TensorBoard's numeric format.

    Credit: OpenAI baselines. Licensed under MIT License.
    Copyright (c) 2017 OpenAI (http://openai.com)
    """

    def __init__(self, dir):
        os.makedirs(dir, exist_ok=True)
        self.dir = dir
        self.step = 1
        prefix = "events"
        path = osp.join(osp.abspath(dir), prefix)
        import tensorflow.compat.v1 as tf
        try:
            from tensorflow.python import _pywrap_events_writer as event_writer # v2
        except:
            print("using tensorflow 1.x")
            from tensorflow.python import pywrap_tensorflow as event_writer# v1
        else:
            print("using tensorflow 2.x")

        from tensorflow.core.util import event_pb2
        from tensorflow.python.util import compat

        self.tf = tf
        self.event_pb2 = event_pb2
        self.event_writer = event_writer
        self.writer = event_writer.EventsWriter(compat.as_bytes(path))

    def writekvs(self, kvs):
        def summary_val(k, v):
            kwargs = {"tag": k, "simple_value": float(v)}
            return self.tf.Summary.Value(**kwargs)

        summary = self.tf.Summary(value=[summary_val(k, v) for k, v in kvs.items()])
        event = self.event_pb2.Event(wall_time=time.time(), summary=summary)
        event.step = self.step
        self.writer.WriteEvent(event)
        self.writer.Flush()
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