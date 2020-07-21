from functools import partial
from typing import Callable

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


class BufferLogger(keras.layers.Layer):
    def __init__(self, freq: int, handler: Callable, aggregate: bool, **kwargs):
        super().__init__(**kwargs)

        self.freq = freq
        self.handler = handler
        self.aggregate = aggregate

        self.log_values = []
        self.log_step = None
        self.dependency = None

    def build(self, input_shape):
        self.log_step = self.add_weight(shape=(), trainable=False,
                                        initializer=lambda *args, **kwargs: 0, dtype=tf.int32)
        self.dependency = self.add_weight(shape=(), trainable=False,
                                          initializer=lambda *args, **kwargs: 0, dtype=tf.int32)

    def call(self, inputs, **kwargs):
        self.log_step = tf.assign_add(self.log_step, 1)

        if self.aggregate:
            with tf.control_dependencies([tf.py_func(lambda v: self.log_values.append(v), (inputs,), [])]):
                with tf.control_dependencies([tf.py_func(partial(cond_flush, buffer_logger=self),
                                                         (self.log_step,), ())]):
                    self.dependency = tf.identity(self.dependency)
        else:
            with tf.control_dependencies([tf.py_func(partial(cond_flush, buffer_logger=self),
                                                     (self.log_step, inputs), ())]):
                self.dependency = tf.identity(self.dependency)

        return self.dependency


def cond_flush(step: int, values: np.ndarray = None, buffer_logger: 'BufferLogger' = None) -> None:
    if values is None:
        values = buffer_logger.log_values
    if step % buffer_logger.freq == 0:
        buffer_logger.handler(values)
        buffer_logger.log_values = []
