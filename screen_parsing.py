from typing import Tuple

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from utils import normalized_columns_initializer


# a much better design is required for all classes here
# change to NCHW and check the speed

class ScreenEncoder:
    def __init__(self, crop_top_left: Tuple[int, int], crop_size: Tuple[int, int], screen_new_shape: Tuple[int, int],
                 contrast_alpha: float, padding_type: str,
                 kernel_sizes: Tuple, filter_nums: Tuple, stride_sizes: Tuple, maxpool_sizes: Tuple):
        self.crop_top_left = crop_top_left
        self.crop_size = crop_size
        self.screen_new_shape = screen_new_shape
        self.contrast_alpha = contrast_alpha
        self.padding_type = padding_type
        self.kernel_sizes = kernel_sizes
        self.filter_nums = filter_nums
        self.stride_sizes = stride_sizes
        self.maxpool_sizes = maxpool_sizes

    # any kind of max pooling here?
    def __call__(self, inputs):
        env_states, actions, rewards, model_states = inputs
        # use crop_and_resize instead
        # add an option for grayscale
        # before i had to /255.0 here but now i don't. why's that? make sure the input to the network is in (0, 1)
        hidden = tf.image.resize(
            tf.image.crop_to_bounding_box(tf.image.rgb_to_grayscale(tf.cast(env_states[0], tf.float32)),
                                          self.crop_top_left[0], self.crop_top_left[1],
                                          self.crop_size[0], self.crop_size[1]), self.screen_new_shape)
        hidden = tf.where(tf.reduce_sum(tf.cast(hidden < .5, tf.float64)) < tf.size(hidden) / 2, 1 - hidden, hidden)
        if self.contrast_alpha > 0:
            hidden = tf.sigmoid(self.contrast_alpha * (hidden - .5))
        hidden = (hidden - tf.reduce_min(hidden)) / (tf.reduce_max(hidden) - tf.reduce_min(hidden))
        # hidden = tf.concat([tf.repeat(hidden[:1, :, :], self.padding[0], axis=0), hidden,
        #                     tf.repeat(hidden[-1:, :, :], self.padding[1], axis=0)], axis=0)
        # hidden = tf.concat([tf.repeat(hidden[:, :1, :], self.padding[2], axis=1), hidden,
        #                     tf.repeat(hidden[:, -1:, :], self.padding[3], axis=1)], axis=1)
        self.processed_screen = tf.expand_dims(hidden, axis=0)
        for k, f, s, mx in zip(self.kernel_sizes, self.filter_nums, self.stride_sizes, self.maxpool_sizes):
            hidden = slim.conv2d(activation_fn=tf.nn.elu, inputs=hidden,
                                 num_outputs=f, kernel_size=k, stride=s, padding=self.padding_type)
            if np.any(np.array(mx) > 1):
                hidden = tf.nn.max_pool2d(hidden, mx, mx, self.padding_type)
        return tf.expand_dims(hidden, axis=0), ()

    def get_processed_screen(self) -> tf.Tensor:
        return self.processed_screen


# any kind of normalization?
class PolicyGenerator:
    def __init__(self, action_shape: Tuple[int, int, int], padding_type: str,
                 kernel_sizes: Tuple, filter_nums: Tuple, deconv_output_shapes: Tuple):
        self.action_shape = action_shape
        self.padding_type = padding_type
        self.kernel_sizes = kernel_sizes
        self.filter_nums = filter_nums
        self.deconv_output_shapes = deconv_output_shapes

    def __call__(self, inputs):
        def deconv(activation, input, filters, kernel, output_shape):
            output_shape = np.array(output_shape)
            input_shape = np.array([input.shape[1], input.shape[2]])
            stride = output_shape // input_shape
            res = slim.conv2d_transpose(activation_fn=activation, inputs=input, num_outputs=filters,
                                        kernel_size=kernel, stride=stride, padding=self.padding_type)
            assert np.all(res.shape[-3:-1] == output_shape)
            return res

        hidden = inputs[0][0]
        for k, f, o in zip(self.kernel_sizes[:-1], self.filter_nums, self.deconv_output_shapes):
            hidden = deconv(tf.nn.elu, hidden, f, k, o)
        # any use of normalized_columns_initializer like ajuliani did?
        if len(self.kernel_sizes) > 0:
            hidden = deconv(None, hidden, self.action_shape[-1], self.kernel_sizes[-1], self.action_shape[:-1])
        else:
            hidden = deconv(None, hidden, self.action_shape[-1], 1, self.action_shape[:-1])
        res = tf.expand_dims(slim.flatten(hidden), axis=0)
        assert tuple(map(int, (hidden.shape[1:]))) == self.action_shape
        return res, ()


# better value estimator
class ValueEstimator:
    def __init__(self, value, value_range: Tuple[float, float] = None, padding_type: str = None):
        self.value = value
        self.value_range = value_range
        self.padding_type = padding_type
        if value is not None:
            # read the manual one more time to see what is the different to pass tf.constant vs python float
            self.statc_vals = ValueEstimator.get_static_vals(tf.constant(value, dtype=tf.float32))

    @staticmethod
    def get_static_vals(value):
        @tf.custom_gradient
        def static_vals(input: tf.Tensor) -> tf.Tensor:
            def grad(y):
                return tf.zeros_like(input)

            return tf.ones((tf.shape(input)[0], tf.shape(input)[1], 1)) * value, grad

        return static_vals

    def __call__(self, inputs):
        if self.value is None:
            hidden = slim.flatten(slim.conv2d(activation_fn=None,
                                              inputs=inputs[0][0], num_outputs=1,
                                              kernel_size=1, stride=1, padding=self.padding_type))
            res = slim.fully_connected(hidden, 1, activation_fn=None if self.value_range is None else tf.nn.sigmoid,
                                       weights_initializer=normalized_columns_initializer(1.0), biases_initializer=None)
            if self.value_range is not None:
                res = res * (self.value_range[1] - self.value_range[0]) + self.value_range[0]
            return tf.expand_dims(res, 0), ()
        return self.statc_vals(inputs[0]), ()
