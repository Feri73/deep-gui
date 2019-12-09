# import multiprocessing as mp
from typing import Tuple

# import gensim
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models


# class Word2Vec:
#     def __init__(self, file_path: str):
#         model = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)
#         words = list(model.vocab.keys())
#         self.word2index_dict = {word: word_i for word_i, word in enumerate(words)}
#         self.mp_arrays = []
#         for word in words:
#             self.mp_arrays += [mp.Array('f', model[word].tolist())]
#
#     def __call__(self, *args, **kwargs) -> tf.Tensor:
#         words = args[0]
#         result = []
#         for word in words:
#             result += [self.mp_arrays[self.word2index_dict[word]][:]]
#         return tf.constant(result)


class ScreenPreprocessor(layers.Layer):
    def __init__(self, crop_top_left: Tuple[int, int], crop_size: Tuple[int, int], new_shape: Tuple[int, int]):
        super().__init__()
        self.new_shape = new_shape
        self.crop_top_left = crop_top_left
        self.crop_size = crop_size

    @tf.function
    def call(self, input: tf.Tensor) -> tf.Tensor:
        # do i have to divide it by 255 or do something in this line?
        # use crop_and_resize instead
        res = tf.image.rgb_to_grayscale(tf.image.resize(
            tf.image.crop_to_bounding_box(tf.cast(input, tf.float32) / 255.0,
                                          self.crop_top_left[0], self.crop_top_left[1],
                                          self.crop_size[0], self.crop_size[1]), self.new_shape))
        # if res.__class__.__name__ == 'EagerTensor':
        #     plt.imshow(res[0, :, :, 0])
        #     plt.show()
        return res
        # return tf.image.resize(tf.cast(input, tf.float32) / 255.0, self.new_shape)


# change to NCHW and check the speed
class ScreenEncoder(models.Sequential):
    def __init__(self, crop_top_left: Tuple[int, int], crop_size: Tuple[int, int],
                 screen_new_shape: Tuple[int, int], output_size: int):
        super().__init__([
            ScreenPreprocessor(crop_top_left, crop_size, screen_new_shape),
            # make sure that weights here are learnable
            # alpha should be in config
            # give this more thought and pick the best architecture
            #   also, consider changing this to simple 2-3 stacked cnns, because it takes much less that way
            keras.applications.MobileNetV2(input_shape=(*screen_new_shape, 1), include_top=False, weights=None,
                                           alpha=.9),
            layers.Flatten(),
            # no activation here
            layers.Dense(output_size)
        ])


# any kind of normalization?
class PolicyGenerator(models.Sequential):
    def __init__(self, output_shape: Tuple[int, int, int]):
        output_screen_size = output_shape[:-1]
        output_channels_size = output_shape[-1]
        super().__init__([
            # this should be better. i should not use the dense in ScreenEncoder and this dense here. Instead, i
            #   should use cnn to reshape the output of ScreenEncoder to the correct shape and then proceed with conv2dt
            layers.Dense(np.prod(output_screen_size) // 64 * 32, activation='elu'),
            # there is a bug in tf 2.0 that does not work with np.int32 integers in eager mode
            layers.Reshape((*tuple([int(x) for x in np.array(output_screen_size) // 8]), 32)),
            layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='elu'),
            layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='elu'),
            # do i need activation here?
            layers.Conv2DTranspose(filters=output_channels_size, kernel_size=3,
                                   strides=2, padding='same', activation='elu'),
            layers.Flatten()
        ])


class ValueEstimator(layers.Layer):
    def __init__(self, value):
        super().__init__()
        # read the manual one more time to see what is the different to pass tf.constant vs python float
        self.statc_vals = ValueEstimator.get_static_vals(tf.constant(value, dtype=tf.float32))
        # self.inner_network = layers.Dense(1, activation='sigmoid')

    @staticmethod
    def get_static_vals(value):
        @tf.custom_gradient
        def static_vals(input: tf.Tensor) -> tf.Tensor:
            def grad(y):
                return tf.zeros_like(input)

            return tf.ones((tf.shape(input)[0], 1)) * value, grad
        return static_vals

    @tf.function
    def call(self, input: tf.Tensor) -> tf.Tensor:
        return self.statc_vals(input)
        # return self.inner_network(input)
