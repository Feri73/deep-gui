import tensorflow as tf
import tensorflow.keras as keras

from utils import Config


class ScreenPreprocessor(keras.layers.Layer):
    def __init__(self, cfg: Config, **kwargs):
        self.grayscale = cfg['grayscale']
        self.crop_top_left = cfg['crop_top_left']
        self.crop_size = cfg['crop_size']
        self.resize_size = cfg['resize_size']
        self.scale_color = cfg['scale_color']
        self.equalize_background = cfg['equalize_background']
        self.contrast_alpha = cfg['contrast_alpha']

        super().__init__(**kwargs)

    def call(self, screens, **kwargs):
        screens = tf.cast(screens, tf.float32)
        if self.grayscale:
            screens = tf.image.rgb_to_grayscale(screens)
        screens = tf.image.crop_to_bounding_box(screens, *self.crop_top_left, *self.crop_size)
        screens_shape = (None, *[int(d) for d in screens.shape[1:]])
        if screens_shape[1:3] != self.resize_size:
            screens = tf.image.resize(screens, self.resize_size)
        screens_shape = (None, *[int(d) for d in screens.shape[1:]])
        if self.scale_color:
            if screens_shape[-1] != 1:
                raise AttributeError('cannot scale colored images.')
            axes = [1, 2, 3]
            screens = (screens - tf.reduce_min(screens, axis=axes, keep_dims=True)) / \
                      (tf.reduce_max(screens, axis=axes, keep_dims=True) -
                       tf.reduce_min(screens, axis=axes, keep_dims=True) + 1e-6)
        if self.equalize_background:
            if screens_shape[-1] != 1:
                raise AttributeError('cannot equalize background for colored images.')
            image_size = screens_shape[1] * screens_shape[2]
            color_sums = tf.reduce_sum(tf.cast(screens < .5, tf.float32), axis=[1, 2, 3])
            screens, _ = tf.map_fn(lambda elems:
                                   (tf.where(elems[1] < image_size / 2, 1 - elems[0], elems[0]), elems[1]),
                                   (screens, color_sums))
        if self.contrast_alpha > 0:
            if not screens_shape[-1] != 1:
                raise AttributeError('cannot change contrast of colored images.')
            screens = tf.sigmoid(self.contrast_alpha * (screens - .5))
        return screens


class EncodingRewardPredictor(keras.layers.Layer):
    def __init__(self, screen_encoder: keras.layers.Layer, reward_decoder: keras.layers.Layer, **kwarg):
        self.screen_encoder = screen_encoder
        self.reward_decoder = reward_decoder

        super().__init__(**kwarg)

    def call(self, screens, **kwargs):
        return self.reward_decoder(self.screen_encoder(screens))


class SimpleScreenEncoder(keras.layers.Layer):
    def __init__(self, cfg: Config, **kwargs):
        self.padding_type = cfg['padding_type']
        self.filter_nums = cfg['filter_nums']
        self.kernel_sizes = cfg['kernel_sizes']
        self.stride_sizes = cfg['stride_sizes']
        self.maxpool_sizes = cfg['maxpool_sizes']

        super().__init__(**kwargs)

        self.convs = []
        self.maxpools = []

    def build(self, input_shape):
        self.convs = [keras.layers.Conv2D(filters, kernel_size, stride, self.padding_type, activation=tf.nn.elu)
                      for filters, kernel_size, stride in zip(self.filter_nums, self.kernel_sizes, self.stride_sizes)]
        self.maxpools = [keras.layers.Lambda(lambda x: x)
                         if pool_size == 1 else keras.layers.MaxPool2D(pool_size, pool_size, self.padding_type)
                         for pool_size in self.maxpool_sizes]

    def call(self, screens, **kwargs):
        for conv, maxpool in zip(self.convs, self.maxpools):
            screens = maxpool(conv(screens))
        return screens


class SimpleRewardDecoder(keras.layers.Layer):
    def __init__(self, action_type_count: int, reward_categories_count: int, cfg: Config, **kwargs):
        super().__init__(**kwargs)

        self.prediction_shape = tuple(cfg['prediction_shape'])
        self.reward_categories_count = reward_categories_count

        self.action_type_count = action_type_count

        self.last_layer = None

        if self.reward_categories_count != 2:
            raise ValueError('For now only support binary rewards.')

    def build(self, input_shape):
        self.last_layer = keras.layers.Conv2D(self.action_type_count, 1, 1, 'VALID', activation=tf.nn.sigmoid)

    def call(self, parsed_screens, **kwargs):
        parsed_screens = self.last_layer(parsed_screens)
        assert tuple(map(int, (parsed_screens.shape[1:-1]))) == self.prediction_shape
        return parsed_screens


class SimpleRewardPredictor(EncodingRewardPredictor):
    def __init__(self, action_type_count: int, reward_categories_count: int, cfg: Config, **kwargs):
        prediction_shape = cfg['prediction_shape']
        screen_encoder_cfg = cfg['screen_encoder_configs']

        reward_decoder_cfg = {'prediction_shape': prediction_shape}

        super().__init__(SimpleScreenEncoder(screen_encoder_cfg),
                         SimpleRewardDecoder(action_type_count, reward_categories_count, reward_decoder_cfg), **kwargs)


class UNetScreenEncoder(keras.layers.Layer):
    def __init__(self, cfg: Config, **kwargs):
        self.output_layer_names = cfg['output_layer_names']
        self.inner_configs = cfg['inner_configs']
        super().__init__(**kwargs)
        self.encoder = None

    def build(self, input_shape):
        net = keras.applications.MobileNetV2(input_shape=tuple(int(x) for x in input_shape[1:]),
                                             include_top=False, **self.inner_configs)
        self.encoder = keras.Model(inputs=net.input,
                                   outputs=[net.get_layer(name).output for name in self.output_layer_names])

    def call(self, screens, **kwargs):
        screens = keras.applications.mobilenet.preprocess_input(screens)
        return self.encoder(screens)


class UNetRewardDecoder(keras.layers.Layer):
    def __init__(self, action_type_count: int, reward_categories_count: int, cfg: Config, **kwargs):
        self.filter_nums = cfg['filter_nums']
        self.kernel_sizes = cfg['kernel_sizes']
        self.stride_sizes = cfg['stride_sizes']
        self.padding_types = cfg['padding_types']
        self.prediction_shape = tuple(cfg['prediction_shape'])

        super().__init__(**kwargs)

        self.action_type_count = action_type_count
        self.reward_categories_count = reward_categories_count

        self.decoders = None
        self.last_layer = None

        if self.reward_categories_count != 2:
            raise NotImplementedError('For now only support binary rewards.')

    @staticmethod
    def deconv(filters: int, size: int, stride: int, padding: str, activation: str, normalization: bool):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2DTranspose(filters, size, stride, padding, activation=activation,
                                                   kernel_initializer=initializer, use_bias=False))
        if normalization:
            result.add(keras.layers.BatchNormalization())
        return result

    @staticmethod
    def val2list(val, size: int) -> list:
        return val if isinstance(val, list) else [val] * size

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.filter_nums = self.val2list(self.filter_nums, len(input_shape) - 1)
        self.kernel_sizes = self.val2list(self.kernel_sizes, len(input_shape))
        self.stride_sizes = self.val2list(self.stride_sizes, len(input_shape))
        self.padding_types = self.val2list(self.padding_types, len(input_shape))
        self.decoders = [self.deconv(filter, size, stride, padding, 'relu', True) for filter, size, stride, padding in
                         zip(self.filter_nums, self.kernel_sizes[:-1], self.stride_sizes[:-1], self.padding_types[:-1])]
        self.last_layer = self.deconv(self.action_type_count, self.kernel_sizes[-1], self.stride_sizes[-1],
                                      self.padding_types[-1], 'sigmoid', False)

    def call(self, parsed_screens_layers, **kwargs):
        hidden = parsed_screens_layers[0]
        skips = parsed_screens_layers[1:]
        for decoder, skip in zip(self.decoders, skips):
            hidden = decoder(hidden)
            concat = tf.keras.layers.Concatenate()
            hidden = concat([hidden, skip])
        result = self.last_layer(hidden)
        assert tuple(map(int, (result.shape[1:-1]))) == self.prediction_shape
        return result


class UNetRewardPredictor(EncodingRewardPredictor):
    def __init__(self, action_type_count: int, reward_categories_count: int, cfg: Config, **kwargs):
        screen_encoder_cfg = cfg['screen_encoder_configs']
        reward_decoder_cfg = cfg['reward_decoder_configs']
        reward_decoder_cfg['prediction_shape'] = cfg['prediction_shape']

        super().__init__(UNetScreenEncoder(screen_encoder_cfg),
                         UNetRewardDecoder(action_type_count, reward_categories_count, reward_decoder_cfg), **kwargs)


class RandomRewardPredictor(keras.layers.Layer):
    def __init__(self, action_type_count: int, reward_categories_count: int, cfg: Config, **kwargs):
        self.prediction_shape = cfg['prediction_shape']

        super().__init__(**kwargs)

        self.action_type_count = action_type_count
        self.reward_categories_count = reward_categories_count

        if self.reward_categories_count != 2:
            raise NotImplementedError('For now only support binary rewards.')

    def call(self, screens, **kwargs):
        return tf.random.uniform((screens.shape[0], *self.prediction_shape, self.action_type_count))
