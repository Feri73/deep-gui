from typing import Tuple, Callable

import numpy as np
import tensorflow as tf

from tf1 import ArrayTuple
from reinforcement_learning import RLModel
from utils import Config

TensorTuple = Tuple[tf.Tensor, ...]


# assumes reward is either 0 or 1
class SSBRDQN(RLModel):
    def __init__(self, q_layer, q_user: Callable, q_layer_output_shape: Tuple[int, ...], cfg: Config):
        self.q_layer = q_layer
        self.q_user = q_user
        self.q_layer_output_shape = q_layer_output_shape

        self.class_normalization_window = cfg['class_normalization_window']

    def binary_cross_entropy(self, rewards_b, preds_b, high_reward_weight):
        return 2 * tf.reduce_mean(tf.keras.backend.binary_crossentropy(rewards_b, preds_b) *
                                  (1 - rewards_b + 2 * high_reward_weight * rewards_b - high_reward_weight))

    def calc_next_action(self, env_states_bt: tf.Tensor, bef_actions_bt: tf.Tensor, bef_rewards_bt: tf.Tensor,
                         bef_model_states_eb: TensorTuple) -> Tuple[tf.Tensor, TensorTuple]:
        q_b = self.q_layer(env_states_bt[:, 0])

        return self.q_user(tf.expand_dims(q_b, axis=0)), (tf.expand_dims(q_b, axis=0),)

    def calc_loss(self, actions_bt: tf.Tensor, rewards_bt: tf.Tensor, model_states_ebt: TensorTuple,
                  finished_b: tf.Tensor) -> Tuple[tf.Tensor, TensorTuple]:
        high_reward_ratio_sum = tf.Variable(0.0, trainable=False)
        high_reward_ratio_size = tf.Variable(0.0, trainable=False)
        high_reward_ratio_prev = tf.Variable(-1.0, trainable=False)

        is_reset_time = tf.equal(high_reward_ratio_size, tf.constant(float(self.class_normalization_window)))
        high_reward_ratio_mean = high_reward_ratio_sum / high_reward_ratio_size

        update_prev_op = high_reward_ratio_prev.assign(tf.where(is_reset_time, high_reward_ratio_mean,
                                                                high_reward_ratio_prev))
        update_ratio_sum_op = high_reward_ratio_sum.assign(
            tf.reduce_sum(rewards_bt) + tf.where(is_reset_time, tf.constant(0.0), high_reward_ratio_sum))
        update_ratio_size_op = high_reward_ratio_size.assign(
            tf.cast(tf.size(rewards_bt), tf.float32) +
            tf.where(is_reset_time, tf.constant(0.0), high_reward_ratio_size))

        high_reward_weight = tf.where(high_reward_ratio_prev >= 0,
                                      tf.minimum(1 - high_reward_ratio_prev, tf.constant(.99)), tf.constant(.5))

        q_b = model_states_ebt[0][:, 0]
        reward_b = rewards_bt[:, 0]
        actions_b = actions_bt[:, 0]

        action_onehots_b = tf.one_hot(actions_b, q_b.shape[1])

        with tf.control_dependencies([update_prev_op]):
            with tf.control_dependencies([update_ratio_sum_op]):
                with tf.control_dependencies([update_ratio_size_op]):
                    loss = self.binary_cross_entropy(reward_b, tf.reduce_sum(q_b * action_onehots_b,
                                                                             axis=list(range(1, len(q_b.shape)))),
                                                     high_reward_weight)
        return loss, (loss, high_reward_weight, high_reward_ratio_size, high_reward_ratio_prev)

    def get_default_action(self) -> np.ndarray:
        return np.array(0)

    def get_default_reward(self) -> float:
        return 0.0

    def get_default_states(self) -> ArrayTuple:
        return np.zeros(self.q_layer_output_shape),
