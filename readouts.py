from typing import Callable

import tensorflow as tf
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from utils import Config

prediction_normalizer = None
action_prob_coeffs = None


def index_to_action(index: tf.Tensor, preds: tf.Tensor) -> tf.Tensor:
    shape = preds.shape[1:]
    y = tf.cast(index // np.prod(shape[1:]), tf.int32)
    x = tf.cast((index // shape[2]) % shape[1], tf.int32)
    type = tf.cast(index % shape[2], tf.int32)
    return tf.transpose(tf.concat([[y], [x], [type]], axis=0))


def most_probable_weighted_policy_user(logits: tf.Tensor) -> tf.Tensor:
    if prediction_normalizer is None:
        return tf.argmax(tf.distributions.Multinomial(1.0, logits=logits).sample(), axis=-1)
    return tf.argmax(tf.distributions.Multinomial(1.0, probs=prediction_normalizer(logits, axis=-1)).sample(), axis=-1)


def better_reward_to_action(preds: tf.Tensor) -> tf.Tensor:
    preds = preds * action_prob_coeffs
    preds_f = tf.reshape(preds, (-1, np.prod(preds.shape[1:])))
    return index_to_action(most_probable_weighted_policy_user(preds_f), preds)


def worse_reward_to_action(preds: tf.Tensor) -> tf.Tensor:
    return better_reward_to_action(-preds + 1)


def least_certain_reward_to_action(preds: tf.Tensor) -> tf.Tensor:
    return better_reward_to_action(.5 - tf.abs(-preds + .5))


def most_certain_reward_to_action(preds: tf.Tensor) -> tf.Tensor:
    return better_reward_to_action(tf.abs(-preds + .5))


def random_reward_to_action(preds: tf.Tensor) -> tf.Tensor:
    return better_reward_to_action(tf.ones_like(preds, dtype=tf.float32))


class PredictionClusterer:
    def __init__(self, cfg: Config):
        self.start_clickable_threshold = cfg['start_clickable_threshold']
        self.clickable_threshold_speed = cfg['clickable_threshold_speed']
        self.clickable_threshold_speed_step = cfg['clickable_threshold_speed_step']
        self.speed_steps_per_clickable_threshold_reset = cfg['speed_steps_per_clickable_threshold_reset']
        self.distance_threshold = cfg['distance_threshold']
        self.cluster_count_threshold = cfg['cluster_count_threshold']

        self.callbacks = []
        self.steps = 0

    def add_callback(self, callback: Callable) -> None:
        self.callbacks.append(callback)

    def __call__(self, preds: tf.Tensor) -> tf.Tensor:
        clickable_threshold = self.start_clickable_threshold - self.clickable_threshold_speed * \
                              ((self.steps // self.clickable_threshold_speed_step) %
                               self.speed_steps_per_clickable_threshold_reset)
        self.steps += 1
        if preds.shape[0] > 1:
            raise NotImplementedError('cluster reward is not implemented for batch size > 1.')
        preds_old = preds
        type_count = preds.shape[-1]

        all_clickables = []
        all_clusters = []
        all_valid_clusters_nums = []
        for type in range(type_count):
            all_clickables.append([])
            all_clusters.append([])
            all_valid_clusters_nums.append([])
            preds = preds_old[0, :, :, type]
            clickables = tf.cast(tf.where(preds > clickable_threshold), tf.int32)
            if len(clickables) == 0 or len(clickables) == 1:
                all_clickables[-1] = clickables
                continue
            clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=self.distance_threshold,
                                                compute_full_tree=True, linkage='single')
            clusters = clusterer.fit_predict(clickables)
            clusters_nums, clusters_counts = np.unique(clusters, axis=0, return_counts=True)
            valid_clusters_nums = clusters_nums[clusters_counts >= self.cluster_count_threshold]
            if len(valid_clusters_nums) == 0:
                all_clickables[-1] = clickables
                continue
            all_clickables[-1] = clickables
            all_clusters[-1] = clusters
            all_valid_clusters_nums[-1] = valid_clusters_nums

        if sum(map(len, all_clusters)) == 0:
            return random_reward_to_action(preds_old)

        valid_types = [i for i in range(len(all_clusters)) if len(all_clusters[i]) > 0]
        valid_type_coeffs = np.array([action_prob_coeffs[i]
                                      for i in range(len(all_clusters)) if len(all_clusters[i]) > 0])
        chosen_type = np.random.choice(valid_types, 1, p=valid_type_coeffs / np.sum(valid_type_coeffs))[0]
        valid_clusters_nums = all_valid_clusters_nums[chosen_type]
        clickables = all_clickables[chosen_type]
        clusters = all_clusters[chosen_type]
        chosen_cluster_num = np.random.choice(valid_clusters_nums, 1)
        chosen_clickables = tf.boolean_mask(clickables, clusters == chosen_cluster_num)
        chosen_clickable_i = most_probable_weighted_policy_user(tf.gather_nd(preds, chosen_clickables))
        chosen_clickable = tf.concat([tf.gather(chosen_clickables, chosen_clickable_i), [chosen_type]], axis=-1)
        for callback in self.callbacks:
            callback(all_clickables, all_clusters, all_valid_clusters_nums)
        return tf.expand_dims(chosen_clickable, axis=0)
