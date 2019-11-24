from typing import Callable

import tensorflow as tf

import utils
from utils import Config

keras = tf.keras
layers = keras.layers

from reinforcement_learning import RLModel

PolicyUser = Callable[[tf.Tensor], tf.Tensor]


class A2C(RLModel):
    def __init__(self, process_layer: layers.Layer, actor_layer: layers.Layer, critic_layer: layers.Layer,
                 policy_user: PolicyUser, config: Config):
        # is it going to be faster if i define the layers and input to the super's constructor here?
        super(A2C, self).__init__()
        self.process_layer = process_layer
        self.actor_layer = actor_layer
        self.critic_layer = critic_layer
        self.policy_user = policy_user
        self.gamma = config['gamma']
        self.value_loss_coeff = config['value_loss_coeff']
        self.entropy_loss_coeff = config['entropy_loss_coeff']

        self.mean_policy_loss = keras.metrics.Mean(dtype=tf.float32)
        self.mean_value_loss = keras.metrics.Mean(dtype=tf.float32)
        self.mean_entropy = keras.metrics.Mean(dtype=tf.float32)
        self.mean_advantage = keras.metrics.Mean(dtype=tf.float32)

    @tf.function
    def call(self, inputs):
        state = inputs
        representation = self.process_layer(state)
        policy = self.actor_layer(representation)
        value_estimate = self.critic_layer(representation)
        return self.policy_user(tf.nn.softmax(policy)), policy, value_estimate

    def get_log_values(self):
        result = [('Policy Loss:', self.mean_policy_loss.result()),
                  ('Value Loss:', self.mean_value_loss.result()),
                  ('Entropy:', self.mean_entropy.result()),
                  ('Advantage:', self.mean_advantage.result())]
        self.mean_policy_loss.reset_states()
        self.mean_value_loss.reset_states()
        self.mean_entropy.reset_states()
        self.mean_advantage.reset_states()
        return result

    @tf.function
    def compute_loss(self, actions, rewards, policies, value_estimates):
        action_onehots = tf.one_hot(actions, policies.shape[2])[:, :, 0, :]
        # no bootstrap for now. i assume all episodes are over
        # value_estimates = np.asarray(value_estimates.tolist() + [0])

        # no bootstrap for now. i assume all episodes are over
        discounted_rewards = utils.discount(self.gamma, rewards)

        advantages = discounted_rewards - value_estimates

        prob_logs = tf.nn.log_softmax(policies)
        probs = tf.nn.softmax(policies)
        responsible_prob_logs = tf.reduce_sum(prob_logs * action_onehots, axis=-1)
        policy_loss = -tf.reduce_sum(responsible_prob_logs * tf.stop_gradient(advantages[:, :, 0]))
        value_loss = tf.reduce_sum(tf.square(advantages))
        entropy = -tf.reduce_sum(probs * prob_logs)

        self.mean_policy_loss.update_state(policy_loss)
        self.mean_value_loss.update_state(value_loss)
        self.mean_entropy.update_state(entropy)
        self.mean_advantage.update_state(tf.reduce_mean(tf.reduce_mean(advantages[:, :, 0], axis=-1)))

        # why my loss becomes so negative?
        return policy_loss + self.value_loss_coeff * value_loss - self.entropy_loss_coeff * entropy
