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

    @tf.function
    def call(self, inputs):
        state = inputs
        representation = self.process_layer(state)
        policy = self.actor_layer(representation)
        value_estimate = self.critic_layer(representation)
        return self.policy_user(policy), policy, value_estimate

    @tf.function
    def compute_loss(self, actions, rewards, policies, value_estimates):
        action_onehots = tf.one_hot(actions, policies.shape[2])[:, :, 0, :]
        responsible_outputs = tf.reduce_sum(policies * action_onehots, [2])
        # no bootstrap for now. i assume all episodes are over
        # value_estimates = np.asarray(value_estimates.tolist() + [0])

        # why this is different than the paper?
        # advantages = rewards + self.gamma * value_estimates[1:] - value_estimates[:-1]

        # no bootstrap for now. i assume all episodes are over
        discounted_rewards = utils.discount(self.gamma, rewards)

        advantages = discounted_rewards - value_estimates

        policy_loss = -tf.reduce_sum(tf.math.log(responsible_outputs) * advantages)
        # do i need .5 coeff?
        value_loss = tf.reduce_sum(tf.square(advantages))
        entropy = -tf.reduce_sum(policies * tf.math.log(policies))

        # why my loss becomes so negative?
        return policy_loss + self.value_loss_coeff * value_loss - self.entropy_loss_coeff * entropy
