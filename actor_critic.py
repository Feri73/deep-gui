from typing import Tuple

import numpy as np
import tensorflow as tf

from tf1 import ArrayTuple
from reinforcement_learning import RLModel
from utils import Config

TensorTuple = Tuple[tf.Tensor, ...]


# add a procedure so that users can specify the logs
# remember that here, policy_layer should produce logits, not probs
# create this class in Keras
# try to generalize it to continous space, and policy_user define its own gradient
# this also assumes policy is a 1d thing (NOT considering time or batch) but some parts of the code do not use this
#   assumption while some (in loss evaluation) does
# add typings for layers (and also in other files add the missing types)
# change the layers interface so that instead of inputting tuples, i can input the function
#   arguments like a normal function
class A2C(RLModel):
    def __init__(self, process_layer, actor_layer, critic_layer, policy_user,
                 actor_layer_output_shape: Tuple[int, ...], cfg: Config, default_action: int = 0,
                 default_reward: float = 0.0, default_process_states: ArrayTuple = None,
                 default_actor_states: ArrayTuple = None, default_critic_states: ArrayTuple = None):
        self.process_layer = process_layer
        self.actor_layer = actor_layer
        self.critic_layer = critic_layer
        self.policy_user = policy_user
        self.actor_layer_output_shape = actor_layer_output_shape
        self.gamma = cfg['gamma']
        self.value_loss_coeff = cfg['value_loss_coeff']
        self.entropy_loss_coeff = cfg['entropy_loss_coeff']

        self.default_action = np.array(default_action)
        self.default_reward = default_reward
        self.default_process_states = () if default_process_states is None else default_process_states
        self.default_actor_states = () if default_actor_states is None else default_actor_states
        self.default_critic_states = () if default_critic_states is None else default_critic_states

    @staticmethod
    def encode_states(self_states: tuple, process_states: tuple, actor_states: tuple, critic_states: tuple) -> tuple:
        return (*self_states, *process_states, *actor_states, *critic_states)

    def decode_states(self, states_: TensorTuple) -> \
            Tuple[TensorTuple, TensorTuple, TensorTuple, TensorTuple]:
        return states_[:2], states_[2:2 + len(self.default_process_states)], \
               states_[2 + len(self.default_process_states):
                       2 + len(self.default_process_states) + len(self.default_actor_states)], \
               states_[2 + len(self.default_process_states) + len(self.default_actor_states):]

    def eb2ebt(self, states: TensorTuple) -> TensorTuple:
        return tuple(map(lambda x: tf.expand_dims(x, axis=1), states))

    # this should probably be in utils
    # is should not use self.gamma, but i should convert it to tf.constant to avoid retrace in tf2 situations
    #   there is also one later
    # didn't test this (current test does not work with it)
    def discount(self, array: tf.Tensor) -> tf.Tensor:
        padding = tf.stack([tf.zeros(2, dtype=tf.int32), tf.stack([0, tf.shape(array)[1] - 1]),
                            tf.zeros(2, dtype=tf.int32)])
        return tf.nn.conv1d(
            tf.pad(array, padding, 'CONSTANT'),
            tf.expand_dims(tf.expand_dims(
                tf.map_fn(lambda i: self.gamma ** i,
                          tf.range(tf.cast(tf.shape(array)[1], tf.float32), dtype=tf.float32)), axis=-1), axis=-1),
            stride=1, padding='VALID')

    def calc_next_action(self, env_states_bt: tf.Tensor, bef_actions_bt: tf.Tensor, bef_rewards_bt: tf.Tensor,
                         bef_model_states_eb: TensorTuple) -> Tuple[tf.Tensor, TensorTuple]:
        _, bef_process_states_eb, bef_actor_states_eb, bef_critic_states_eb = self.decode_states(bef_model_states_eb)
        representation_bt, process_states_eb = self.process_layer((env_states_bt, bef_actions_bt, bef_rewards_bt,
                                                                   bef_process_states_eb))
        policy_bt, actor_states_eb = self.actor_layer((representation_bt, bef_actor_states_eb))
        value_estimate_bt, critic_states_eb = self.critic_layer((representation_bt, bef_critic_states_eb))

        # remember, here i am supposed to return ebt, but for states that i do not use in loss, i am fakely generating
        #   the t dimension. try to solve this.
        return self.policy_user(tf.nn.softmax(policy_bt)), self.encode_states((policy_bt, value_estimate_bt),
                                                                              self.eb2ebt(process_states_eb),
                                                                              self.eb2ebt(actor_states_eb),
                                                                              self.eb2ebt(critic_states_eb))

    def calc_loss(self, actions_bt: tf.Tensor, rewards_bt: tf.Tensor, model_states_ebt: TensorTuple,
                  finished_b: tf.Tensor) -> Tuple[tf.Tensor, TensorTuple]:
        rewards_bt = tf.expand_dims(rewards_bt, axis=-1)
        policies_bt, value_estimates_bt = self.decode_states(model_states_ebt)[0]

        # is this ok to call get_next_action here? how can i use an existing graph
        bootstrap_value_b = tf.where(finished_b,
                                     tf.zeros_like(rewards_bt[:, -1]),
                                     tf.stop_gradient(value_estimates_bt[:, -1]))
        bootstrap_value_bt = tf.expand_dims(bootstrap_value_b, axis=1)

        actions_bt = tf.expand_dims(actions_bt, axis=-1)
        policies_bt = policies_bt[:, :-1]
        value_estimates_bt = value_estimates_bt[:, :-1]

        # with a breakpoint here make sure you are doing it right and all the ranks and indexings are correct
        action_onehots_bt = tf.one_hot(actions_bt, policies_bt.shape[2])[:, :, 0]

        augmented_rewards_bt = tf.concat([rewards_bt, bootstrap_value_bt], axis=1)
        augmented_value_estimates_bt = tf.concat([value_estimates_bt, bootstrap_value_bt], axis=1)

        discounted_rewards_bt = self.discount(augmented_rewards_bt)[:, :-1]

        # read about this and think why this is good
        advantages_bt = rewards_bt + self.gamma * augmented_value_estimates_bt[:, 1:] - \
                        augmented_value_estimates_bt[:, :-1]
        advantages_bt = self.discount(advantages_bt)[:, :, 0]

        prob_logs_bt = tf.nn.log_softmax(policies_bt)
        probs_bt = tf.nn.softmax(policies_bt)
        responsible_prob_logs_bt = tf.reduce_sum(prob_logs_bt * action_onehots_bt, axis=-1)
        policy_loss = -tf.reduce_sum(responsible_prob_logs_bt * tf.stop_gradient(advantages_bt))
        # policy_loss = tf.reduce_sum(tf.square(rewards_bt - tf.reduce_sum(probs_bt * action_onehots_bt, axis=-1)))
        value_loss = tf.reduce_sum(tf.square(discounted_rewards_bt - value_estimates_bt))
        entropy = -tf.reduce_sum(probs_bt * prob_logs_bt)

        return policy_loss + self.value_loss_coeff * value_loss - self.entropy_loss_coeff * entropy, \
               (policy_loss, value_loss, entropy, tf.reduce_mean(tf.reduce_mean(advantages_bt, axis=-1)))

    def get_default_action(self) -> np.ndarray:
        return self.default_action

    def get_default_reward(self) -> float:
        return self.default_reward

    def get_default_states(self) -> ArrayTuple:
        return self.encode_states((np.zeros(self.actor_layer_output_shape), np.array([0])),
                                  self.default_process_states, self.default_actor_states, self.default_critic_states)
