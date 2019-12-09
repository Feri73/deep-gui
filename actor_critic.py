from typing import Callable, Tuple, List

import tensorflow as tf
import tensorflow.keras as keras

import utils
from utils import Config

from reinforcement_learning import RLModel, StatelessModel, ModelStates

PolicyUser = Callable[[tf.Tensor], tf.Tensor]


class A2CMetrics:
    def __init__(self):
        self.mean_policy_loss = keras.metrics.Mean(dtype=tf.float32)
        self.mean_value_loss = keras.metrics.Mean(dtype=tf.float32)
        self.mean_entropy = keras.metrics.Mean(dtype=tf.float32)
        self.mean_advantage = keras.metrics.Mean(dtype=tf.float32)

    def reset(self) -> None:
        self.mean_policy_loss.reset_states()
        self.mean_value_loss.reset_states()
        self.mean_entropy.reset_states()
        self.mean_advantage.reset_states()

    def update(self, policy_loss, value_loss, entropy, advantage) -> None:
        self.mean_policy_loss.update_state(policy_loss)
        self.mean_value_loss.update_state(value_loss)
        self.mean_entropy.update_state(entropy)
        self.mean_advantage.update_state(advantage)

    def get_values(self) -> Tuple[Tuple[str, tf.Tensor], ...]:
        return (('Policy Loss:', self.mean_policy_loss.result()),
                ('Value Loss:', self.mean_value_loss.result()),
                ('Entropy:', self.mean_entropy.result()),
                ('Advantage:', self.mean_advantage.result()))


class A2C(RLModel):
    def __init__(self, process_layer: StatelessModel, actor_layer: StatelessModel, critic_layer: StatelessModel,
                 policy_user: PolicyUser, cfg: Config):
        process_init_states = process_layer.get_initial_model_states()
        self.process_states_len = len(process_init_states)
        actor_init_states = actor_layer.get_initial_model_states()
        self.actor_states_len = len(actor_init_states)
        critic_init_states = critic_layer.get_initial_model_states()

        # is it going to be faster if i define the layers and input to the super's constructor here?
        super().__init__(self.encode_model_states((None, None),
                                                  process_init_states, actor_init_states, critic_init_states))
        self.process_layer = process_layer
        self.actor_layer = actor_layer
        self.critic_layer = critic_layer
        self.policy_user = policy_user
        self.gamma = tf.constant(cfg['gamma'])
        self.value_loss_coeff = cfg['value_loss_coeff']
        self.entropy_loss_coeff = cfg['entropy_loss_coeff']
        self.model_states_len = None
        self.inner_metrics = A2CMetrics()

    @staticmethod
    def encode_model_states(self_states: ModelStates, process_states: ModelStates,
                            actor_states: ModelStates, critic_states: ModelStates) -> ModelStates:
        return (*self_states, *process_states, *actor_states, *critic_states)

    def decode_model_states(self, model_states: ModelStates) -> \
            Tuple[ModelStates, ModelStates, ModelStates, ModelStates]:
        return model_states[:2], model_states[2:2 + self.process_states_len], \
               model_states[2 + self.process_states_len:2 + self.process_states_len + self.actor_states_len], \
               model_states[2 + self.process_states_len + self.actor_states_len:]

    @tf.function
    def call(self, env_state, last_action, last_reward, model_states):
        _, process_states, actor_states, critic_states = self.decode_model_states(model_states)

        representation, process_states = self.process_layer(env_state, last_action, last_reward,
                                                            model_states=process_states)
        policy, actor_states = self.actor_layer(representation, model_states=actor_states)
        value_estimate, critic_states = self.critic_layer(representation, model_states=critic_states)

        # i don't really need the stop_gradient here, but just in case i have it
        return tf.stop_gradient(self.policy_user(tf.nn.softmax(policy))), \
               self.encode_model_states((policy, value_estimate), process_states, actor_states, critic_states)

    def get_log_values(self):
        result = self.inner_metrics.get_values()
        self.inner_metrics.reset()
        return result

    @tf.function
    def compute_loss(self, curr_env_state, actions, rewards, model_states_histories):
        rewards = tf.expand_dims(rewards, axis=-1)
        if curr_env_state is None:
            bootstrap_value = tf.zeros_like(rewards[:, -1])
        else:
            bootstrap_value = tf.stop_gradient(self(curr_env_state, actions[:, -1], rewards[:, -1, 0],
                                                    model_states=tuple([model_state_history[:, -1]
                                                                        for model_state_history in
                                                                        model_states_histories]))[1][1])
        bootstrap_value = tf.expand_dims(bootstrap_value, axis=1)
        policies, value_estimates = self.decode_model_states(model_states_histories)[0]

        # with a breakpoint here make sure you are doing it right and all the ranks and indexings are correct
        action_onehots = tf.one_hot(actions, tf.shape(policies)[2])[:, :, 0]

        augmented_rewards = tf.concat([rewards, bootstrap_value], axis=1)
        augmented_value_estimates = tf.concat([value_estimates, bootstrap_value], axis=1)

        discounted_rewards = utils.discount(self.gamma, augmented_rewards)[:, :-1]

        # read about this and think why this is good
        advantages = rewards + self.gamma * augmented_value_estimates[:, 1:] - augmented_value_estimates[:, :-1]
        advantages = utils.discount(self.gamma, advantages)[:, :, 0]

        prob_logs = tf.nn.log_softmax(policies)
        probs = tf.nn.softmax(policies)
        responsible_prob_logs = tf.reduce_sum(prob_logs * action_onehots, axis=-1)
        policy_loss = -tf.reduce_sum(responsible_prob_logs * tf.stop_gradient(advantages))
        value_loss = tf.reduce_sum(tf.square(discounted_rewards - value_estimates))
        entropy = -tf.reduce_sum(probs * prob_logs)

        self.inner_metrics.update(policy_loss, value_loss, entropy,
                                  tf.reduce_mean(tf.reduce_mean(advantages, axis=-1)))
        return policy_loss + self.value_loss_coeff * value_loss - self.entropy_loss_coeff * entropy
