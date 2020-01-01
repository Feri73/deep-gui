from typing import Tuple, Optional, List, Union

import numpy as np
import tensorflow as tf

from reinforcement_learning import Episode, RLAgent, RLModel, RLCoordinator
from utils import Config

ArrayTuple = Tuple[np.ndarray, ...]


# gradients vs gradient (everywhere, not only this file)
# change the name of this file
# hey! apparently i can set non-placeholders in the feed dict. first search about it, and it legitimate, use it here
#   to simplify everything
# maybe instead of scope, i cna use graphs. this way i can have my own session in this file
# this idea cna easily become a lazy ML framework.
# maybe i need to have a class in reinforcement_learning.py for Gradient that includes is_target(RLAgent)
#   and this way the set_target_gradient call in test_doom.py becomes more reasonable
class LazyGradient:
    def __init__(self, *, value: ArrayTuple = None, episode: Episode = None,
                 compute_agent: 'TF1RLAgent' = None):
        self.value = value
        self.episode = episode
        self.compute_agent = compute_agent
        if self.compute_agent is not None and self.compute_agent.check_weight_for_gradient:
            self.compute_agent.incomplete_gradients += [self]
        self.loss = None
        self.logs_e = None

    @staticmethod
    def create_gradient(value: ArrayTuple) -> 'LazyGradient':
        return LazyGradient(value=value)

    def is_computed(self):
        return self.value is not None

    # remember to call this after applying gradients, so that i do not use static_gradient_apply
    def has_logs(self):
        return self.logs_e is not None

    def compute(self) -> ArrayTuple:
        if self.is_computed():
            return self.value
        compute_agent = self.compute_agent
        self.value, self.loss, self.logs_e = self.eval(
            (compute_agent.output_gradients, compute_agent.output_loss, compute_agent.output_logs_e), compute_agent)
        return self.value

    # how can i make this more efficient, so that i still use dynamic apply even when adding gradients
    #   e.g. i am not using batch here, while my networks support batch_size > 1
    def __add__(self, other: Optional['LazyGradient']) -> 'LazyGradient':
        if other is None:
            return self
        if not isinstance(other, LazyGradient):
            raise TypeError()
        self.compute()
        other.compute()
        return LazyGradient(value=tuple(self.value[grad_i] + other.value[grad_i] for grad_i in range(len(self.value))))

    def __reduce__(self):
        return LazyGradient.create_gradient, (self.compute(),)

    # currently i do not support gradient computation for partial episodes (which needs model_state other than default)
    def eval(self, fetches, compute_agent: 'TF1RLAgent'):
        return compute_agent.session.run(fetches, feed_dict={
            compute_agent.input_env_states_bt: compute_agent.tb2bt(self.episode.states_tb),
            compute_agent.input_bef_actions_bt: compute_agent.tb2bt(self.episode.actions_tb),
            compute_agent.input_bef_rewards_bt: compute_agent.tb2bt(self.episode.rewards_tb),
            compute_agent.input_episode_finished_b: [self.episode.finished],
            compute_agent.inputs_bef_model_states_eb: tuple(map(lambda x: np.expand_dims(x, axis=0),
                                                                compute_agent.default_model_states))
        })

    def apply(self, target_agent: 'TF1RLAgent') -> None:
        if self.compute_agent is not None and self.compute_agent.gradient_target_scope == target_agent.scope:
            agent = self.compute_agent
        elif target_agent.gradient_target_scope == target_agent.scope:
            self.compute()
            agent = target_agent
        else:
            comp_agent = 'None' if self.compute_agent is None else f'{self.compute_agent.scope}'
            raise ValueError(f'Trying to apply gradient based on {comp_agent} weights on {target_agent.scope}, while '
                             f'{target_agent.scope} can only apply gradients on {target_agent.gradient_target_scope}.')
        # also think about this: if i do not get gradients as outputs, will it be faster
        if self.is_computed():
            agent.session.run(agent.op_apply_static_gradients,
                              feed_dict={agent.input_gradients: self.compute()})
        else:
            self.value, self.loss, self.logs_e, _ = self.eval(
                (self.compute_agent.output_gradients, self.compute_agent.output_loss, self.compute_agent.output_logs_e,
                 self.compute_agent.op_apply_dynamic_gradients), self.compute_agent)


# a generalized class for all backends keras supports
# not every scope needs evey part of this graph
class TF1RLAgent(RLAgent):
    def __init__(self, id: int, rl_model: RLModel, coordinator: RLCoordinator, scope: str, session: tf.Session,
                 env_state_shape: Tuple[int, ...], optimizer: tf.train.Optimizer, cfg: Config):
        super().__init__(id, rl_model, coordinator, cfg)
        self.gradient_clip_max = cfg['gradient_clip_max']
        self.check_weight_for_gradient = cfg['check_weight_for_gradient']
        self.inherent_clip = cfg['inherent_clip']
        self.apply_clip = cfg['apply_clip']

        self.scope = scope
        self.session = session
        self.optimizer = optimizer

        self.incomplete_gradients = []

        with tf.variable_scope(scope):
            # i am assuming dtype of all of these except action
            self.input_env_states_bt = tf.placeholder(shape=[None, None, *env_state_shape], dtype=tf.float32)
            self.input_bef_actions_bt = tf.placeholder(shape=[None, None, *self.default_action.shape],
                                                       dtype=self.default_action.dtype)
            self.input_bef_rewards_bt = tf.placeholder(shape=[None, None], dtype=tf.float32)
            self.inputs_bef_model_states_eb = tuple(tf.placeholder(shape=[None, *model_state.shape], dtype=tf.float32)
                                                    for model_state in self.default_model_states)
            self.input_episode_finished_b = tf.placeholder(shape=[None], dtype=tf.bool)

            # how can i force the get_action layer to have this form of input (and also, preferably make it general
            #   so that other backends (tf1, theano, etc.) can use it too)
            self.output_actions_bt, self.output_model_states_ebt = \
                rl_model.calc_next_action(self.input_env_states_bt, self.input_bef_actions_bt,
                                          self.input_bef_rewards_bt, self.inputs_bef_model_states_eb)
            # note: the calc_loss in RLModel has one more element in model_state than actions and rewards
            #   somehow document this
            self.output_loss, self.output_logs_e = rl_model.calc_loss(self.input_bef_actions_bt[:, 1:],
                                                                      self.input_bef_rewards_bt[:, 1:],
                                                                      self.output_model_states_ebt,
                                                                      self.input_episode_finished_b)

            self.trainable_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

            self.output_gradients = tf.gradients(self.output_loss, self.trainable_weights)
            if self.inherent_clip:
                self.output_gradients = self.clip_gradient(self.output_gradients)

            # what is a batter way that [0] in these 2 lines
            #   (specify input_shape instead of specifying batch_size separately)
            self.input_gradients = tuple(tf.placeholder(shape=weight.shape, dtype=tf.float32)
                                                 for weight in self.trainable_weights)
            self.input_target_weights = tuple(tf.placeholder(shape=weight.shape, dtype=tf.float32)
                                              for weight in self.trainable_weights)

            # use the output of clip for faster gradient norm computation

            self.gradient_target_scope = None
            self.op_apply_dynamic_gradients = None
            self.op_apply_static_gradients = None

            self.op_replace_dynamic_weights = {}
            self.op_replace_static_weights = [
                self.trainable_weights[weight_i].assign(self.input_target_weights[weight_i])
                for weight_i in range(len(self.trainable_weights))]

        # do i have optimizer weights in self.trainable_weights? this is performance-wise important in gradient
        #       computation but is critical ot not have in replacing weights of one scope to another
        session.run(tf.initialize_variables(tf.global_variables(scope)))

    @staticmethod
    def tb2bt(tb: list) -> np.ndarray:
        tb = np.array(tb)
        return np.transpose(tb, axes=[1, 0, *range(2, len(tb.shape))])

    @staticmethod
    def te2et(te: List[tuple]) -> Tuple[list, ...]:
        return tuple(zip(*te))

    def clip_gradient(self, gradient: Tuple[tf.Tensor, ...]) -> Tuple[tf.Tensor, ...]:
        if self.gradient_clip_max is not None:
            return tf.clip_by_global_norm(gradient, self.gradient_clip_max)[0]
        return gradient

    # this function (and generally this class) can work with only one batch (i mean, why should i have batched data
    #   in this function when this is only called from RLAgent and RLAgent only calls it with 1 batch data?)
    def calc_next_action(self, env_state_b: np.ndarray, action_b: np.ndarray, reward_b: np.ndarray,
                         model_states_eb: ArrayTuple) -> Tuple[np.ndarray, ArrayTuple]:
        action_bt, model_states_ebt = self.session.run((self.output_actions_bt, self.output_model_states_ebt),
                                                       feed_dict={
                                                           self.input_env_states_bt: self.tb2bt([env_state_b]),
                                                           self.input_bef_actions_bt: self.tb2bt([action_b]),
                                                           self.input_bef_rewards_bt: self.tb2bt([reward_b]),
                                                           self.inputs_bef_model_states_eb: tuple(map(np.array,
                                                                                                      model_states_eb))
                                                       })
        return action_bt[:, -1], tuple(map(lambda x: x[:, -1], model_states_ebt))

    # i think i should not input episode. only rewards and evn_states
    def calc_gradient(self, episode: Episode, states_teb: List[ArrayTuple]) -> LazyGradient:
        return LazyGradient(episode=episode, compute_agent=self)

    def calc_incomplete_gradients(self, to_be_applied: Optional[LazyGradient]) -> None:
        if to_be_applied is not None:
            ret = True
            for gradient in self.incomplete_gradients:
                if not gradient.is_computed() and gradient != to_be_applied:
                    ret = False
                    break
            if ret:
                return
        for gradient in [g for g in self.incomplete_gradients]:
            # write some warning here
            gradient.compute()
            # maybe do this inside the compute function of gradient. if done, then i can get rid of the first loop
            #   (but i do need to check if its only to_be_applied that is not computed)
            self.incomplete_gradients.remove(gradient)

    def apply_gradient(self, gradients: Optional[LazyGradient]) -> None:
        if gradients is None:
            return
        self.calc_incomplete_gradients(gradients)
        gradients.apply(self)

    def add_replacement_target(self, target: 'TF1RLAgent') -> None:
        target_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, target.scope)
        self.op_replace_dynamic_weights[target.scope] = [
            self.trainable_weights[weight_i].assign(target_weights[weight_i])
            for weight_i in range(len(self.trainable_weights))]

    def replace_parameter(self, target: Union['TF1RLAgent', ArrayTuple]) -> None:
        self.calc_incomplete_gradients(None)
        if isinstance(target, TF1RLAgent):
            self.session.run(self.op_replace_dynamic_weights[target.scope])
        else:
            self.session.run(self.op_replace_static_weights, feed_dict={self.input_target_weights: target})

    def get_parameter(self):
        return self.session.run(self.trainable_weights)

    # instead of "set_...", make it "add_..."
    def set_generated_gradient_target(self, target: 'TF1RLAgent'):
        self.gradient_target_scope = target.scope
        before_vars = self.optimizer.variables()
        dynamic_grads = self.clip_gradient(self.output_gradients) if self.apply_clip else self.output_gradients
        static_grads = self.clip_gradient(self.input_gradients) if self.apply_clip else self.input_gradients
        self.op_apply_dynamic_gradients = self.optimizer.apply_gradients(
            zip(dynamic_grads, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, target.scope)))
        self.op_apply_static_gradients = self.optimizer.apply_gradients(
            zip(static_grads, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, target.scope)))
        after_vars = self.optimizer.variables()
        self.session.run(tf.initialize_variables(after_vars[len(before_vars):]))

    # think about this: i should change things so that RLAgent has RLModel and RLModel has model_calculator (this
    #   class) and this these are in calculator
