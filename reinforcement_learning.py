# do not use eager execution when possible (use tf.function)
# make this a library --> general rl library in tf 2 , maybe even make it in a way that tf parts of it are separated
#     and its core is platform-independent

# look everywhere to see if i am indirectly modifying stuff (instead of functional style)
# instead of weights, use trainable weights (both in replacing and getting and generally everywhere)
# 2 thigs to do --> check openai implementation to see if they clip the add_gradients
# correct the way i synchronize the workers and global network so that i update the workers before they go any further
#     in both coordinators
# check it i am using tapes correctly --> (gradients are not accumulated from one episode to another)
# check the implementation of asynchronous coordinators to see of i am updating weights
#   and computing gradients correctly
#   also see why the weights and gradients and everything keeps increasing
# then define a simple task (test) and make sure it works
# and also think about the fact that a2c the way i implementde it only gets trained on episodes where reward is 0
#   (is it? i remember the situations where it would get stuck because and reward was 0 so it would never learn)
#   if the above is true, then 1. i can use it to create better policyUsers 2. maybe if no paper says this,
#   i can make a theory out of it
# eager mode is very slow!!!
import time
from typing import Tuple, Any, Optional, List, Callable, Union

from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from environment import EnvironmentCallbacks, EnvironmentController
from utils import Config, Gradient, add_gradients, MemVariable, MemList, batchify

ModelStates = Tuple[Optional[tf.Tensor], ...]


class StatelessModel(keras.layers.Layer):
    def __init__(self, initial_model_states: ModelStates, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_model_states = initial_model_states

    def call(self, inputs: tf.Tensor, *, model_states: ModelStates):
        return super().call(inputs)

    # this should get a batch size as input
    def get_initial_model_states(self) -> ModelStates:
        return self.initial_model_states


class NoStateStatelessModel(StatelessModel):
    def __init__(self, inner_model: keras.layers.Layer):
        super().__init__(())
        self.inner_model = inner_model

    def call(self, *args, model_states: ModelStates):
        return self.inner_model.call(*args), ()


# this needs to be stateless
class RLModel(StatelessModel, ABC):
    @abstractmethod
    def call(self, env_state: tf.Tensor, last_action: Optional[tf.Tensor], last_reward: Optional[tf.Tensor],
             *, model_states: ModelStates) -> Tuple[tf.Tensor, Tuple[tf.Tensor, ...]]:
        pass

    @abstractmethod
    def compute_loss(self, curr_env_state: Optional[tf.Tensor], action_history: tf.Tensor, reward_history: tf.Tensor,
                     model_states_histories: ModelStates) -> tf.Tensor:
        pass

    @abstractmethod
    def get_log_values(self) -> Tuple[Tuple[str, tf.Tensor], ...]:
        pass


class RLCoordinator(ABC):
    @abstractmethod
    def start_learning(self) -> None:
        pass

    @abstractmethod
    def add_gradient(self, agent_id: int, gradient: Gradient) -> None:
        pass


# rethink the functions (like iteration count should be input of start learning, etc.) or names and meaning of
#   classes (like RLCoordinator is only about learning, while RLAgent plays too!)
# this should inherit from Model
# assumes fixed-length episodes
class RLAgent(EnvironmentCallbacks, EnvironmentController):
    def __init__(self, id: int, rl_model: RLModel, realize_action: Callable[[np.ndarray], Any],
                 coordinator: Optional[RLCoordinator], optimizer: keras.optimizers.Optimizer, cfg: Config):
        self.id = id
        self.rl_model = rl_model
        self.realize_action = realize_action
        self.coordinator = coordinator
        self.optimizer = optimizer

        self.rl_model_initial_states = self.rl_model.get_initial_model_states()

        self.steps_per_gradient_update = cfg['steps_per_gradient_update']
        self.steps_per_agent = cfg['steps_per_agent']
        self.max_steps_per_episode = cfg['max_steps_per_episode'] or np.inf
        self.gradient_clip_max = cfg['gradient_clip_max']

        self.step = 0
        self.in_on_wait = False
        self.last_tape_stop_recording = None

        self.env_state_history = MemVariable(lambda: [])
        self.action_history = MemVariable(lambda: [])
        self.realized_action_history = MemVariable(lambda: [])
        self.reward_history = MemVariable(lambda: [])
        self.model_states_histories = MemVariable(lambda: [])
        self.tape = MemVariable(lambda: tf.GradientTape())
        self.episode_truncated = MemVariable(lambda: False)
        self.episode_vars = MemList([self.env_state_history, self.action_history, self.realized_action_history,
                                     self.reward_history, self.model_states_histories, self.tape,
                                     self.episode_truncated])

        self.total_gradient = MemVariable(lambda: 0)

    def on_episode_gradient_computed(self, loss: tf.Tensor, gradient: Gradient, state_history: List[np.ndarray],
                                     realized_action_history: List[Any], reward_history: List[float]) -> None:
        pass

    @tf.function
    def apply_gradient(self, gradient: Gradient) -> None:
        self.optimizer.apply_gradients(zip(gradient, self.rl_model.trainable_weights))

    # this appears to be slow. find better solutions
    # also i may be able to make this function @tf.function
    def replace_weights(self, reference: Union['RLAgent', List[tf.Tensor]]) -> None:
        if isinstance(reference, RLAgent):
            ref_trainable_weights = reference.rl_model.trainable_weights
        else:
            ref_trainable_weights = reference
        self_trainable_weights = self.rl_model.trainable_weights
        for weight_i in range(len(self_trainable_weights)):
            self_trainable_weights[weight_i].assign(ref_trainable_weights[weight_i])
        # self.rl_model.set_weights(reference_agent.rl_model.get_weights())

    def build_model(self, input_shape: tuple) -> None:
        self.rl_model(tf.zeros(shape=input_shape), None, None, model_states=self.rl_model_initial_states)

    def is_built(self) -> bool:
        return self.rl_model.built

    # this is not exactly what arthur juliani does. cuz i do not continue anymore
    # why if i set max_steps to a lower value, the average of episode lengths also becomes lower?
    # i think here > should be >=
    def should_continue_episode(self) -> bool:
        self.episode_truncated.value = len(self.env_state_history.value) >= self.max_steps_per_episode
        return not self.episode_truncated.value

    def should_start_episode(self) -> bool:
        res = self.step < self.steps_per_agent
        # i should not make these calls here. I should have a clalback that says everything is done, and call there.
        if not res:
            self.on_wait()
            self.total_gradient.archive()
            self.on_wait()
        return res

    def on_episode_start(self, state: np.ndarray) -> None:
        self.step += 1
        self.env_state_history.value += [state]
        if self.tape.has_archive():
            self.last_tape_stop_recording = self.tape.last_value().stop_recording()
            self.last_tape_stop_recording.__enter__()
        self.tape.value.__enter__()
        # i think i should epxlicitly say here what to watch, otherwise
        #   it may watch everything from all threads (search tho)

    def get_next_action(self, state: np.ndarray) -> Any:
        action, model_states = self.rl_model(batchify(state),
                                             batchify(self.action_history.value[-1])
                                             if len(self.action_history.value) > 0 else None,
                                             batchify(self.reward_history.value[-1])
                                             if len(self.reward_history.value) > 0 else None,
                                             model_states=tuple([s[-1] for s in
                                                                 self.model_states_histories.value])
                                             if len(self.model_states_histories.value) > 0
                                             else self.rl_model.get_initial_model_states())
        realized_action = self.realize_action(action.numpy())

        self.action_history.value += [action]
        # remove this, in calling rl_model, simply ise historyes[-1], and then when calling compute_loss, use
        #   tuple([tf.transpose(state, [1, 0, 2]) for state in zip(*a2_states)])
        for model_state_i, model_state in enumerate(model_states):
            if len(self.model_states_histories.value) == model_state_i:
                self.model_states_histories.value += [[]]
            self.model_states_histories.value[model_state_i] += [model_state]

        return realized_action

    # make gradient clipping optional
    def compute_last_gradient(self):
        if self.episode_vars.has_archive():
            model_states_histories = self.model_states_histories.last_value()

            self.last_tape_stop_recording.__exit__(None, None, None)
            # why do i need the last dimension in action and reward histories?
            loss = self.rl_model.compute_loss(
                batchify(self.env_state_history.last_value()[-1])
                if self.episode_truncated.last_value() else None,
                batchify(self.action_history.last_value()),
                batchify(self.reward_history.last_value()),
                tuple([tf.transpose(measure_history, [1, 0, 2]) for measure_history in model_states_histories]))
            self.tape.last_value().__exit__(None, None, None)
            # this should use tf function. but how?
            gradient = self.tape.last_value().gradient(loss, self.rl_model.trainable_weights)
            if self.gradient_clip_max is not None:
                gradient, _ = tf.clip_by_global_norm(gradient, self.gradient_clip_max)  # .5
            self.on_episode_gradient_computed(loss, gradient, self.env_state_history.last_value(),
                                              self.realized_action_history.last_value(),
                                              self.reward_history.last_value())  # .18
            self.total_gradient.value = add_gradients(self.total_gradient.value, gradient)  # .18
            self.episode_vars.reset_archive()

    def add_last_gradient(self):
        if self.total_gradient.has_archive() and self.total_gradient.last_value() != 0:
            self.coordinator.add_gradient(self.id, self.total_gradient.last_value())
            self.total_gradient.reset_archive()

    def on_wait(self) -> None:
        self.in_on_wait = True
        if len(self.env_state_history.value) > 0:
            with self.tape.value.stop_recording():
                self.compute_last_gradient()
                self.add_last_gradient()
        else:
            self.compute_last_gradient()
            self.add_last_gradient()
        self.in_on_wait = False

    def on_state_change(self, src_state: np.ndarray, action: Any, dst_state: np.ndarray, reward: float) -> None:
        self.env_state_history.value += [dst_state]
        self.realized_action_history.value += [action]
        self.reward_history.value += [reward]

    def on_episode_end(self) -> None:
        self.episode_vars.archive()
        if self.step % self.steps_per_gradient_update == 0:
            self.total_gradient.archive()

    def on_error(self) -> None:
        self.tape.value.__exit__(None, None, None)
        self.episode_vars.reset_value()
        if self.in_on_wait and self.episode_vars.has_archive():
            self.tape.last_value().__exit__(None, None, None)
            self.episode_vars.reset_archive()
