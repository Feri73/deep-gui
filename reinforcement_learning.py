# do not use eager execution when possible (use tf.function)
# make this a library --> general rl library in tf 2 , maybe even make it in a way that tf parts of it are separated
#     and its core is platform-independent
from typing import Tuple, Any, Optional, List, Callable

from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from environment import EnvironmentCallbacks, EnvironmentController
from utils import Config, Gradient, add_gradients, MemVariable, MemList

keras = tf.keras


class RLModel(keras.Model, ABC):
    @abstractmethod
    def call(self, inputs: np.ndarray) -> Tuple[tf.Tensor, ...]:
        pass

    @abstractmethod
    def compute_loss(self, action_history: tf.Tensor, reward_history: tf.Tensor,
                     *inner_measures_histories: List[tf.Tensor]) -> tf.Tensor:
        pass

    @abstractmethod
    def get_log_values(self) -> List[Tuple[str, tf.Tensor]]:
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
# this method is very long! factorize it.
class RLAgent(EnvironmentCallbacks, EnvironmentController):
    def __init__(self, id: int, rl_model: RLModel, realize_action: Callable[[np.ndarray], Any],
                 coordinator: Optional[RLCoordinator], optimizer: keras.optimizers.Optimizer, cfg: Config):
        self.id = id
        self.rl_model = rl_model
        self.realize_action = realize_action
        self.coordinator = coordinator
        self.optimizer = optimizer

        self.steps_per_gradient_update = cfg['steps_per_gradient_update']
        self.steps_per_agent = cfg['steps_per_agent']

        self.step = 0
        self.in_on_wait = False

        self.state_history = MemVariable(lambda: [])
        self.action_history = MemVariable(lambda: [])
        self.realized_action_history = MemVariable(lambda: [])
        self.reward_history = MemVariable(lambda: [])
        self.inner_measures_histories = MemVariable(lambda: [])
        self.tape = MemVariable(lambda: tf.GradientTape())
        self.episode_vars = MemList([self.state_history, self.action_history, self.realized_action_history,
                                     self.reward_history, self.inner_measures_histories, self.tape])

        self.total_gradient = MemVariable(lambda: 0)

    def on_episode_gradient_computed(self, loss: tf.Tensor, gradient: Gradient, state_history: List[np.ndarray],
                                     realized_action_history: List[Any], reward_history: List[float]) -> None:
        pass

    @tf.function
    def apply_gradient(self, gradient: Gradient) -> None:
        self.optimizer.apply_gradients(zip(gradient, self.rl_model.trainable_weights))

    def replace_weights(self, reference_agent: 'RLAgent') -> None:
        self.rl_model.set_weights(reference_agent.rl_model.get_weights())

    def build_model(self, input_shape: tuple) -> None:
        self.rl_model.build(input_shape)

    def is_built(self) -> bool:
        return self.rl_model.built

    def should_start_episode(self) -> bool:
        res = self.step < self.steps_per_agent
        if not res:
            self.on_wait()
            self.total_gradient.archive()
            self.on_wait()
        return res

    def on_episode_start(self, state: np.ndarray) -> None:
        self.step += 1
        self.state_history.value += [state]
        self.tape.value.__enter__()

    def get_next_action(self, state: np.ndarray) -> Any:
        # is this ok? it should not create new model (i.e. weights) every time!
        action, *inner_measures = self.rl_model(tf.expand_dims(state, axis=0))
        realized_action = self.realize_action(action.numpy())

        self.realized_action_history.value += [realized_action]
        for measure_i, inner_measure in enumerate(inner_measures):
            if len(self.inner_measures_histories.value) == measure_i:
                self.inner_measures_histories.value += [[]]
            self.inner_measures_histories.value[measure_i] += [inner_measure[0]]

        return realized_action

    # add gradient clipping
    def on_wait(self) -> None:
        self.in_on_wait = True
        with self.tape.value.stop_recording():
            if self.episode_vars.has_archive():
                last_inner_measures_histories = self.inner_measures_histories.last_value()

                # why do i need the last dimension in action and reward histories?
                loss = self.rl_model.compute_loss(tf.reshape(self.action_history.last_value(), shape=(1, -1, 1)),
                                                  tf.reshape(self.reward_history.last_value(), shape=(1, -1, 1)),
                                                  *[tf.reshape(measure_history,
                                                               shape=(1, len(last_inner_measures_histories[0]), -1))
                                                    for measure_history in last_inner_measures_histories])
                self.tape.last_value().__exit__(None, None, None)
                # this should use tf function. but how?
                gradient = self.tape.last_value().gradient(loss, self.rl_model.trainable_weights)
                self.on_episode_gradient_computed(loss, gradient, self.state_history.value,
                                                  self.realized_action_history.value, self.reward_history.value)
                self.total_gradient.value = add_gradients(self.total_gradient.value, gradient)
            if self.total_gradient.has_archive() and self.total_gradient.last_value() != 0:
                self.coordinator.add_gradient(self.id, self.total_gradient.last_value())
            self.episode_vars.reset_archive()
            self.total_gradient.reset_archive()
        self.in_on_wait = False

    def on_state_change(self, src_state: np.ndarray, action: Any, dst_state: np.ndarray, reward: float) -> None:
        self.state_history.value += [dst_state]
        self.action_history.value += [action]
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
