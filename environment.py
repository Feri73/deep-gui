from abc import ABC, abstractmethod
from typing import Any, List, Callable

import numpy as np


# maybe i can change all numpy usage to tensorflow

class EnvironmentCallbacks:
    def on_episode_start(self, state: np.ndarray) -> None:
        pass

    # for every action, this should be called exactly once, before on_state_change is called
    def on_wait(self) -> None:
        pass

    def on_state_change(self, src_state: np.ndarray, action: Any, dst_state: np.ndarray, reward: float) -> None:
        pass

    def on_episode_end(self) -> None:
        pass

    def on_error(self) -> None:
        pass


class EnvironmentController(ABC):
    @abstractmethod
    def should_start_episode(self) -> bool:
        pass

    @abstractmethod
    def get_next_action(self, state: np.ndarray) -> Any:
        pass


class Environment(ABC):
    def __init__(self, callbacks: List[EnvironmentCallbacks], controller: EnvironmentController):
        self.callbacks = callbacks
        self.controller = controller

    def start(self):
        while self.should_start_episode():
            self.restart()
            self.on_episode_start(self.read_state())
            while not self.is_finished():
                last_state = self.read_state()
                action = self.get_next_action(self.read_state())
                reward = self.act(action, self.on_wait)
                self.on_state_change(last_state, action, self.read_state(), reward)
            self.on_episode_end()

    @abstractmethod
    def restart(self) -> None:
        pass

    @abstractmethod
    def read_state(self) -> np.ndarray:
        pass

    @abstractmethod
    def is_finished(self) -> bool:
        pass

    @abstractmethod
    def act(self, action: Any, wait_action: Callable) -> float:
        pass

    def should_start_episode(self) -> bool:
        return self.controller.should_start_episode()

    def get_next_action(self, state: np.ndarray) -> Any:
        return self.controller.get_next_action(state)

    def on_episode_start(self, state: np.ndarray) -> None:
        for callback in self.callbacks:
            callback.on_episode_start(state)

    def on_episode_end(self) -> None:
        for callback in self.callbacks:
            callback.on_episode_end()

    def on_state_change(self, src_state: np.ndarray, action: Any, dst_state: np.ndarray, reward: float) -> None:
        for callback in self.callbacks:
            callback.on_state_change(src_state, action, dst_state, reward)

    def on_wait(self) -> None:
        for callback in self.callbacks:
            callback.on_wait()

    def on_error(self) -> None:
        for callback in self.callbacks:
            callback.on_error()
