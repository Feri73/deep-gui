from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np


# maybe i can change all numpy usage to tensorflow

# better names
class EnvironmentCallbacks:
    def on_episode_start(self, state: np.ndarray) -> None:
        pass

    def on_episode_end(self) -> None:
        pass

    def on_state_change(self, src_state: np.ndarray, action: Any, dst_state: np.ndarray, reward: float) -> None:
        pass

    # for every action, this should be called exactly once, before on_state_change_us_called
    def on_wait(self) -> None:
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

    @abstractmethod
    def start(self):
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
