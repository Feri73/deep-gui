from abc import ABC, abstractmethod
from typing import Any, Callable

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

    def on_episode_end(self, premature: bool) -> None:
        pass

    def on_error(self) -> None:
        pass

    def on_environment_finished(self) -> None:
        pass


class EnvironmentController(ABC):
    def should_continue_episode(self) -> bool:
        return True

    @abstractmethod
    def should_start_episode(self) -> bool:
        pass

    @abstractmethod
    def get_next_action(self, state: np.ndarray) -> Any:
        pass


class Environment(ABC):
    def __init__(self, controller: EnvironmentController):
        self.callbacks = []
        self.controller = controller
        self.stopped = False

    def add_callback(self, callback: EnvironmentCallbacks) -> None:
        self.callbacks += [callback]

    def stop(self):
        self.stopped = True

    def start(self):
        self.stopped = False
        while self.should_start_episode() and not self.stopped:
            self.restart()
            cur_state = self.read_state()
            self.on_episode_start(cur_state)
            premature = False
            while not self.is_finished():
                if self.stopped:
                    self.on_environment_finished()
                    return
                if not self.should_continue_episode():
                    premature = True
                    break
                last_state = cur_state
                action = self.get_next_action(last_state)
                # add something to on_wait to make sure it is called, if not, call it here (but also write a warning)
                reward = self.act(action, self.on_wait)
                cur_state = self.read_state()
                self.on_state_change(last_state, action, cur_state, reward)
            self.on_episode_end(premature)
        self.on_environment_finished()

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

    def should_continue_episode(self) -> bool:
        return self.controller.should_continue_episode()

    def should_start_episode(self) -> bool:
        return self.controller.should_start_episode()

    def get_next_action(self, state: np.ndarray) -> Any:
        return self.controller.get_next_action(state)

    def on_episode_start(self, state: np.ndarray) -> None:
        for callback in self.callbacks:
            callback.on_episode_start(state)

    def on_episode_end(self, premature: bool) -> None:
        for callback in self.callbacks:
            callback.on_episode_end(premature)

    def on_state_change(self, src_state: np.ndarray, action: Any, dst_state: np.ndarray, reward: float) -> None:
        for callback in self.callbacks:
            callback.on_state_change(src_state, action, dst_state, reward)

    def on_wait(self) -> None:
        for callback in self.callbacks:
            callback.on_wait()

    def on_error(self) -> None:
        for callback in self.callbacks:
            callback.on_error()

    def on_environment_finished(self) -> None:
        for callback in self.callbacks:
            callback.on_environment_finished()
