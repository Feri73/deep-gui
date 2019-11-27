import os
from typing import Callable, Dict, Any, List, Union

import numpy as np
import tensorflow as tf

# force it to be tf.function
StateProcessor = Callable[[np.ndarray], np.ndarray]
Config = Dict[str, Any]
Gradient = Union[List[tf.Tensor], int]


@tf.function
def add_gradients(gradient1: Gradient, gradient2: Gradient) -> Gradient:
    if gradient1 == 0:
        return gradient2
    if gradient2 == 0:
        return gradient1
    return [gradient1[i] + gradient2[i] for i in range(len(gradient1))]


# apparently because of range this does not run fully in tf
@tf.function
def discount(gamma, rewards):
    return tf.nn.conv1d(
        tf.pad(rewards, tf.constant([[0, 0], [0, rewards.shape[1] - 1], [0, 0]]), 'CONSTANT'),
        [[[gamma ** i]] for i in range(rewards.shape[1])], stride=1, padding='VALID')


def run_parallel_command(command: str) -> None:
    if os.name == 'nt':
        command = f'start /min {command}'
    else:
        command = f'{command} &'
    os.system(command)


class MemVariable:
    def __init__(self, init_value: Callable[[], Any], memory_size: int = 1):
        self.value = init_value()
        self.init_value = init_value
        self.memory_size = memory_size
        self.reset_archive()

    def archive(self) -> None:
        self.memory = (self.memory + [self.value])[:self.memory_size]
        self.value = self.init_value()

    def last_value(self, index: int = 0) -> Any:
        return self.memory[index]

    def reset_archive(self) -> None:
        self.memory = []

    def has_archive(self) -> bool:
        return len(self.memory) > 0

    def reset_value(self) -> None:
        self.value = self.init_value()


class MemList(MemVariable):
    def __init__(self, content: List[MemVariable]):
        self.content = content
        super().__init__(lambda: [c.value for c in self.content])

    def archive(self) -> None:
        for c in self.content:
            c.archive()
        self.value = self.init_value()

    def last_value(self, index: int = 0) -> List[Any]:
        return [c.memory[index] for c in self.content]

    def reset_archive(self) -> None:
        for c in self.content:
            c.reset_archive()

    def has_archive(self) -> bool:
        return sum([c.has_archive() for c in self.content]) == len(self.content)

    def reset_value(self) -> None:
        for c in self.content:
            c.reset_value()
