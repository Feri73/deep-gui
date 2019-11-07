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
