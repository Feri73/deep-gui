import os
import random
import shutil
from functools import partial
from typing import Tuple, Optional, Callable

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
import yaml

import phone
from actor_critic import A2C
from coordinators import MultiprocessRLCoordinator, MultithreadRLCoordinator
from relevant_action import RelevantActionEnvironment
from screen_parsing import ScreenEncoder, PolicyGenerator, ValueEstimator

from reinforcement_learning import RLAgent, RLCoordinator, NoStateStatelessModel
from utils import Config


# is it possible to prevent retracing when shape changes?
#   (if i generate the input inside @tf.function it does not retrace)
# apparently i can use tf.function on all functions regardless of if it is tf control flow or now. use this to ditch
#   multiprocessing (did i mean retracing by this?)--> test it tho first (in a simple test file)
# do some of the stuff that people do to prevent high variance and instability in policy gradient
# look at all warnings
# v values can change during training --> steps_per_gradient_update can affect this
# maybe i can cache the trainable weights for speed
# add the logs for the final model as well
# note that the optimizers ar eof no use except for the final agent
# one idea is to have multiple agents that have different architectures with shared components, e.g. some use continuous
#     and some discrete actions
# check emulator sometimes it resets (generally, have a mechanism for resetting a worker when something fails)
# add action resolution parameter
# add an off policy buffer and add similar actions
# create a dependency yaml file
# add different policyUsers
# plot the network architecture
# plot images and scenarios in tensorboard
# save different version of the model (with different names, probably identifying the accuracy at the moment of saving)
# set name for each of the network elements
# how to run emulator on a specific core
# look at the todos and comments in the other project (code)
# test the network independent of the task (only on a couple of image in a trivial dataset that i create to test)
# maybe actor critic is not the best option here. look into q learning and REINFORCE
class Agent(RLAgent):
    def __init__(self, id: int, coordinator: Optional[RLCoordinator], optimizer: keras.optimizers.Optimizer,
                 policy_user: Callable[[tf.Tensor], tf.Tensor], input_shape: Tuple[int, ...], value: float,
                 cfg: Config):
        self.screen_new_shape = tuple(cfg['screen_new_shape'])
        self.action_type_count = cfg['action_type_count']
        self.representation_size = cfg['representation_size']
        self.crop_top_left = cfg['crop_top_left']
        self.crop_size = cfg['crop_size']
        save_to_path = cfg['save_to_path']
        summary_path = cfg['summary_path']
        self.steps_per_log = cfg['steps_per_log']

        self.mean_episode_reward = keras.metrics.Mean(dtype=tf.float32)
        self.mean_loss = keras.metrics.Mean(dtype=tf.float32)
        self.mean_gradient_norm = keras.metrics.Mean(dtype=tf.float32)
        self.mean_weights_norm = keras.metrics.Mean(dtype=tf.float32)

        rl_model = A2C(NoStateStatelessModel(ScreenEncoder(self.crop_top_left, self.crop_size,
                                                           self.screen_new_shape, self.representation_size)),
                       NoStateStatelessModel(PolicyGenerator((*self.screen_new_shape, self.action_type_count))),
                       NoStateStatelessModel(ValueEstimator(value)), policy_user, cfg)

        self.summary_writer = tf.summary.create_file_writer(f'{summary_path}/agent{id}')

        super(Agent, self).__init__(id, rl_model, self.realize_action, coordinator, optimizer, cfg)

        if os.path.exists(save_to_path):
            self.build_model(input_shape)
            rl_model.load_weights(save_to_path)

    def on_episode_gradient_computed(self, loss, gradient, state_history,
                                     realized_action_history, reward_history) -> None:
        self.mean_episode_reward.update_state(sum(reward_history))
        self.mean_loss.update_state(loss)
        # self.mean_gradient_norm.update_state(tf.linalg.global_norm(gradient))
        # self.mean_weights_norm.update_state(tf.linalg.global_norm(self.rl_model.trainable_weights))
        with self.summary_writer.as_default():
            if self.step % self.steps_per_log == 0:
                # use callbacks here
                tf.summary.scalar('RLAgent/mean episode reward', self.mean_episode_reward.result(), self.step)
                tf.summary.scalar('RLAgent/mean loss', self.mean_loss.result(), self.step)
                # tf.summary.scalar('RLAgent/gradient', self.mean_gradient_norm.result(), self.step)
                # tf.summary.scalar('RLAgent/weights', self.mean_weights_norm.result(), self.step)
                for metric_name, metric_value in self.rl_model.get_log_values():
                    tf.summary.scalar(f'RLModel/{metric_name}', metric_value, self.step)
                self.mean_episode_reward.reset_states()
                self.mean_loss.reset_states()
                self.mean_gradient_norm.reset_states()
                self.mean_weights_norm.reset_states()

            action = realized_action_history[0]
            state_history = state_history.copy()
            state_history[0][max(action[1] - 4, 0):action[1] + 4, max(action[0] - 4, 0): action[0] + 4, :] = [255, 0, 0]
            # how come here i don't need to use self.summary_writer??
            tf.summary.image('episode', state_history[0:1], self.step)

    def realize_action(self, action: np.ndarray) -> Tuple[int, int, int]:
        action = int(action)
        type = action // np.prod(self.screen_new_shape)
        scaled_y = (action // self.screen_new_shape[1]) % self.screen_new_shape[0]
        scaled_x = action % self.screen_new_shape[1]
        y, x = tuple((np.array(self.crop_size) / np.array(self.screen_new_shape) *
                      np.array([scaled_y, scaled_x]) + np.array(self.crop_top_left)).astype(np.int32))
        return x, y, type


def most_probable_weighted_policy_user(policy: tf.Tensor) -> tf.Tensor:
    return tf.argmax(tfp.distributions.Multinomial(1, probs=policy).sample(), axis=-1)


def least_probable_weighted_policy_user(policy: tf.Tensor) -> tf.Tensor:
    probs = 1 / policy
    probs = probs / tf.reduce_sum(probs, axis=-1)
    return tf.argmax(tfp.distributions.Multinomial(1, probs=probs).sample(), axis=-1)


def least_certain_weighted_policy_user(policy: tf.Tensor) -> tf.Tensor:
    probs = 1 / tf.abs(.5 - policy)
    probs = probs / tf.reduce_sum(probs, axis=-1)
    return tf.argmax(tfp.distributions.Multinomial(1, probs=probs).sample(), axis=-1)


# for using this, i have to change the v based on whether the most certain is >.5 or <.5
def most_certain_weighted_policy_user(policy: tf.Tensor) -> tf.Tensor:
    probs = tf.abs(.5 - policy)
    probs = probs / tf.reduce_sum(probs, axis=-1)
    return tf.argmax(tfp.distributions.Multinomial(1, probs=probs).sample(), axis=-1)


def random_policy_user(policy: tf.Tensor) -> tf.Tensor:
    return tf.argmax(tfp.distributions.Multinomial(1, probs=tf.ones_like(policy) / tf.shape(policy)[-1]).sample(),
                     axis=-1)


# policy_users = [(most_probable_weighted_policy_user, 1), (least_probable_weighted_policy_user, 0),
#                 (least_certain_weighted_policy_user, .5)]
policy_users = [(least_certain_weighted_policy_user, .5)]
optimizers = [keras.optimizers.Adam, keras.optimizers.RMSprop]


def create_agent(agent_id, is_target, coord):
    policy_user, value = policy_users[agent_id % len(policy_users)]
    optimizer = random.choice(optimizers)
    print(f'crating agent with policy_user={policy_user.__name__}, value={value}, optimizer={optimizer.__name__}')
    agent = Agent(agent_id, coord, optimizer(), policy_user, input_shape, value, cfg)
    if is_target:
        environment = None
    else:
        environment = RelevantActionEnvironment([agent], agent,
                                                Phone(f'device{agent_id}', 5554 + 2 * agent_id, cfg), cfg)
    return environment, agent


with open('setting.yaml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

eager_mode = cfg['eager_mode']
dummy_mode = cfg['dummy_mode']
inter_op_core_count = cfg['inter_op_core_count']
intra_op_core_count = cfg['intra_op_core_count']
agents_count = cfg['agents_count']
screen_shape = tuple(cfg['screen_shape'])
if eager_mode:
    tf.config.experimental_run_functions_eagerly(True)
if dummy_mode:
    Phone = phone.DummyPhone
else:
    Phone = phone.Phone
tf.config.threading.set_inter_op_parallelism_threads(inter_op_core_count)
tf.config.threading.set_intra_op_parallelism_threads(intra_op_core_count)
# should i set batch size to None or 1?
input_shape = (1, *screen_shape, 3)

if __name__ == '__main__':
    multiprocessing = cfg['multiprocessing']
    reset_summary = cfg['reset_summary']
    summary_path = cfg['summary_path']

    try:
        if reset_summary:
            shutil.rmtree(summary_path)
    except FileNotFoundError:
        pass

    learning_agent_creators = [partial(create_agent, i, False) for i in range(agents_count)]
    final_agent_creator = partial(create_agent, len(learning_agent_creators), True)

    if multiprocessing:
        coord = MultiprocessRLCoordinator(learning_agent_creators, final_agent_creator, input_shape, cfg)
    else:
        coord = MultithreadRLCoordinator(learning_agent_creators, final_agent_creator, input_shape, cfg)
    coord.start_learning()
