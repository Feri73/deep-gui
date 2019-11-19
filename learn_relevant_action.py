import os
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import yaml

from actor_critic import A2C
from coordinators import MultiprocessRLCoordinator, MultithreadRLCoordinator
import phone
from relevant_action import RelevantActionEnvironment
from screen_parsing import ScreenEncoder, PolicyGenerator, ValueEstimator

keras = tf.keras

from reinforcement_learning import RLAgent, RLCoordinator
from utils import Config


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
    def __init__(self, id: int, coordinator: RLCoordinator, agent_id: int, optimizer: keras.optimizers.Optimizer,
                 is_target: bool, input_shape: tuple, cfg: Config):
        self.screen_new_shape = tuple(cfg['screen_new_shape'])
        self.action_type_count = cfg['action_type_count']
        self.representation_size = cfg['representation_size']
        self.crop_top_left = cfg['crop_top_left']
        self.crop_size = cfg['crop_size']
        save_to_path = cfg['save_to_path']
        if is_target:
            environment = None
        else:
            environment = RelevantActionEnvironment(Phone(f'device{agent_id}', 5554 + 2 * agent_id, cfg), cfg)
        rl_model = A2C(ScreenEncoder(self.crop_top_left, self.crop_size,
                                     self.screen_new_shape, self.representation_size),
                       PolicyGenerator((*self.screen_new_shape, self.action_type_count)),
                       ValueEstimator(), weighted_policy_user, cfg)
        # if is_target:
        #     # plot this in tensorboard
        #     keras.utils.plot_model(rl_model, to_file='model.png', expand_nested=True)
        super(Agent, self).__init__(id, coordinator, environment, rl_model, optimizer,
                                    None if is_target else tf.summary.create_file_writer(f'summaries/agent{agent_id}'),
                                    cfg)
        if os.path.exists(save_to_path):
            self.build_model(input_shape)
            rl_model.load_weights(save_to_path)

    def realize_action(self, action: int) -> Tuple[int, int, int]:
        type = action // np.prod(self.screen_new_shape)
        scaled_y = (action // self.screen_new_shape[1]) % self.screen_new_shape[0]
        scaled_x = action % self.screen_new_shape[1]
        y, x = tuple((np.array(self.crop_size) / np.array(self.screen_new_shape) *
                      np.array([scaled_y, scaled_x]) + np.array(self.crop_top_left)).astype(np.int32))
        return x, y, type


def weighted_policy_user(policy: tf.Tensor) -> tf.Tensor:
    return tf.argmax(tfp.distributions.Multinomial(1, probs=policy).sample(), axis=-1)


with open('setting.yaml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

eager_mode = cfg['eager_mode']
dummy_mode = cfg['dummy_mode']
inter_op_core_count = cfg['inter_op_core_count']
intra_op_core_count = cfg['intra_op_core_count']
if eager_mode:
    tf.config.experimental_run_functions_eagerly(True)
if dummy_mode:
    Phone = phone.DummyPhone
else:
    Phone = phone.Phone
tf.config.threading.set_inter_op_parallelism_threads(inter_op_core_count)
tf.config.threading.set_intra_op_parallelism_threads(intra_op_core_count)

if __name__ == '__main__':
    agents_count = cfg['agents_count']
    screen_shape = tuple(cfg['screen_shape'])
    multiprocessing = cfg['multiprocessing']

    # should i set batch size to None or 1?
    input_shape = (1, *screen_shape, 3)


    def agent_info_factory(agent_id, is_target):
        # employ different optimizers for each agent
        return Agent, (agent_id, keras.optimizers.Adam(), is_target, input_shape, cfg)


    learning_agents_info = [agent_info_factory(i, False) for i in range(agents_count)]
    final_agent_info = agent_info_factory(len(learning_agents_info), True)

    if multiprocessing:
        coord = MultiprocessRLCoordinator(learning_agents_info, final_agent_info, input_shape, cfg)
    else:
        coord = MultithreadRLCoordinator(learning_agents_info, final_agent_info, input_shape, cfg)
    coord.start_learning()
