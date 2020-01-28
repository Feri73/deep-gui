import multiprocessing
import random
from functools import partial
from typing import Tuple, Any, Union, List

import numpy as np
import tensorflow as tf
import yaml

import phone
import relevant_action_base as base
from environment import EnvironmentCallbacks
from reinforcement_learning import RLAgent, RLModel, ModelStates, Episode
from relevant_action import RelevantActionEnvironment
from utils import Config


# this is ridiculous that i have to do this. this is because of the bad design in reinforcement_learning.py
class EmptyRLModel(RLModel):

    def calc_next_action(self, env_states_bt, bef_actions_bt, bef_rewards_bt, bef_states_eb: ModelStates) -> Tuple[
        Any, ModelStates]:
        pass

    def calc_loss(self, actions_bt, rewards_bt, model_states_ebt: ModelStates, finished_b) -> Tuple[Any, tuple]:
        pass

    def get_default_action(self):
        pass

    def get_default_reward(self) -> float:
        pass

    def get_default_states(self) -> ModelStates:
        return ()


class MonkeyAgent(RLAgent):
    def __init__(self, id: int, cfg: Config):
        action_type_count = cfg['action_type_count']
        self.debug_mode = cfg['debug_mode']
        self.grid_logs = cfg['grid_logs']
        self.steps_per_log = cfg['steps_per_log']
        self.action_length = np.prod(action_shape) * action_type_count
        self.summary_writer = tf.summary.FileWriter(f'{summary_path}/monkey')
        self.log_step = 0
        self.action2pos = action2pos
        self.trainable = False
        self.mean_episode_reward = []
        super().__init__(id, EmptyRLModel(), None, cfg, False)

    def calc_next_action(self, *args) -> Tuple[Any, ModelStates]:
        return [random.randint(0, self.action_length - 1)], ()

    def set_generated_gradient_target(self, target: 'RLAgent'):
        pass

    def calc_gradient(self, episode: Episode, states_teb: List[ModelStates]):
        pass

    def apply_gradient(self, gradients) -> None:
        pass

    def add_replacement_target(self, target: 'RLAgent') -> None:
        pass

    def replace_parameter(self, target: Union['RLAgent', Any]) -> None:
        pass

    def get_parameter(self):
        pass

    def log_episode(self, episode: Episode) -> None:
        base.Agent.log_episode(self, episode)

    def on_episode_end(self, premature: bool) -> None:
        base.Agent.on_episode_end(self, premature)

    def on_episode_start(self, env_state) -> None:
        base.Agent.on_episode_start(self, env_state)


def base_agent_creator(agent_id):
    return base.Agent(agent_id, None, None, base.most_probable_weighted_policy_user, action2pos, action_shape,
                      value_estimator_values[0], tf.Session().__enter__(), f'global', input_shape, False, cfg)

def monkey_agent_creator(agent_id):
    return MonkeyAgent(agent_id, cfg)


agent_creators = [base_agent_creator, monkey_agent_creator]


class VisitedActivityCallback(EnvironmentCallbacks):
    def __init__(self, phone, agent):
        self.phone = phone
        self.agent = agent
        self.step = 0

    def on_episode_end(self, premature: bool) -> None:
        self.step += 1
        summary = tf.Summary()
        summary.value.add(tag='Performance/ActivityCount', simple_value=len(self.phone.visited_activities))
        self.agent.summary_writer.add_summary(summary, self.step)
        self.agent.summary_writer.flush()


def create_agent(agent_id, agent_creator):
    try:
        agent = agent_creator(agent_id)
        environment = RelevantActionEnvironment(agent, Phone(f'device{agent_id}', 5554 + 2 * agent_id, cfg),
                                                action2pos, cfg)
        environment.add_callback(agent)
        environment.add_callback(VisitedActivityCallback(environment.phone, agent))
        environment.start()
    except:
        print(f'visited activity count in {agent_id}: {len(environment.phone.visited_activities)}')
        print(f'visited activities in {agent_id}: {environment.phone.visited_activities}')


with open('setting.yaml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

dummy_mode = cfg['dummy_mode']
inter_op_core_count = cfg['inter_op_core_count']
intra_op_core_count = cfg['intra_op_core_count']
agents_count = cfg['agents_count']
screen_shape = tuple(cfg['screen_shape'])
screen_new_shape = tuple(cfg['screen_new_shape'])
action_shape = tuple(cfg['action_shape'])
crop_top_left = cfg['crop_top_left']
crop_size = cfg['crop_size']
reset_summary = cfg['reset_summary']
summary_path = cfg['summary_path']
debug_mode = cfg['debug_mode']
value_estimator_values = cfg['value_estimator_values']
apks_path = cfg['apks_path']

cfg['load_model'] = True
cfg['maintain_visited_activities'] = True
cfg['shuffle'] = False
summary_path = cfg['summary_path'] = f'test_{summary_path}'
apks_path = cfg['apks_path'] = f'test_{apks_path}'

if dummy_mode:
    Phone = phone.DummyPhone
else:
    Phone = phone.Phone
# make these two optional
tf.config.threading.set_inter_op_parallelism_threads(inter_op_core_count)
tf.config.threading.set_intra_op_parallelism_threads(intra_op_core_count)
tf.disable_v2_behavior()
# should i set batch size to None or 1?
input_shape = (*screen_shape, 3)

action2pos = partial(base.policy2pos, screen_new_shape=screen_new_shape, crop_size=crop_size,
                     crop_top_left=crop_top_left, action_shape=action_shape)

if __name__ == '__main__':
    base.remove_summaries(summary_path, reset_summary)
    processes = []
    mp = multiprocessing.get_context('spawn')
    for agent_id, agent_creator in enumerate(agent_creators):
        processes += [mp.Process(name=f'learning agent #{agent_id}', target=create_agent,
                                 args=(agent_id, agent_creator))]
        processes[-1].start()
    for process in processes:
        process.join()
