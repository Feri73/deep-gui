from typing import Tuple, Callable, Any

import numpy as np

from environment import Environment
from phone import Phone
from utils import Config

import matplotlib.pyplot as plt

import time as tm


# some parts of this should be factorized to a generalized class
class RelevantActionEnvironment(Environment):
    def __init__(self, phone: Phone, cfg: Config):
        self.phone = phone
        self.time = 0
        self.current_state = None
        self.steps_per_app = cfg['steps_per_app']
        self.steps_per_episode = cfg['steps_per_episode']
        self.state_equality_epsilon = cfg['state_equality_epsilon']
        self.action_wait_time = cfg['action_wait_time']
        self.app_start_wait_time = cfg['app_start_wait_time']
        self.crop_top_left = cfg['crop_top_left']
        self.crop_size = cfg['crop_size']
        self.cur_app_index = -1
        self.finished = False
        self.last_state = None
        assert self.steps_per_app % self.steps_per_episode == 0
        self.phone.start_phone()

    def restart(self) -> None:
        self.finished = False
        if self.time % self.steps_per_app == 0:
            self.time = 0
            self.phone.close_app(self.phone.app_names[self.cur_app_index])
            self.cur_app_index = (self.cur_app_index + 1) % len(self.phone.app_names)
            # if self.cur_app_index == 0:
            #     self.phone.load_snapshot('fresh')
            self.phone.open_app(self.phone.app_names[self.cur_app_index])
            tm.sleep(self.app_start_wait_time)
            self.last_state = self.phone.screenshot()

    def is_finished(self) -> bool:
        return self.finished

    def read_state(self) -> np.ndarray:
        return self.last_state

    def crop_state(self, state: np.ndarray) -> np.ndarray:
        return state[
               self.crop_top_left[0]:self.crop_top_left[0] + self.crop_size[0],
               self.crop_top_left[1]:self.crop_top_left[1] + self.crop_size[1]]

    def are_states_equal(self, s1: np.ndarray, s2: np.ndarray) -> bool:
        return np.linalg.norm(self.crop_state(s1) - self.crop_state(s2)) <= self.state_equality_epsilon

    # extend to actions other than click
    # remember to check if the phone is still in the correct app and other wise restart it
    # look at the phone (in dev mode) to make sure the click positions are correctly generated (realize action)
    #   and sent (env and phone (debug these two independently))
    # maybe instead of 2d discrete actions i can have continuous actions (read a3c paper for continuous actions)
    def act(self, action: Tuple[int, int, int], computation: Callable[[], Any]) -> Tuple[float, Any]:
        self.time += 1
        if self.time % self.steps_per_episode == 0:
            self.finished = True
        # st = tm.time()
        self.phone.send_event(*action)
        # print('action:', tm.time() - st)

        precomp_time = tm.time()
        comp_res = computation()
        poscomp_time = tm.time()
        comp_time = poscomp_time - precomp_time
        if self.action_wait_time - comp_time > 0:
            # print('wait:', self.action_wait_time - comp_time)
            tm.sleep(self.action_wait_time - comp_time)

        # st = tm.time()
        cur_state = self.phone.screenshot()
        # print('screen:', tm.time() - st)
        reward = float(not self.are_states_equal(cur_state, self.last_state))

        # st = np.array(cur_state)
        # st[max(action[1] - 4, 0):action[1] + 4, max(action[0] - 4, 0):action[0] + 4, :] = [255, 0, 0]
        # plt.imshow(st)
        # plt.show()
        # if reward > 0:
        #     plt.imshow(self.last_state)
        #     plt.figure()
        #     plt.imshow(cur_state)
        #     print(np.linalg.norm(self.last_state - cur_state))
        #     plt.show()

        self.last_state = cur_state
        return reward, comp_res
