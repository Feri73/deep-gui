import random
from datetime import datetime
import os
import time
import socket
import subprocess
from typing import Any, Callable, Union, Optional

import numpy as np
import matplotlib.image as mpimg

from environment import Environment, EnvironmentController
from utils import Config


class RelevantActionMonkeyClient(Environment):
    def __init__(self, controller: EnvironmentController, action2pos: Callable,
                 server_port: int, adb_port: int, cfg: Config):
        super().__init__(controller)
        self.action2pos = action2pos
        self.server_port = server_port
        self.adb_port = adb_port

        self.adb_path = cfg['adb_path']
        self.scroll_min_value = cfg['scroll_min_value']
        self.scroll_max_value = cfg['scroll_max_value']
        self.scroll_event_count = cfg['scroll_event_count']
        self.crop_top_left = cfg['crop_top_left']
        self.crop_size = cfg['crop_size']
        self.pos_reward = cfg['pos_reward']
        self.neg_reward = cfg['neg_reward']
        self.screenshots_interval = cfg['screenshots_interval']
        self.global_equality_threshold = cfg['global_equality_threshold']

        self.socket = None
        self.connected = False
        self.current_state = None
        self.finished = True
        if not os.path.exists(f'.client_screenshots'):
            os.makedirs(f'.client_screenshots')

    def adb(self, command: str, as_bytes: bool = False) -> Union[str, bytes]:
        command = f'{self.adb_path} -s emulator-{self.adb_port} {command}'
        res = subprocess.check_output(command, shell=True)
        if not as_bytes:
            return res.decode('utf-8')
        return res

    def connect(self, expected: Optional[str], blocking: bool = True) -> bool:
        while True:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                if not blocking:
                    self.socket.setblocking(blocking)
                self.socket.connect(('localhost', self.server_port))
                data = self.socket.recv(512)
                if len(data) > 0:
                    if expected is None or data == f'OK:{expected}\n'.encode():
                        print(f'{datetime.now()}: received {data} from {self.server_port}')
                        self.connected = True
                        return True
                    raise NotImplementedError(f'expected ping! got {data}')
                elif not blocking:
                    return False
            except ConnectionRefusedError:
                if not blocking:
                    return False
                pass
            print(f'{datetime.now()}: server {self.server_port} is down. trying again in 0.5 seconds.')
            time.sleep(0.5)
            continue

    def disconnect(self) -> None:
        self.socket.close()
        self.connected = False

    def restart(self) -> None:
        pass

    def read_state(self) -> np.ndarray:
        if not self.connected:
            self.connect('ping')
        if self.current_state is None:
            self.current_state = self.screenshot()
        return self.current_state

    def screenshot(self) -> np.ndarray:
        screenshot_dir = os.path.abspath(f'.client_screenshots')
        screenshot_path = f'{screenshot_dir}/{self.adb_port}.png'
        self.adb(f'emu screenrecord screenshot {screenshot_path}')
        print(f'{datetime.now()}: took a screenshot from {self.server_port}')
        res = mpimg.imread(screenshot_path)[:, :, :-1]
        return (res * 255).astype(np.uint8)

    def is_finished(self) -> bool:
        self.finished = not self.finished
        return self.finished

    def crop_state(self, state: np.ndarray) -> np.ndarray:
        return state[
               self.crop_top_left[0]:self.crop_top_left[0] + self.crop_size[0],
               self.crop_top_left[1]:self.crop_top_left[1] + self.crop_size[1]]

    def are_states_equal(self, s1: np.ndarray, s2: np.ndarray) -> bool:
        return np.linalg.norm(self.crop_state(s1) - self.crop_state(s2)) <= self.global_equality_threshold

    def act(self, action: Any, wait_action: Callable) -> float:
        type = action[2]
        action = self.action2pos(action)
        if type == 0:
            print(f'{datetime.now()}: sending click on {action[0]}, {action[1]} to {self.server_port}')
            # read the output of monkey here: should be OK
            self.socket.send(f'touch down {action[0]} {action[1]}\n'.encode())
            self.socket.send(f'touch up {action[0]} {action[1]}\n'.encode())
        elif type == 1:
            up_scroll = random.uniform(0, 1) > .5
            val = random.randint(self.scroll_min_value, self.scroll_max_value) * (-1) ** up_scroll
            x, y = action[0], action[1]
            print(f'{datetime.now()}: sending scroll {"up" if up_scroll else "down"} on {x},{y}'
                  f'to {self.server_port}')
            self.socket.send(f'touch down {x} {y}\n'.encode())
            for _y in range(y + val // self.scroll_event_count, y + val, val // self.scroll_event_count):
                self.socket.send(f'touch move {x} {int(_y)}\n'.encode())
            self.socket.send(f'touch up {x} {int(_y)}\n'.encode())
        elif type == 2:
            left_scroll = random.uniform(0, 1) > .5
            val = random.randint(self.scroll_min_value, self.scroll_max_value) * (-1) ** left_scroll
            x, y = action[0], action[1]
            print(f'{datetime.now()}: sending swipe {"left" if left_scroll else "right"} on {x},{y}'
                  f'to {self.server_port}')
            self.socket.send(f'touch down {x} {y}\n'.encode())
            for _x in range(x + val // self.scroll_event_count, x + val, val // self.scroll_event_count):
                self.socket.send(f'touch move {int(_x)} {y}\n'.encode())
            self.socket.send(f'touch up {int(_x)} {y}\n'.encode())
        else:
            raise NotImplementedError()
        self.socket.send(f'done\n'.encode())
        self.disconnect()

        reward = 0
        if type == 0:
            while not self.connect('action_done', blocking=False):
                self.disconnect()
                shot = self.screenshot()
                if not self.are_states_equal(shot, self.current_state):
                    reward = 1
                    break
                time.sleep(self.screenshots_interval)
        else:
            self.connect('action_done')
        shot = self.screenshot()
        if not self.are_states_equal(shot, self.current_state):
            reward = 1
        self.socket.send(f'done\n'.encode())
        self.disconnect()

        self.current_state = None
        wait_action()

        return reward * self.pos_reward + (1 - reward) * self.neg_reward
