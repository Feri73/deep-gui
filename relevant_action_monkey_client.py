import random
from datetime import datetime
import os
import time
import socket
import subprocess
import traceback
from typing import Any, Callable, Union, Optional

import numpy as np
import matplotlib.image as mpimg
from PIL import Image

from environment import Environment, EnvironmentController
from utils import Config


class RelevantActionMonkeyClient(Environment):
    def __init__(self, controller: EnvironmentController, action2pos: Callable,
                 server_port: int, adb_port: int, control_port: Optional[int], control_handler: Callable, cfg: Config):
        super().__init__(controller)
        self.action2pos = action2pos
        self.server_port = server_port
        self.adb_port = adb_port
        self.control_port = control_port
        self.control_handler = control_handler

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
        self.calculate_reward = cfg['calculate_reward']
        self.screen_shape = tuple(cfg['screen_shape'])

        self.socket = None
        self.control_socket = None
        self.pinged = False
        self.current_state = None
        self.finished = True
        self.true_screen_shape = None
        if not os.path.exists(f'.client_screenshots'):
            os.makedirs(f'.client_screenshots')

    def adb(self, command: str, as_bytes: bool = False) -> Union[str, bytes]:
        command = f'{self.adb_path} -s emulator-{self.adb_port} {command}'
        res = subprocess.check_output(command, shell=True)
        if not as_bytes:
            return res.decode('utf-8')
        return res

    def start(self):
        while True:
            try:
                super().start()
                break
            except Exception:
                print(f'{datetime.now()}: exception in {self.server_port}:\n{traceback.format_exc()}')
                self.on_error()

    def check_control(self) -> None:
        if self.control_port is None:
            return
        print(f'{datetime.now()}: probing control port in {self.server_port}.')
        if self.control_socket is None:
            self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.control_socket.setblocking(0)
            self.control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.control_socket.bind(('localhost', self.control_port))
            self.control_socket.listen()
        try:
            con, _ = self.control_socket.accept()
            data = con.recv(512, socket.MSG_DONTWAIT)
            if len(data) > 0:
                self.control_handler(data)
            con.close()
        except BlockingIOError:
            pass

    def connect(self, check_control: bool = False) -> None:
        while True:
            if check_control:
                self.check_control()
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect(('localhost', self.server_port))
                print(f'{datetime.now()}: conntected to server {self.server_port}.')
                return
            except ConnectionRefusedError:
                print(f'{datetime.now()}: server {self.server_port} is down. trying again in 0.5 seconds.')
                time.sleep(0.5)

    def receive(self, expected: Optional[str], blocking: bool = True) -> Optional[bytes]:
        try:
            if blocking:
                data = self.socket.recv(512)
            else:
                data = self.socket.recv(512, socket.MSG_DONTWAIT)
            if len(data) > 0:
                if expected is None or data == f'OK:{expected}\n'.encode():
                    print(f'{datetime.now()}: received {data} from {self.server_port}')
                    return data
                raise RuntimeError(f'expected {expected} got {data}')
            else:
                raise ConnectionRefusedError()
        except BlockingIOError:
            assert not blocking
            return None

    def send(self, data: str) -> None:
        self.socket.send((data + '\n').encode())

    def disconnect(self) -> None:
        self.send('done')
        self.socket.close()

    def restart(self) -> None:
        self.current_state = None
        self.finished = True

    def read_state(self) -> np.ndarray:
        if self.current_state is None:
            if self.pinged:
                self.connect()
            else:
                self.connect(check_control=True)
                while True:
                    try:
                        self.receive('ping')
                        self.pinged = True
                        break
                    except ConnectionRefusedError:
                        time.sleep(0.5)
                        self.connect(check_control=True)
            self.current_state = self.screenshot()
        return self.current_state

    def screenshot(self) -> np.ndarray:
        screenshot_dir = os.path.abspath(f'.client_screenshots')
        screenshot_path = f'{screenshot_dir}/{self.adb_port}.png'
        self.adb(f'emu screenrecord screenshot {screenshot_path}')
        print(f'{datetime.now()}: took a screenshot from {self.server_port}')
        res = mpimg.imread(screenshot_path)[:, :, :-1]
        self.true_screen_shape = res.shape[:2]
        res = (res * 255).astype(np.uint8)
        if self.true_screen_shape != self.screen_shape:
            res = np.array(Image.fromarray(res).resize((self.screen_shape[1], self.screen_shape[0])))
        return res

    def is_finished(self) -> bool:
        self.finished = not self.finished
        return self.finished

    def crop_state(self, state: np.ndarray) -> np.ndarray:
        return state[
               self.crop_top_left[0]:self.crop_top_left[0] + self.crop_size[0],
               self.crop_top_left[1]:self.crop_top_left[1] + self.crop_size[1]]

    def are_states_equal(self, s1: np.ndarray, s2: np.ndarray) -> bool:
        return np.linalg.norm(self.crop_state(s1) - self.crop_state(s2)) <= self.global_equality_threshold

    # I assume when in this function, the connection with server is live
    def act(self, action: Any, wait_action: Callable) -> float:
        self.pinged = False
        type = action[2]
        action = self.action2pos(action)
        y = int(action[1] * self.true_screen_shape[0] / self.screen_shape[0])
        x = int(action[0] * self.true_screen_shape[1] / self.screen_shape[1])
        action = (x, y, action[2])
        if type == 0:
            print(f'{datetime.now()}: sending click on {action[0]}, {action[1]} to {self.server_port}')
            # read the output of monkey here: should be OK
            self.send(f'touch down {action[0]} {action[1]}')
            self.send(f'touch up {action[0]} {action[1]}')
        elif type == 1:
            up_scroll = random.uniform(0, 1) > .5
            val = random.randint(self.scroll_min_value, self.scroll_max_value) * (-1) ** up_scroll
            x, y = action[0], action[1]
            print(f'{datetime.now()}: sending scroll {"up" if up_scroll else "down"} on {x},{y} '
                  f'to {self.server_port}')
            self.send(f'touch down {x} {y}')
            for _y in range(y + val // self.scroll_event_count, y + val, val // self.scroll_event_count):
                self.send(f'touch move {x} {int(_y)}')
            self.send(f'touch up {x} {int(_y)}')
        elif type == 2:
            left_scroll = random.uniform(0, 1) > .5
            val = random.randint(self.scroll_min_value, self.scroll_max_value) * (-1) ** left_scroll
            x, y = action[0], action[1]
            print(f'{datetime.now()}: sending swipe {"left" if left_scroll else "right"} on {x},{y} '
                  f'to {self.server_port}')
            self.send(f'touch down {x} {y}')
            for _x in range(x + val // self.scroll_event_count, x + val, val // self.scroll_event_count):
                self.send(f'touch move {int(_x)} {y}')
            self.send(f'touch up {int(_x)} {y}')
        else:
            raise NotImplementedError()
        self.disconnect()
        self.pinged = False

        self.connect()
        reward = 0
        try:
            if type == 0 and self.calculate_reward:
                while not self.receive('action_done', blocking=False):
                    shot = self.screenshot()
                    if not self.are_states_equal(shot, self.current_state):
                        reward = 1
                        self.receive('action_done')
                        break
                    time.sleep(self.screenshots_interval)
            else:
                self.receive('action_done')
        except RuntimeError:
            # I assume it is a ping
            self.pinged = True
            raise
        if self.calculate_reward:
            shot = self.screenshot()
            if not self.are_states_equal(shot, self.current_state):
                reward = 1
            self.current_state = shot
        else:
            reward = .5
        self.disconnect()

        wait_action()

        return reward * self.pos_reward + (1 - reward) * self.neg_reward
