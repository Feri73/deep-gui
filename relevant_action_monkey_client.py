import random
from datetime import datetime
import os
import time
import socket
import subprocess
from typing import Any, Callable, Union

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

    def connect(self) -> None:
        while True:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect(('localhost', self.server_port))
                data = self.socket.recv(512)
                if len(data) > 0:
                    if data == b'OK:ping\n':
                        print(f'{datetime.now()}: received ping from {self.server_port}')
                        self.connected = True
                        return
                    raise NotImplementedError(f'expected ping! got {data}')
            except ConnectionRefusedError:
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
            self.connect()
        if self.current_state is None:
            screenshot_dir = os.path.abspath(f'.client_screenshots')
            screenshot_path = f'{screenshot_dir}/{self.adb_port}.png'
            self.adb(f'emu screenrecord screenshot {screenshot_path}')
            print(f'{datetime.now()}: took a screenshot from {self.server_port}')
            res = mpimg.imread(screenshot_path)[:, :, :-1]
            self.current_state = (res * 255).astype(np.uint8)
        return self.current_state

    def is_finished(self) -> bool:
        self.finished = not self.finished
        return self.finished

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
        self.current_state = None
        wait_action()
        return 0
