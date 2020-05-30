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
        if action[2] != 0:
            raise NotImplementedError()
        action = self.action2pos(action)
        print(f'{datetime.now()}: sending click on {action[0]}, {action[1]} to {self.server_port}')
        # read the output of monkey here: should be OK
        self.socket.send(f'touch down {action[0]} {action[1]}\n'.encode())
        self.socket.send(f'touch up {action[0]} {action[1]}\n'.encode())
        self.socket.send(f'done\n'.encode())
        self.disconnect()
        self.current_state = None
        wait_action()
        return 0

