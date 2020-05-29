import os
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

    def adb(self, command: str, as_bytes: bool = False) -> Union[str, bytes]:
        command = f'{self.adb_path} -s emulator-{self.adb_port} {command}'
        res = subprocess.check_output(command, shell=True)
        if not as_bytes:
            return res.decode('utf-8')
        return res

    def connect(self) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect(('localhost', self.server_port))
        data = self.socket.recv(512)
        if data == 'ping':
            return
        raise NotImplementedError('expected ping!')

    def restart(self) -> None:
        pass

    def read_state(self) -> np.ndarray:
        self.connect()
        screenshot_dir = os.path.abspath(f'.client_screenshots')
        screenshot_path = f'{screenshot_dir}/{self.adb_port}.png'
        self.adb(f'emu screenrecord screenshot {screenshot_path}')
        res = mpimg.imread(screenshot_path)[:, :, :-1]
        return (res * 255).astype(np.uint8)

    def is_finished(self) -> bool:
        return True

    def act(self, action: Any, wait_action: Callable) -> float:
        if action[2] != 0:
            raise NotImplementedError()
        action = self.action2pos(action)
        # read the output of monkey here: should be OK
        self.socket.send(f'touch down {action[0]} {action[1]}')
        self.socket.send(f'touch up {action[0]} {action[1]}')
        self.socket.send(f'done')
        self.socket.close()
        wait_action()
        return 0

