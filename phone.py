from datetime import datetime
import os
import re
import subprocess
import time
from typing import Union

import matplotlib.image as mpimg
import numpy as np

import glob

from utils import Config


class Phone:
    def __init__(self, device_name: str, port: int, cfg: Config):
        self.device_name = device_name
        self.port = port
        self.emulator_path = cfg['emulator_path']
        self.adb_path = cfg['adb_path']
        self.phone_start_wait_time = cfg['phone_start_wait_time']
        self.phone_restart_wait_time = cfg['phone_restart_wait_time']
        self.snapshot_load_wait_time = cfg['snapshot_load_wait_time']
        # self.screenshot_trials = cfg['screenshot_trials']
        self.avd_path = cfg['avd_path']
        apks_path = cfg['apks_path']
        self.aapt_path = cfg['aapt_path']
        self.app_activity_dict = {}
        self.apk_names = glob.glob(f'{apks_path}/*.apk')
        self.app_names = [self.get_app_name(apk_path) for apk_path in self.apk_names]
        if not os.path.exists(f'tmp-{device_name}'):
            os.makedirs(f'tmp-{device_name}')

    def adb(self, command: str, as_bytes: bool = False) -> Union[str, bytes]:
        command = f'{self.adb_path} -s emulator-{self.port} {command}'
        res = subprocess.check_output(command, shell=True)
        if not as_bytes:
            return res.decode('utf-8')
        return res

    def start_phone(self) -> None:
        # ref_snapshot_path = f'{self.avd_path}/snapshots/fresh'
        local_snapshot_path = f'{self.avd_path}/{self.device_name}.avd/snapshots/fresh'
        self.start_emulator()
        # if os.path.exists(ref_snapshot_path):
        #     if not os.path.exists(local_snapshot_path):
        #         copy_tree(ref_snapshot_path, local_snapshot_path)
        if os.path.exists(local_snapshot_path):
            # use -wipe-data instead
            self.load_snapshot('fresh')
        else:
            self.initial_setups()
            self.save_snapshot('fresh')
            # copy_tree(local_snapshot_path, ref_snapshot_path)

    def start_emulator(self) -> None:
        command = f'{self.emulator_path} -avd {self.device_name} -ports {self.port},{self.port + 1}'
        if os.name == 'nt':
            command = f'start /min {command}'
        else:
            command = f'{command} &'
        os.system(command)
        time.sleep(self.phone_start_wait_time)

    def initial_setups(self) -> None:
        # now that I've updated adb see if i can use this again
        # apks = ' '.join(self.apk_names)
        # self.adb(f'install-multi-package --instant "{apks}"')
        for apk_name in self.apk_names:
            self.adb(f'install -r "{os.path.abspath(apk_name)}"')

        self.adb('shell settings put global window_animation_scale 0')
        self.adb('shell settings put global transition_animation_scale 0')
        self.adb('shell settings put global animator_duration_scale 0')

    def get_app_name(self, apk_path: str) -> str:
        apk_path = os.path.abspath(apk_path)
        command = f'{self.aapt_path} dump badging "{apk_path}" | grep package'
        res = subprocess.check_output(command, shell=True).decode('utf-8')
        regex = re.compile(r'name=\'([^\']+)\'')
        return regex.search(res).group(1)

    def save_snapshot(self, name: str) -> None:
        self.adb(f'emu avd snapshot save {name}')

    def load_snapshot(self, name: str) -> None:
        if self.snapshot_load_wait_time >= 0:
            self.adb(f'emu avd snapshot load {name}')
            time.sleep(self.snapshot_load_wait_time)
            self.sync_time()

    def sync_time(self):
        self.adb('shell su root date ' + datetime.now().strftime('%m%d%H%M%Y.%S'))

    def close_app(self, app_name: str) -> None:
        self.adb(f'shell su root pm clear {app_name}')

    def add_app_activity(self, app_name: str) -> None:
        dat = self.adb(f'shell dumpsys package {app_name} | grep -A1 "android.intent.action.MAIN:"')
        lines = dat.splitlines()
        activityRE = re.compile('([A-Za-z0-9_.]+/[A-Za-z0-9_.]+)')
        self.app_activity_dict[app_name] = activityRE.search(lines[1]).group(1)

    def open_app(self, app_name: str) -> None:
        if app_name not in self.app_activity_dict:
            self.add_app_activity(app_name)
        self.adb(f'shell am start -n {self.app_activity_dict[app_name]}')

    # check the output (correct image is passed) because the code is changed
    def screenshot(self) -> np.ndarray:
        screenshot_dir = os.path.abspath(f'tmp-{self.device_name}')
        self.adb(f'emu screenrecord screenshot {screenshot_dir}')
        image_path = glob.glob(f'tmp-{self.device_name}/Screenshot*.png')[0]
        res = mpimg.imread(image_path)[:, :, :-1]
        os.remove(image_path)
        return res

    def send_event(self, x: int, y: int, type: int) -> None:
        if type != 0:
            raise NotImplementedError()
        # better logging
        print(f'{datetime.now()}: phone {self.device_name}: click on {x},{y}')
        self.adb(f'emu event mouse {x} {y} 0 1')
        self.adb(f'emu event mouse {x} {y} 0 0')
