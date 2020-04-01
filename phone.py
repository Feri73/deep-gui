from datetime import datetime
import os
import re
import subprocess
import time
from typing import Union, Optional

import matplotlib.image as mpimg
import numpy as np

import glob

from utils import Config, run_parallel_command


# prints here should be centralized in a logger
class Phone:
    def __init__(self, device_name: str, port: int, cfg: Config):
        self.device_name = device_name
        self.port = port
        self.emulator_path = cfg['emulator_path']
        self.adb_path = cfg['adb_path']
        self.app_start_wait_time = cfg['app_start_wait_time']
        self.app_exit_wait_time = cfg['app_exit_wait_time']
        self.phone_boot_wait_time = cfg['phone_boot_wait_time']
        self.snapshot_load_wait_time = cfg['snapshot_load_wait_time']
        # self.screenshot_trials = cfg['screenshot_trials']
        self.avd_path = cfg['avd_path']
        apks_path = cfg['apks_path']
        self.aapt_path = cfg['aapt_path']
        self.clone_script_path = cfg['clone_script_path']
        self.install_apks = cfg['install_apks']
        self.maintain_visited_activities = cfg['maintain_visited_activities']
        self.app_activity_dict = {}
        self.apk_names = glob.glob(f'{apks_path}/*.apk')
        self.app_names = [self.get_app_name(apk_path) for apk_path in self.apk_names]
        self.apk_names, self.app_names = zip(*[x for x in zip(self.apk_names, self.app_names) if x[1] is not None])
        self.app_names = list(self.app_names)
        self.apk_names = list(self.apk_names)
        if not os.path.exists(f'tmp-{device_name}'):
            os.makedirs(f'tmp-{device_name}')
        self.step = 0
        self.visited_activities = set()
        self.app_in_stack = None

    def adb(self, command: str, as_bytes: bool = False) -> Union[str, bytes]:
        command = f'{self.adb_path} -s emulator-{self.port} {command}'
        res = subprocess.check_output(command, shell=True)
        if not as_bytes:
            return res.decode('utf-8')
        return res

    def maintain_current_activity(self):
        try:
            tmp = self.adb('shell "dumpsys activity activities | grep mResumedActivity"').strip()
            match = re.match('.*ActivityRecord{.+ .+ (.+) .+}.*', tmp)
            print(f'{datetime.now()}: activity {match[1]} is visited in {self.device_name}')
            self.visited_activities.add(match[1])
        except Exception as ex:
            print(f'{datetime.now()}: exception happened while maintaining current activity -> {ex}')

    def is_in_app(self, app_name: str, force_front: bool) -> bool:
        try:
            # add timeout here
            res = self.adb('shell "dumpsys activity | grep TaskRecord"')
            matches = re.findall(r'.*\* Recent .+: TaskRecord{.+#\d+ .+=(.+) .+StackId=(\d+).*}', res)
            print(f'{datetime.now()}: top app of {self.device_name}: {matches[0][0]}')
            # test for when force_front = False
            top_stack_id = matches[0][1]
            for match in matches:
                if match[0] == app_name and match[1] == top_stack_id:
                    self.app_in_stack = app_name
            if force_front:
                return matches[0][0] == app_name
            else:
                return self.app_in_stack == app_name
        except Exception:
            pass
        return False

    def is_booted(self):
        print(f'{datetime.now()}: checking boot status of {self.device_name}')
        try:
            # this 2s should be param probably
            return self.adb('shell timeout 2s getprop sys.boot_completed') == ('1\r\n' if os.name == 'nt' else '1\n')
        except subprocess.CalledProcessError:
            return False

    def wait_for_start(self) -> None:
        self.adb('wait-for-device')
        while not self.is_booted():
            time.sleep(2)
        time.sleep(self.phone_boot_wait_time)

    def restart(self, recreate_phone: bool = False):
        print(f'{datetime.now()}: restarting {self.device_name}')
        self.adb('emu kill')
        # this is not a good way of checking if the phone is off. because the phone may be already starting,
        #    not completely booted tho. this means that i try to start the phone twice.
        while self.is_booted():
            time.sleep(1)
        if recreate_phone:
            self.recreate_emulator()
        self.start_phone(True)

    def start_phone(self, fresh: bool = False) -> None:
        # ref_snapshot_path = f'{self.avd_path}/snapshots/fresh'
        local_snapshot_path = f'{self.avd_path}/{self.device_name}.avd/snapshots/fresh'
        self.start_emulator(fresh)
        # if os.path.exists(ref_snapshot_path):
        #     if not os.path.exists(local_snapshot_path):
        #         copy_tree(ref_snapshot_path, local_snapshot_path)
        if os.path.exists(os.path.expanduser(local_snapshot_path)):
            # use -wipe-data instead
            self.load_snapshot('fresh')
        else:
            self.initial_setups()
            self.save_snapshot('fresh')
            # copy_tree(local_snapshot_path, ref_snapshot_path)

    def recreate_emulator(self) -> None:
        print(f'{datetime.now()}: recreating emulator for {self.device_name}')
        while True:
            try:
                os.remove(f'{self.avd_path}/{self.device_name}.ini')
                os.rmdir(f'{self.avd_path}/{self.device_name}.avd/')
                break
            except:
                pass
        os.system(f'{self.clone_script_path} {self.device_name}')

    def start_emulator(self, fresh: bool = False) -> None:
        print(f'{datetime.now()}: starting emulator {self.device_name}. fresh={fresh}')
        run_parallel_command(f'{self.emulator_path} -avd {self.device_name} -ports {self.port},{self.port + 1}' +
                             (f' -no-cache' if fresh else ''))
        self.wait_for_start()

    def install_apk(self, apk_name: str) -> None:
        self.adb(f'install -r -g "{os.path.abspath(apk_name)}"')

    def initial_setups(self) -> None:
        # now that I've updated adb see if i can use this again
        # apks = ' '.join(self.apk_names)
        # self.adb(f'install-multi-package --instant "{apks}"')

        if self.install_apks:
            for apk_name, app_name in list(zip(self.apk_names, self.app_names)):
                try:
                    print(f'{datetime.now()}: installing {apk_name} in {self.device_name}')
                    self.install_apk(apk_name)
                except Exception:
                    print(f'{datetime.now()}: couldn\'t install {apk_name}. removing it')
                    self.apk_names.remove(apk_name)
                    self.app_names.remove(app_name)

        # self.adb('shell settings put global window_animation_scale 0')
        # self.adb('shell settings put global transition_animation_scale 0')
        # self.adb('shell settings put global animator_duration_scale 0')

    def get_app_name(self, apk_path: str) -> Optional[str]:
        try:
            apk_path = os.path.abspath(apk_path)
            command = f'{self.aapt_path} dump badging "{apk_path}" | grep package'
            res = subprocess.check_output(command, shell=True).decode('utf-8')
            regex = re.compile(r'name=\'([^\']+)\'')
            return regex.search(res).group(1)
        except Exception as ex:
            print(f'{datetime.now()}: could not get app name for {apk_path} in {self.device_name} because of {ex}')
            return None

    def save_snapshot(self, name: str) -> None:
        self.adb(f'emu avd snapshot save {name}')

    def load_snapshot(self, name: str) -> None:
        if self.snapshot_load_wait_time >= 0:
            self.adb(f'emu avd snapshot load {name}')
            time.sleep(self.snapshot_load_wait_time)
            self.sync_time()

    def sync_time(self):
        self.adb('shell su root date ' + datetime.now().strftime('%m%d%H%M%Y.%S'))

    def close_app(self, app_name: str, reset_maintained_activities: bool = True) -> None:
        print(f'{datetime.now()}: closing {app_name} in {self.device_name}')
        # self.adb(f'shell su root pm clear {app_name}')
        self.adb(f'shell am force-stop {app_name}')
        time.sleep(self.app_exit_wait_time)
        if reset_maintained_activities:
            self.visited_activities = set()

    def add_app_activity(self, app_name: str) -> None:
        dat = self.adb(f'shell dumpsys package {app_name} | grep -A1 "android.intent.action.MAIN:"')
        lines = dat.splitlines()
        activityRE = re.compile('([A-Za-z0-9_.]+/[A-Za-z0-9_.]+)')
        self.app_activity_dict[app_name] = activityRE.search(lines[1]).group(1)

    def open_app(self, app_name: str) -> None:
        print(f'{datetime.now()}: opening {app_name} in {self.device_name}')
        if app_name not in self.app_activity_dict:
            self.add_app_activity(app_name)
        self.adb(f'shell am start -n {self.app_activity_dict[app_name]}')
        # this is not the best way i can do it, cuz it needs to make sure i call is_in_app every time i call this
        if self.app_in_stack == app_name:
            print(f'{datetime.now()}: not waiting because the app was already in the stack in #{self.device_name}')
        else:
            time.sleep(self.app_start_wait_time)

    def screenshot(self, perform_checks: bool = False) -> np.ndarray:
        if self.maintain_visited_activities and perform_checks:
            self.maintain_current_activity()
        self.step += 1
        screenshot_dir = os.path.abspath(f'tmp-{self.device_name}')
        self.adb(f'emu screenrecord screenshot {screenshot_dir}')
        image_path = glob.glob(f'tmp-{self.device_name}/Screenshot*.png')[0]
        res = mpimg.imread(image_path)[:, :, :-1]
        os.remove(image_path)
        return (res * 255).astype(np.uint8)

    def send_event(self, x: int, y: int, type: int) -> np.ndarray:
        if type != 0:
            raise NotImplementedError()
        # better logging
        print(f'{datetime.now()}: phone {self.device_name}: click on {x},{y}')
        self.adb(f'emu event mouse {x} {y} 0 1')
        res = self.screenshot()
        self.adb(f'emu event mouse {x} {y} 0 0')
        return res


class DummyPhone:
    def __init__(self, device_name: str, port: int, cfg: Config):
        self.screen_shape = tuple(cfg['screen_shape'])
        self.configs = cfg['dummy_mode_configs']
        self.crop_top_left = cfg['crop_top_left']
        self.crop_size = cfg['crop_size']
        self.points_nums_avg = self.configs[0]
        self.points_nums_var = self.configs[1]
        self.points_size_avg = self.configs[2]
        self.points_size_var = self.configs[3]
        self.points_size_min = self.configs[4]
        self.points_border_size = self.configs[5]
        self.click_validity_margin = self.configs[6]
        self.background_color_average_max = self.configs[7]
        self.background_color_average_min = self.configs[8]
        self.background_color_var = self.configs[9]
        self.device_name = device_name
        self.app_names = ['dummy']
        self.visited_activities = set()
        self.screen = None
        self.background = None

    def restart(self, recreate_phone: bool = False) -> None:
        pass

    def start_phone(self, fresh: bool = False) -> None:
        pass

    def install_apk(self, apk_name: str) -> None:
        pass

    def close_app(self, app_name: str, reset_maintained_activities: bool = True) -> None:
        self.background = None

    def open_app(self, app_name: str) -> None:
        self.screen = None
        self.background = np.random.uniform(self.background_color_average_min, self.background_color_average_max, (3,))

    def is_in_app(self, app_name: str, force_front: bool) -> bool:
        return True

    def screenshot(self, perform_checks: bool = False) -> np.ndarray:
        if self.screen is None:
            self.screen = np.minimum(1.0, np.maximum(0.0, np.random.normal(self.background, self.background_color_var,
                                                                           (*self.screen_shape, 3))))
            points_nums = int(np.maximum(1, np.random.normal(self.points_nums_avg, self.points_nums_var)))
            self.points = list(zip(np.random.randint(self.crop_size[0], size=points_nums) + self.crop_top_left[0],
                                   np.random.randint(self.crop_size[1], size=points_nums) + self.crop_top_left[1]))
            self.points_margins = []
            for p in self.points:
                point_margin = [max(self.points_size_min,
                                    int(np.random.normal(self.points_size_avg, self.points_size_var) // 2)),
                                max(self.points_size_min,
                                    int(np.random.normal(self.points_size_avg, self.points_size_var) // 2))]
                self.points_margins += [point_margin]
                point_top_left = [max(0, p[0] - point_margin[0]), max(0, p[1] - point_margin[1])]
                point_bottom_right = [min(self.screen_shape[0], p[0] + point_margin[0]),
                                      min(self.screen_shape[1], p[1] + point_margin[1])]
                point_color = np.minimum(1, np.maximum(0, np.random.normal(1 - self.background, .2, (3,))))
                brd_out = int(np.ceil(self.points_border_size / 2))
                brd_in = self.points_border_size // 2
                self.screen[point_top_left[0]:point_bottom_right[0],
                max(0, point_top_left[1] - brd_out):point_top_left[1] + brd_in] = point_color
                self.screen[point_top_left[0]:point_bottom_right[0],
                max(0, point_bottom_right[1] - brd_in):point_bottom_right[1] + brd_out] = point_color
                self.screen[max(0, point_top_left[0] - brd_out):point_top_left[0] + brd_in,
                point_top_left[1]:point_bottom_right[1]] = point_color
                self.screen[max(0, point_bottom_right[0] - brd_in):point_bottom_right[0] + brd_out,
                point_top_left[1]:point_bottom_right[1]] = point_color
        return (self.screen * 255).astype(np.uint8)

    def send_event(self, x: int, y: int, type: int) -> np.ndarray:
        if type != 0:
            raise NotImplementedError()
        bef_screen = self.screenshot()
        br_in = self.points_border_size // 2
        mrg = self.click_validity_margin
        for p, mr in zip(self.points, self.points_margins):
            if p[0] + mr[0] - br_in + mrg >= y >= p[0] - mr[0] + br_in - mrg and \
                    p[1] + mr[1] - br_in + mrg >= x >= p[1] - mr[1] + br_in - mrg:
                self.screen = None
                self.screenshot()
                break
        return bef_screen
