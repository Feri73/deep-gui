from datetime import datetime
import os
import shutil
import re
import subprocess
import time
from typing import Union, Optional, List

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
        self.phone_start_boot_max_wait_time = cfg['phone_start_boot_max_wait_time']
        self.phone_restart_kill_max_wait_time = cfg['phone_restart_kill_max_wait_time']
        # self.screenshot_trials = cfg['screenshot_trials']
        self.avd_path = cfg['avd_path']
        apks_path = cfg['apks_path']
        self.aapt_path = cfg['aapt_path']
        self.clone_script_path = cfg['clone_script_path']
        self.emma_jar_path = cfg['emma_jar_path']
        self.grep_command = cfg['grep_command']
        self.timeout_template = cfg['timeout_template']
        self.apk_install_command = cfg['apk_install_command']
        self.app_stop_command = cfg['app_stop_command']
        self.current_activity_grep = cfg['current_activity_grep']
        self.current_activity_regex = cfg['current_activity_regex']
        self.is_in_app_grep = cfg['is_in_app_grep']
        self.is_in_app_regex = cfg['is_in_app_regex']
        self.install_apks = cfg['install_apks']
        self.maintain_visited_activities = cfg['maintain_visited_activities']
        self.unlock = cfg['unlock']
        self.app_activity_dict = {}
        self.apk_names = glob.glob(f'{apks_path}/*.apk')
        self.app_names = [self.get_app_name(apk_path) for apk_path in self.apk_names]
        self.apk_names, self.app_names = zip(*[x for x in zip(self.apk_names, self.app_names) if x[1] is not None])
        self.app_names = list(self.app_names)
        self.apk_names = list(self.apk_names)
        if not os.path.exists(f'.tmp-{device_name}'):
            os.makedirs(f'.tmp-{device_name}')
        self.step = 0
        self.visited_activities = set()

    def adb(self, command: str, as_bytes: bool = False) -> Union[str, bytes]:
        command = f'{self.adb_path} -s emulator-{self.port} {command}'
        res = subprocess.check_output(command, shell=True)
        if not as_bytes:
            return res.decode('utf-8')
        return res

    def update_code_coverage(self, apk_name: str) -> Optional[float]:
        try:
            self.adb('shell am broadcast -a edu.gatech.m3.emma.COLLECT_COVERAGE')
            coverage_path = os.path.abspath(f'.cov_tmp-{self.device_name}.ec')
            self.adb(f'pull /mnt/sdcard/coverage.ec "{coverage_path}"')
            while not os.path.isfile(f'{coverage_path}'):
                continue
            self.adb(f'shell rm /mnt/sdcard/coverage.ec')
            command = f'java -cp "{self.emma_jar_path}" emma report -r txt --in "{apk_name}.em" -in "{coverage_path}"' \
                      f' -Dreport.txt.out.file="{coverage_path}.txt"'
            subprocess.check_output(command, shell=True)
            cov_sum = 0
            all_sum = 0
            with open(f'{coverage_path}.txt', 'r') as report_file:
                lines = report_file.readlines()
                in_table = False
                for line_a in lines:
                    line = line_a[:-1] if line_a[-1] == '\n' else line_a
                    if line == 'COVERAGE BREAKDOWN BY PACKAGE:':
                        in_table = True
                        continue
                    elif not in_table or len(line) < 3 or line[0] == '[' or line[0] == '-':
                        continue
                    ln, nm = tuple(line.split('\t')[-2:])
                    if nm.endswith('EmmaInstrument'):
                        continue
                    cov, all = tuple(ln.split('(')[1].split(')')[0].split('/'))
                    cov_sum += float(cov)
                    all_sum += float(all)
            os.remove(coverage_path)
            os.remove(f'{coverage_path}.txt')
            return cov_sum / all_sum
        except Exception as ex:
            print(f'{datetime.now()}: '
                  f'exception happened while maintaining code coverage in {self.device_name} -> {ex}')
            return None

    def add_grep(self, command: str, filter: str) -> str:
        return command + ('' if filter is None else f' | {self.grep_command} {filter}')

    def maintain_current_activity(self) -> None:
        try:
            shell_cmd = self.add_grep('dumpsys activity activities', self.current_activity_grep)
            tmp = self.adb(f'shell "{shell_cmd}"').strip()
            match = re.findall(self.current_activity_regex, tmp)
            print(f'{datetime.now()}: activity {match[0]} is visited in {self.device_name}')
            self.visited_activities.add(match[0])
        except Exception as ex:
            print(f'{datetime.now()}: '
                  f'exception happened while maintaining current activity in {self.device_name} -> {ex}')

    def is_in_app(self, app_name: str, force_front: bool) -> bool:
        if not force_front:
            raise NotImplementedError('not supporting force_front=False at this time.')
        try:
            # add timeout here
            shell_cmd = self.add_grep('dumpsys activity activities', self.is_in_app_grep)
            res = self.adb(f'shell "{shell_cmd}"')
            matches = re.findall(self.is_in_app_regex, res)
            print(f'{datetime.now()}: top app of {self.device_name}: {matches[0]}')
            return matches[0] == app_name
        except Exception:
            pass
        return False

    def is_booted(self):
        print(f'{datetime.now()}: checking boot status of {self.device_name}: ', end='')
        try:
            # this 2s should be param probably
            timeout_cmd = self.timeout_template.replace('{}', '2')
            res = self.adb(f'shell {timeout_cmd} getprop sys.boot_completed').strip() == '1'
        except subprocess.CalledProcessError:
            res = False
        print(f'{res}')
        return res

    def wait_for_start(self) -> None:
        print(f'{datetime.now()}: wait-for-device in {self.device_name}')
        self.adb('wait-for-device')
        print(f'{datetime.now()}: passed wait-for-device in {self.device_name}')
        st = time.time()
        while time.time() - st < self.phone_start_boot_max_wait_time and not self.is_booted():
            time.sleep(2)
        time.sleep(self.phone_boot_wait_time)

    def restart(self, recreate_phone: bool = False):
        print(f'{datetime.now()}: restarting {self.device_name}')
        self.adb('emu kill')
        st = time.time()
        # this is not a good way of checking if the phone is off. because the phone may be already starting,
        #    not completely booted tho. this means that i try to start the phone twice.
        while time.time() - st < self.phone_restart_kill_max_wait_time and self.is_booted():
            time.sleep(1)
        if recreate_phone:
            self.recreate_emulator()
        self.start_phone(True)

    def start_phone(self, fresh: bool = False) -> None:
        # ref_snapshot_path = f'{self.avd_path}/snapshots/fresh'
        local_snapshot_path = f'{self.avd_path}/{self.device_name}.avd/snapshots/fresh'
        self.start_emulator(fresh)
        if self.unlock:
            self.adb('shell input keyevent 82')
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
        while os.path.exists(f'{self.avd_path}/{self.device_name}.ini'):
            os.remove(f'{self.avd_path}/{self.device_name}.ini')
        while os.path.exists(f'{self.avd_path}/{self.device_name}.avd/'):
            shutil.rmtree(f'{self.avd_path}/{self.device_name}.avd/')
        os.system(f'{self.clone_script_path} {self.device_name}')

    def start_emulator(self, fresh: bool = False) -> None:
        print(f'{datetime.now()}: starting emulator {self.device_name}. fresh={fresh}')
        run_parallel_command(f'{self.emulator_path} -avd {self.device_name} -ports {self.port},{self.port + 1}' +
                             (f' -no-snapshot-load' if fresh else ''))
        self.wait_for_start()

    def install_apk(self, apk_name: str) -> None:
        self.adb(f'{self.apk_install_command} "{os.path.abspath(apk_name)}"')

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
        stop_cmd = self.app_stop_command.replace('{}', app_name)
        self.adb(f'shell {stop_cmd}')
        time.sleep(self.app_exit_wait_time)
        if reset_maintained_activities:
            self.visited_activities = set()

    def add_app_activity(self, app_name: str) -> None:
        cmd = self.add_grep(f'dumpsys package {app_name}', '-A1 ""android.intent.action.MAIN:""')
        dat = self.adb(f'shell "{cmd}"')
        lines = dat.splitlines()
        activityRE = re.compile('([A-Za-z0-9_.]+/[A-Za-z0-9_.]+)')
        self.app_activity_dict[app_name] = activityRE.search(lines[1]).group(1)

    def get_app_all_activities(self, apk_path: str) -> List[str]:
        # add caching here
        try:
            apk_path = os.path.abspath(apk_path)
            sed_pattern = r'/ activity /{:loop n;s/^.*android:name.*="\([^"]\{1,\}\)".*/\1/;T loop;p;t}'
            command = f'{self.aapt_path} list -a "{apk_path}" | sed -n \'{sed_pattern}\''
            res = subprocess.check_output(command, shell=True).decode('utf-8')
            res = res.split('\n')
            return [r for r in res if len(r) > 0]
        except Exception as ex:
            print(f'{datetime.now()}: could not get all activities for'
                  f' {apk_path} in {self.device_name} because of {ex}')
            return []

    def open_app(self, app_name: str) -> None:
        print(f'{datetime.now()}: opening {app_name} in {self.device_name}')
        if app_name not in self.app_activity_dict:
            self.add_app_activity(app_name)
        self.adb(f'shell am start -W -n {self.app_activity_dict[app_name]}')
        time.sleep(self.app_start_wait_time)

    def screenshot(self, perform_checks: bool = False) -> np.ndarray:
        if self.maintain_visited_activities and perform_checks:
            self.maintain_current_activity()
        self.step += 1
        screenshot_dir = os.path.abspath(f'.tmp-{self.device_name}')
        for file in glob.glob(f'.tmp-{self.device_name}/Screenshot*.png'):
            os.remove(file)
        self.adb(f'emu screenrecord screenshot {screenshot_dir}')
        image_path = glob.glob(f'.tmp-{self.device_name}/Screenshot*.png')[0]
        res = mpimg.imread(image_path)[:, :, :-1]
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
        self.maintain_visited_activities = cfg['maintain_visited_activities']
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
        self.apk_names = ['dummy-apk']
        self.visited_activities = {'dummy.activity'}
        self.screen = None
        self.background = None

    def restart(self, recreate_phone: bool = False) -> None:
        pass

    def start_phone(self, fresh: bool = False) -> None:
        pass

    def recreate_emulator(self) -> None:
        pass

    def install_apk(self, apk_name: str) -> None:
        pass

    def close_app(self, app_name: str, reset_maintained_activities: bool = True) -> None:
        self.background = None

    def get_app_all_activities(self, apk_path: str) -> List[str]:
        return ['dummy.activity']

    def open_app(self, app_name: str) -> None:
        self.screen = None
        self.background = np.random.uniform(self.background_color_average_min, self.background_color_average_max, (3,))

    def is_in_app(self, app_name: str, force_front: bool) -> bool:
        return True

    def update_code_coverage(self, apk_name: str) -> float:
        return 0.0

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
