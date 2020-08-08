import os
import sys
import random
import string
from datetime import datetime
from typing import Callable, Any, List, Optional

import numpy as np
import matplotlib.image as mpimg
from PIL import Image

from selenium import webdriver
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary

from utils import Config


class Browser:
    def __init__(self, device_name: str, cfg: Config):
        self.device_name = device_name

        self.screen_shape = cfg['screen_shape']
        self.window_size = cfg['window_size']
        self.mozilla_path = cfg['mozilla_path']
        self.user_agent = cfg['user_agent']
        self.query_max_length = cfg['query_max_length']
        self.screenshots_dir = cfg['screenshots_dir']
        self.scroll_min_value = cfg['scroll_min_value']
        self.scroll_max_value = cfg['scroll_max_value']
        self.apps_count = cfg['apps_count']
        self.headless = cfg['headless']

        
        self.app_names =[''.join(random.choices(string.ascii_letters + string.digits,
                                                k=random.randint(1, self.query_max_length))) 
                         for _ in range(self.apps_count)]
        self.apk_names = self.app_names.copy()
        self.maintain_visited_activities = True
        self.visited_activities = set()
        self.action_metadata_callbacks = []
        self.driver = None
        self.current_app = None

        if not os.path.exists(f'{self.screenshots_dir}/.tmp-{device_name}'):
            os.makedirs(f'{self.screenshots_dir}/.tmp-{device_name}')

    def add_action_metadata_callback(self, callback: Callable) -> None:
        self.action_metadata_callbacks.append(callback)

    def send_action_metadata(self, metadata: Any) -> None:
        for callback in self.action_metadata_callbacks:
            callback(metadata)

    def update_code_coverage(self, apk_name: str, ec_file_name: str = None) -> None:
        raise NotImplementedError()

    def is_in_app(self, app_name: str, force_front: bool) -> bool:
        return app_name == self.current_app

    def is_booted(self):
        return True

    def restart(self, recreate_phone: bool = False) -> None:
        return

    def start_phone(self, fresh: bool = False) -> None:
        if self.driver is not None:
            return
        if self.headless:
            os.environ['MOZ_HEADLESS'] = '1'
        binary = FirefoxBinary(self.mozilla_path, log_file=sys.stdout)
        profile = webdriver.FirefoxProfile()
        profile.set_preference("general.useragent.override", self.user_agent)
        self.driver = webdriver.Firefox(firefox_profile=profile, firefox_binary=binary)
        self.driver.set_window_size(self.window_size[1], self.window_size[0])

    def recreate_emulator(self) -> None:
        if self.driver is not None:
            self.driver.quit()
            self.driver = None
        return

    def install_apk(self, apk_name: str, restart: bool = True) -> None:
        return

    def close_app(self, app_name: str, reset_maintained_activities: bool = True) -> None:
        self.current_app = None
        if reset_maintained_activities:
            self.visited_activities = set()

    def get_app_all_activities(self, apk_path: str) -> List[str]:
        return ['dummy']

    def open_app(self, app_name: str) -> None:
        self.current_app = app_name
        self.driver.get(f'https://google.com/search?q={app_name}')

    def screenshot(self, perform_checks: bool = False) -> np.ndarray:
        self.visited_activities.add(self.current_app + '.' + self.driver.current_url)
        screenshot_dir = os.path.abspath(f'{self.screenshots_dir}/.tmp-{self.device_name}')
        image_path = f'{screenshot_dir}/scr.png'
        self.driver.get_screenshot_as_file(image_path)
        res = mpimg.imread(image_path)[:, :, :-1]
        self.true_screen_shape = res.shape[:2]
        res = (res * 255).astype(np.uint8)
        if self.true_screen_shape != self.screen_shape:
            res = np.array(Image.fromarray(res).resize((self.screen_shape[1], self.screen_shape[0])))
        return res

    def send_event(self, x: int, y: int, type: int) -> Optional[np.ndarray]:
        y = int(y * self.true_screen_shape[0] / self.screen_shape[0])
        x = int(x * self.true_screen_shape[1] / self.screen_shape[1])
        # better logging
        if type == 0:
            print(f'{datetime.now()}: phone {self.device_name}: click on {x},{y}')
            self.driver.execute_script(f'el = document.elementFromPoint({x}, {y}); el.click();')
            res = self.screenshot()
            self.send_action_metadata(None)
            return res
        if type == 1:
            up_scroll = random.uniform(0, 1) > .5
            val = random.randint(self.scroll_min_value, self.scroll_max_value) * (-1) ** up_scroll
            print(f'{datetime.now()}: phone {self.device_name}: scroll {"up" if up_scroll else "down"} on {x},{y}')
            self.driver.execute_script(f'scroll(0, {val});')
            self.send_action_metadata(val)
            return None
        if type == 2:
            left_scroll = random.uniform(0, 1) > .5
            val = random.randint(self.scroll_min_value, self.scroll_max_value) * (-1) ** left_scroll
            print(f'{datetime.now()}: phone {self.device_name}: swipe {"left" if left_scroll else "right"} on {x},{y}')
            self.driver.execute_script(f'scroll({val}, 0);')
            self.send_action_metadata(val)
            return None
        raise NotImplementedError()
