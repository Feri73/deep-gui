import requests
import yaml
import os
from html.parser import HTMLParser

file_path = input('Enter html file path:')
with open('setting.yaml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
apks_path = cfg['apks_path']

downloaded_files = [file[:-4] for file in os.listdir(apks_path)]


class AppNameExtactor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.apps = []

    def handle_starttag(self, tag, attrs):
        if tag != 'a':
            return
        valid = False
        for attr in attrs:
            if attr[0] == 'class' and attr[1] == 'poRVub':
                valid = True
            if attr[0] == 'href':
                href = attr[1]
        if valid:
            self.apps += [href.split('id=')[1]]


class AppLinkExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.link = None

    def handle_starttag(self, tag, attrs):
        if tag != 'a':
            return
        valid = False
        for attr in attrs:
            if attr[0] == 'id' and attr[1] == 'download_link':
                valid = True
            if attr[0] == 'href':
                href = attr[1]
        if valid:
            self.link = href


parser = AppNameExtactor()
with open(file_path, encoding='utf-8') as f:
    html = f.read()
parser.feed(html)

for app in parser.apps:
    try:
        print(f'starting downloading {app}')
        if app in downloaded_files:
            print('skipped')
            continue
        parser = AppLinkExtractor()
        parser.feed(requests.get(f'https://apkpure.com/termux-api/{app}/download').content.decode('utf-8'))
        apk_content = requests.get(parser.link)
        with open(f'{apks_path}/{app}.apk', 'wb') as output:
            output.write(apk_content.content)
    except Exception as e:
        print(f'error occurred: {e}')
