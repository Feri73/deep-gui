import tensorflow as tf
from collections import defaultdict
import sys
import os
import glob
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
from io import BytesIO
import time

_, logs_dir, experiment_name, output_dir, apps_dir = tuple(sys.argv)
seen_files=defaultdict(list)
apps=[os.path.basename(apk) for apk in glob.glob(f'{apps_dir}/*.apk')]
writers=dict()
writers_open=dict()
# indices: 0: current chunk 1: current global step 2: current offset
steps=defaultdict(lambda: [0,0,0])
tf.disable_v2_behavior()

def get_image_summary(image):
    bio = BytesIO()
    Image.fromarray(image).save(bio, format="png")
    res = tf.Summary.Image(encoded_image_string=bio.getvalue(), height=image.shape[0], width=image.shape[1])
    bio.close()
    return res

def write(tool,tester,chunk,seen_files_key,coverages=None,img=None,step=None):
    summary=tf.Summary()
    if step is None:
        step=len(seen_files[seen_files_key])
    step_key=f'{tool}-{tester}-{"cvg" if img is None else "img"}'
    if step < steps[step_key][0]:
        steps[step_key][2]=steps[step_key][1]
    steps[step_key][0]=step
    steps[step_key][1]=steps[step_key][2]+step
    if img is not None:
        summary.value.add(tag='Screenshots', image=img)
    if coverages is not None:
        summary.value.add(tag='Coverage/Class', simple_value=coverages[0])
        summary.value.add(tag='Coverage/Method', simple_value=coverages[1])
        summary.value.add(tag='Coverage/Block', simple_value=coverages[2])
        summary.value.add(tag='Coverage/Line', simple_value=coverages[3])
    writer_key=f'{tool}-{tester}-{chunk}'
    if writer_key not in writers:
        writers[writer_key]=tf.summary.FileWriter(f'{output_dir}/{tool}_{tester}_chunk_{chunk}')
        writers_open[writer_key] = True
    elif not writers_open[writer_key]:
        writers[writer_key].reopen()
        writers_open[writer_key] = True
        time.sleep(.5)
    writers[writer_key].add_summary(summary, steps[step_key][1])
    #writers[writer_key].close()

def parse_report(log):
    cov_sum = np.zeros(4)
    all_sum = np.zeros(4)
    with open(log, 'r') as report_file:
        lines = report_file.readlines()
        if len(lines)==0:
            return None
        if lines[0]=='nan\n':
            return (np.nan, np.nan, np.nan, np.nan)
        in_table = False
        for line_a in lines:
            line = line_a[:-1] if line_a[-1] == '\n' else line_a
            if line == 'COVERAGE BREAKDOWN BY PACKAGE:':
                in_table = True
                continue
            elif not in_table or len(line) < 3 or line[0] == '[' or line[0] == '-':
                continue
            *vals, nm = tuple(line.split('\t'))
            if nm.endswith('EmmaInstrument'):
                continue
            cov, all = tuple(zip(*[val.split('(')[1].split(')')[0].split('/') for val in vals]))
            cov_sum += tuple(map(float, cov))
            all_sum += tuple(map(float, all))
    return tuple(cov_sum / all_sum)

while True:
    for tool_experiment in os.listdir(logs_dir):
        if not tool_experiment.startswith(f'coverage_{experiment_name}-'):
            continue
        tool=tool_experiment.split(f'{experiment_name}-')[1]
        tool_exp_dir=f'{logs_dir}/{tool_experiment}'
        for tester in os.listdir(tool_exp_dir):
            tester_dir=f'{tool_exp_dir}/{tester}'
            for apk in os.listdir(tester_dir):
                apk_dir=f'{tester_dir}/{apk}'
                for rnd in os.listdir(apk_dir):
                    rnd_dir=f'{apk_dir}/{rnd}'
                    chunk=apps.index(apk) + len(apps) * int(rnd)
                    
                    scr_dir=f'{rnd_dir}/screenshots'
                    if os.path.isdir(scr_dir):
                        for screenshot in sorted(os.listdir(scr_dir), key=lambda f: int(f.split('.')[0])):# os.listdir(scr_dir):
                            if screenshot not in seen_files[scr_dir]:
                                try:
                                    img = mpimg.imread(f'{rnd_dir}/screenshots/{screenshot}')[:, :, :-1]
                                    img = (img * 255).astype(np.uint8)
                                    img = get_image_summary(img)
                                except Exception:
                                    continue
                                print(f'writing {rnd_dir}/screenshots/{screenshot}')
                                write(tool,tester,chunk,scr_dir,img=img,step=int(screenshot.split('.png')[0]))
                                seen_files[scr_dir].append(screenshot)
                    
                    for log in sorted([l for l in os.listdir(rnd_dir) if l.endswith('.txt')], key=lambda f: int(f.split('.')[0])):
                        if log.endswith('.txt') and log not in seen_files[rnd_dir]:
                            coverages=parse_report(f'{rnd_dir}/{log}')
                            if coverages is None:
                                continue
                            print(f'writing {rnd_dir}/{log}')
                            write(tool,tester,chunk,rnd_dir,coverages=coverages, step=int(log.split('.ec')[0]))
                            seen_files[rnd_dir].append(log)
            for k, w in writers.items():
                w.close()
                writers_open[k] = False
            time.sleep(5)
#            for w in writers.values():
#                w.reopen()
