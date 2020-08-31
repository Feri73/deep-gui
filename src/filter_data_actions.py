import glob
from pathlib import Path
import sys
from collections import defaultdict

import numpy as np

from single_state_categorical_reward import LearningAgent, EpisodeFile
from utils import load_obj, dump_obj

input_data_files_dir = sys.argv[1]
output_data_files_dir = sys.argv[2]
max_action = int(sys.argv[3])


def is_valid(action: np.ndarray) -> bool:
    return action[-1] <= max_action


example_episode = None
for input_meta_file in glob.glob(f'{input_data_files_dir}/*/*.meta'):
    inter_dir, file_name = tuple(input_meta_file.split('/')[-2:])
    file_name = file_name[:-5]

    input_meta = load_obj(input_meta_file)
    example_episode = input_meta['example'] if example_episode is None else \
        LearningAgent.get_general_example(example_episode, input_meta['example'])
    input_episode_file = EpisodeFile(input_meta_file[:-5], input_meta['max_size'], input_meta['example'], 'r')
    valid_actions_indices = []
    new_reward_indices = defaultdict(list)
    for data_i in range(input_meta['size']):
        if is_valid(input_episode_file.actions[data_i]):
            new_reward_indices[int(input_episode_file.rewards[data_i])].append(len(valid_actions_indices))
            valid_actions_indices.append(data_i)

    if len(valid_actions_indices) > 0:
        output_file_dir = f'{output_data_files_dir}/{inter_dir}'
        Path(output_file_dir).mkdir(parents=True, exist_ok=True)
    
        output_meta_file = f'{output_file_dir}/{file_name}.meta'
        dump_obj({'max_size': len(valid_actions_indices), 'size': len(valid_actions_indices),
                  'example': input_meta['example'], 'reward_indices': new_reward_indices}, output_meta_file)
        output_episode_file = EpisodeFile(f'{output_file_dir}/{file_name}',
                                          len(valid_actions_indices), input_meta['example'], 'w+')
        for data_i in range(len(valid_actions_indices)):
            output_episode_file.set(input_episode_file.get(valid_actions_indices[data_i]), data_i)

        output_episode_file.flush()
        output_episode_file.close()

    input_episode_file.close()

