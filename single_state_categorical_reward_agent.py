import glob
import multiprocessing
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from functools import partial
from io import BytesIO
from pathlib import Path
from queue import Empty
from typing import Any, List, Dict, Tuple, Callable, Optional, Union

import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import yaml
from PIL import Image
from sklearn.cluster import AgglomerativeClustering

from environment import EnvironmentCallbacks, EnvironmentController, Environment
from phone import DummyPhone, Phone
from relevant_action import RelevantActionEnvironment
from relevant_action_monkey_client import RelevantActionMonkeyClient
from utils import Config, MemVariable, dump_obj, load_obj


class Episode:
    def __init__(self, state: np.ndarray = None, action: np.ndarray = None,
                 reward: np.ndarray = None, result: np.ndarray = None):
        assert (state is None and result is None) or (state.shape == result.shape and state.dtype == result.dtype)
        self.state = state
        self.action = action
        self.reward = reward
        self.result = result


class EpisodeFile:
    def __init__(self, file_name: str, max_size: int, example: Episode, mode: str):
        self.file_name = file_name

        self.states = np.memmap(file_name + '.states.npy',
                                dtype=example.state.dtype, mode=mode, shape=(max_size, 2, *example.state.shape))
        self.actions = np.memmap(file_name + '.actions.npy',
                                 dtype=example.action.dtype, mode=mode, shape=(max_size, *example.action.shape))
        self.rewards = np.memmap(file_name + '.rewards.npy',
                                 dtype=example.reward.dtype, mode=mode, shape=(max_size, *example.reward.shape))

    def get(self, index: int) -> Episode:
        states = self.states[index]
        return Episode(states[0], self.actions[index], self.rewards[index], states[1])

    def set(self, episode: Episode, index: int) -> None:
        self.states[index][0] = episode.state
        self.states[index][1] = episode.result
        self.actions[index] = episode.action
        self.rewards[index] = episode.reward

    def flush(self):
        self.states.flush()
        self.actions.flush()
        self.rewards.flush()

    def close(self):
        del self.states
        del self.actions
        del self.rewards


class DataCollectionAgent(EnvironmentCallbacks, EnvironmentController):
    def __init__(self, id: int, model: keras.Model, example_episode: Episode,
                 create_environment: Callable[['DataCollectionAgent'], Environment], cfg: Config):
        self.max_episodes = cfg['max_episodes']
        self.max_file_size = cfg['max_file_size']
        self.meta_save_frequency = cfg['meta_save_frequency']
        self.file_dir = cfg['file_dir']
        version_start = cfg['version_start']

        self.id = id
        self.model = model
        self.example_episode = example_episode

        self.current_file_version = version_start - 1
        self.current_file = None
        self.current_file_size = 0
        self.reward_indices = defaultdict(list)
        self.finished_episodes_count = 0
        self.current_episode = MemVariable(lambda: None)
        self.on_file_completed_callbacks = []

        self.reset_file()
        self.environment = create_environment(self)
        self.environment.add_callback(self)

    # the fact that this gets weights means that i cannot set the weights directly from tf (use tf ops for it)
    def update_weights(self, weights: List[tf.Tensor]):
        self.model.set_weights(weights)

    def add_on_file_completed_callbacks(self, callback: Callable[[int, int], None]) -> None:
        self.on_file_completed_callbacks.append(callback)

    def on_file_completed_callback(self) -> None:
        for callback in self.on_file_completed_callbacks:
            callback(self.id, self.current_file_version)

    def start(self):
        self.environment.start()

    def dump_meta(self):
        dump_obj({'max_size': self.max_file_size, 'size': self.current_file_size,
                  'example': self.example_episode, 'reward_indices': self.reward_indices},
                 self.current_file.file_name + '.meta')

    def reset_file(self, new_file: bool = True):
        if self.current_file is not None:
            self.current_file.flush()
            self.current_file.close()
            self.dump_meta()
            self.current_file = None
            self.on_file_completed_callback()
            # notify controller here

        if new_file and self.max_file_size > 0:
            self.current_file_version += 1
            self.current_file_size = 0
            self.reward_indices = defaultdict(list)
            Path(f'{self.file_dir}/{self.current_file_version}').mkdir(parents=True, exist_ok=True)
            self.current_file = EpisodeFile(f'{self.file_dir}/{self.current_file_version}/{self.id}',
                                            self.max_file_size, self.example_episode, 'w+')

    def store_episode(self, episode: Episode) -> None:
        if self.current_file is not None:
            if self.current_file_size == self.max_file_size:
                self.reset_file()
            elif self.current_file_size % self.meta_save_frequency == 0:
                self.dump_meta()
            self.reward_indices[int(episode.reward)].append(self.current_file_size)
            self.current_file.set(episode, self.current_file_size)
            self.current_file_size += 1

    def should_start_episode(self) -> bool:
        res = self.finished_episodes_count < self.max_episodes
        # bad practice
        if not res:
            self.store_episode(self.current_episode.last_value())
            self.reset_file(False)
        return res

    def get_next_action(self, state: np.ndarray) -> Any:
        state = np.expand_dims(state, axis=0)
        return self.model.predict_on_batch(state)[0]

    def on_episode_start(self, state: np.ndarray) -> None:
        self.current_episode.value = Episode()
        self.current_episode.value.state = state

    def on_wait(self) -> None:
        if self.current_episode.has_archive():
            self.store_episode(self.current_episode.last_value())

    def on_state_change(self, src_state: np.ndarray, action: Any, dst_state: np.ndarray, reward: float) -> None:
        self.current_episode.value.reward = np.array(int(reward), self.example_episode.reward.dtype)
        self.current_episode.value.result = dst_state
        self.current_episode.value.action = action

    def on_episode_end(self, premature: bool) -> None:
        self.current_episode.archive()
        self.finished_episodes_count += 1

    def on_error(self) -> None:
        self.current_episode.reset_value()


class LearningAgent:
    def __init__(self, id: int, model: keras.Model, distorter: Callable, cfg: Config):
        self.file_dir = cfg['file_dir']
        self.shuffle = cfg['shuffle']
        self.correct_distributions = cfg['correct_distributions']
        self.augmenting_correction = cfg['augmenting_correction']
        self.batch_size = cfg['batch_size']
        self.epochs_per_version = cfg['epochs_per_version']
        self.logs_dir = cfg['logs_dir']
        self.save_dir = cfg['save_dir']
        self.save_frequency = cfg['save_frequency']

        self.id = id
        # plot the model (maybe here or where it's created)
        self.model = model
        self.distorter = distorter

        # self.log_callback = keras.callbacks.TensorBoard(f'{self.logs_dir}{os.sep}learner_{self.id}')

    class EpisodeFileManager:
        def __init__(self, episode_files: List[EpisodeFile]):
            self.episode_files = episode_files

        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            for episode_file in self.episode_files:
                episode_file.close()

    def get_weights(self) -> List[tf.Tensor]:
        return self.model.get_weights()

    @staticmethod
    def get_general_example(ex1: Episode, ex2: Episode) -> Episode:
        assert ex1.state.shape == ex2.state.shape and ex1.action.shape == ex2.action.shape \
               and ex1.reward.shape == ex2.reward.shape
        return Episode(ex1.state if ex1.state.itemsize > ex2.state.itemsize else ex2.state,
                       ex1.action if ex1.action.itemsize > ex2.action.itemsize else ex2.action,
                       ex1.reward if ex1.reward.itemsize > ex2.reward.itemsize else ex2.reward,
                       ex1.result if ex1.result.itemsize > ex2.result.itemsize else ex2.result)

    @staticmethod
    def merge_reward_indices_list(file_reward_indices_list: List[Dict[np.ndarray, List[int]]]) \
            -> Dict[np.ndarray, List[int]]:
        total_reward_indices = defaultdict(list)
        for file_i, reward_indices in enumerate(file_reward_indices_list):
            for reward in reward_indices:
                total_reward_indices[reward] += [(file_i, index) for index in reward_indices[reward]]
        return total_reward_indices

    @staticmethod
    def correct_distribution(total_reward_indices: Dict[np.ndarray, List[int]]) -> Tuple[np.ndarray, int]:
        # i can do this distribution correction by clicking on slightly different positions,
        #   or by re-using from previous versions
        # is this way of doing this good? because i have the same samples a lot!
        # i can also not have this and instead use weights in keras

        if len(total_reward_indices) == 1:
            return np.array(-1), -1
        if len(total_reward_indices) != 2:
            raise NotImplementedError('Oops. Currently not supporting non-binary rewards.')
        less_represented_reward = list(total_reward_indices.keys())[np.argmin([len(total_reward_indices[reward])
                                                                               for reward in total_reward_indices])]
        augmented_size = abs(np.subtract(*[len(total_reward_indices[reward]) for reward in total_reward_indices]))
        return less_represented_reward, augmented_size

    def read_episode_files(self, version: Union[int, List[int]]) -> \
            Tuple[List[EpisodeFile], List[int], List[Dict[np.ndarray, List[int]]], Episode]:
        if not isinstance(version, list):
            version = [version]

        example_episode = None
        episode_files = []
        file_sizes = []
        file_reward_indices_list = []
        meta_files = []
        for v in version:
            meta_files += glob.glob(f'{self.file_dir}/{v}/*.meta')
        for meta_file in meta_files:
            meta = load_obj(meta_file)
            example_episode = meta['example'] if example_episode is None else \
                self.get_general_example(example_episode, meta['example'])
            episode_files.append(EpisodeFile(meta_file[:-5], meta['max_size'], meta['example'], 'r'))
            file_sizes.append(meta['size'])
            file_reward_indices_list.append(meta['reward_indices'])

        return episode_files, file_sizes, file_reward_indices_list, example_episode

    def create_training_data(self, version: Union[int, List[int]]) -> Tuple[Optional[Callable], int]:
        episode_files, file_sizes, file_reward_indices_list, example_episode = self.read_episode_files(version)

        if len(episode_files) == 0:
            return None, 0

        total_reward_indices = self.merge_reward_indices_list(file_reward_indices_list)
        if self.correct_distributions:
            less_represented_reward, augmented_size = self.correct_distribution(total_reward_indices)
            more_represented_reward = [x for x in list(total_reward_indices.keys()) if x != less_represented_reward][0]
        else:
            augmented_size = 0

        if augmented_size == -1:
            return None, 0

        total_size = sum(file_sizes)
        if self.augmenting_correction:
            training_size = total_size + augmented_size
            poses1 = np.random.choice(np.array(total_reward_indices[more_represented_reward], dtype='int, int'),
                                      (total_size - augmented_size) // 2, replace=False)
            poses1_augment = np.random.choice(np.array(total_reward_indices[more_represented_reward], dtype='int, int'),
                                              augmented_size, replace=False)
            poses2 = np.random.choice(np.array(total_reward_indices[less_represented_reward], dtype='int, int'),
                                      (total_size - augmented_size) // 2, replace=False)
            positions = np.concatenate([poses1, poses1_augment, poses2])
        else:
            training_size = total_size - augmented_size
            poses1 = np.random.choice(np.array(total_reward_indices[more_represented_reward], dtype='int, int'),
                                      training_size // 2, replace=False)
            poses2 = np.random.choice(np.array(total_reward_indices[less_represented_reward], dtype='int, int'),
                                      training_size // 2, replace=False)
            positions = np.concatenate([poses1, poses2])
        positions_order = np.random.permutation(training_size) if self.shuffle else np.arange(training_size)
        positions = positions[positions_order]

        def generator() -> Tuple[Dict[str, np.ndarray], np.ndarray]:
            # if epochs is a lot more than 1, then i should generate a dataset in file instead of this ad hoc method
            current_positions_i = 0
            with self.EpisodeFileManager(episode_files):
                while True:
                    batch_size = min(self.batch_size, training_size - current_positions_i)
                    if batch_size < self.batch_size:
                        current_positions_i = 0
                        batch_size = self.batch_size

                    x = {'state': np.zeros((batch_size, *example_episode.state.shape),
                                           dtype=example_episode.state.dtype),
                         'action': np.zeros((batch_size, *example_episode.action.shape),
                                            dtype=example_episode.action.dtype),
                         'result': np.zeros((batch_size, *example_episode.result.shape),
                                            dtype=example_episode.result.dtype)}
                    y = np.zeros((batch_size, 1), dtype=np.int32)
                    for i in range(batch_size):
                        position = positions[current_positions_i + i]
                        file_i = position[0]
                        data_i = position[1]
                        episode = episode_files[file_i].get(data_i)
                        episode = self.distorter(episode)
                        x['state'][i] = episode.state
                        x['action'][i] = episode.action
                        x['result'][i] = episode.result
                        y[i][0] = episode.reward

                    yield x, y

                    current_positions_i = min(training_size, current_positions_i + self.batch_size) % training_size

        return generator, training_size

    # add logs
    def learn(self, version: Union[int, List[int]]) -> None:
        generator, data_size = self.create_training_data(version)
        if generator is None:
            print(f'{datetime.now()}: The experience version {version} is not expressive enough to learn from.')
        else:
            print(f'{datetime.now()}: starting learning for experience version {version}')
            data = generator()
            steps_per_epoch = int(data_size * self.epochs_per_version / self.batch_size / int(self.epochs_per_version))
            checkpoint_callback = keras.callbacks.ModelCheckpoint(
                f'{self.save_dir}/{version[-1] if isinstance(version, list) else version}-' +
                '{epoch:02d}-{loss:.2f}.hdf5', monitor='loss', save_best_only=False, save_weights_only=True,
                save_freq=int(self.save_frequency * steps_per_epoch * self.batch_size))
            self.model.fit(data, epochs=int(self.epochs_per_version), steps_per_epoch=steps_per_epoch,
                           callbacks=[checkpoint_callback])
            del data


class ThreadLocals:
    def __init__(self):
        self.thread = None
        self.collector = None
        self.new_weight = None

    def pop_and_run_next(self, *local_args, wait=False) -> None:
        self.thread.pop_and_run_next(*local_args, wait=wait)


class Thread(ABC):
    @abstractmethod
    def add_to_run_queue(self, func: Callable, *args) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        pass

    # this can only be accessed from the thread associated to this object
    @abstractmethod
    def pop_and_run_next(self, *local_args, wait=False) -> None:
        pass


class Coordinator(ABC, EnvironmentCallbacks):
    def __init__(self, collector_creators: List[Callable[[], DataCollectionAgent]],
                 learner_creator: Callable[[], LearningAgent],
                 tester_creators: List[Callable[[], DataCollectionAgent]], cfg: Config):
        self.collector_version_start = cfg['collector_version_start']
        self.train = cfg['train']
        self.pre_training = cfg['pre_training']
        self.collect_before_pre_training = cfg['collect_before_pre_training']
        self.sync_weight = cfg['sync_weight']

        self.collector_creators = collector_creators
        self.learner_creator = learner_creator
        self.tester_creators = tester_creators

        self.learner = None
        self.file_completions = defaultdict(list)
        self.environment_completion_count = 0

        self.learner_thread = None
        self.collector_threads = []
        self.tester_threads = []

    @abstractmethod
    def get_thread_locals(self) -> ThreadLocals:
        pass

    @abstractmethod
    def create_thread(self, main_func: Callable, *args) -> Thread:
        pass

    @abstractmethod
    def get_main_thread(self) -> Thread:
        pass

    def start_collector(self, collector_creator: Callable[[], DataCollectionAgent], thread: Thread) -> None:
        collector = collector_creator()
        collector.add_on_file_completed_callbacks(self.on_collector_file_completed)
        collector.environment.add_callback(self)
        self.get_thread_locals().collector = collector
        self.get_thread_locals().thread = thread
        self.get_thread_locals().pop_and_run_next(self, wait=not self.collect_before_pre_training)
        collector.start()

    def on_episode_end(self, premature: bool) -> None:
        self.get_thread_locals().pop_and_run_next(self)
        self.local_update_collector_weight()

    def on_environment_finished(self) -> None:
        self.learner_thread.add_to_run_queue(Coordinator.record_environment_completion)

    def record_environment_completion(self):
        self.environment_completion_count += 1

    def local_set_new_weight(self, new_weight: List[tf.Tensor]) -> None:
        self.get_thread_locals().new_weight = new_weight
        print(f'{datetime.now()}: collector {self.get_thread_locals().collector.id} synced weights.')

    # make these functions with function decorator for coolness :D
    def local_update_collector_weight(self):
        locals = self.get_thread_locals()
        if locals.new_weight is not None:
            locals.collector.update_weights(locals.new_weight)
            locals.new_weight = None

    def send_to_workers(self, func: Callable, *args) -> None:
        for collector_thread in self.collector_threads:
            collector_thread.add_to_run_queue(func, *args)
        for tester_thread in self.tester_threads:
            tester_thread.add_to_run_queue(func, *args)

    def sync_weights(self):
        if not self.sync_weight:
            return
        print(f'{datetime.now()}: sending weights to workers.')
        self.send_to_workers(Coordinator.local_set_new_weight, self.learner.get_weights())

    def dummy(self):
        return

    def record_collector_file_completion(self, version: int) -> None:
        self.file_completions[version].append(True)
        if self.train and len(self.file_completions[version]) == len(self.collector_creators):
            self.learner.learn(version)
            self.sync_weights()

    def on_collector_file_completed(self, id: int, version: int) -> None:
        self.learner_thread.add_to_run_queue(partial(Coordinator.record_collector_file_completion, version=version))

    def start(self):
        self.learner_thread = self.get_main_thread()
        self.collector_threads = [self.create_thread(self.start_collector, c_creator)
                                  for c_creator in self.collector_creators]
        self.tester_threads = [self.create_thread(self.start_collector, t_creator)
                               for t_creator in self.tester_creators]
        [c_thread.run() for c_thread in self.collector_threads]
        [t_thread.run() for t_thread in self.tester_threads]
        self.learner = self.learner_creator()
        if self.pre_training:
            self.learner.learn(list(range(self.collector_version_start)))
        if self.sync_weight:
            self.sync_weights()
        else:
            print(f'{datetime.now()}: sending dummy to workers.')
            self.send_to_workers(Coordinator.dummy)
        while self.environment_completion_count < len(self.collector_creators) + len(self.tester_creators):
            self.learner_thread.pop_and_run_next(self)
            time.sleep(1)


# one problem with current framework is that each Thread has only 1 queue
class Process(Thread):
    def __init__(self, name: Optional[str], main_func: Optional[Callable], *args,
                 cfg: Config, main_process: bool = False):
        type = cfg['type']
        queue_size = cfg['queue_size']

        mp = multiprocessing.get_context(type)

        if main_process:
            assert name is None and main_func is None and len(args) == 0
        else:
            self.process = mp.Process(name=name, target=main_func, args=(*args, self))
        self.queue = mp.Queue(queue_size)

    def add_to_run_queue(self, func: Callable, *args) -> None:
        self.queue.put((func, args))

    def run(self) -> None:
        self.process.start()

    def pop_and_run_next(self, *local_args, wait=False) -> None:
        try:
            if wait:
                func, args = self.queue.get()
            else:
                func, args = self.queue.get_nowait()
            func(*local_args, *args)
        except Empty:
            pass


class ProcessBasedCoordinator(Coordinator):
    def __init__(self, collector_creators: List[Callable[[], DataCollectionAgent]],
                 learner_creator: Callable[[], LearningAgent],
                 tester_creators: List[Callable[[], DataCollectionAgent]], cfg: Config):
        self.process_configs = cfg['process_configs']

        super().__init__(collector_creators, learner_creator, tester_creators, cfg)

        self.thread_count = 0
        self.thread_locals = None

    def get_thread_locals(self) -> ThreadLocals:
        if self.thread_locals is None:
            self.thread_locals = ThreadLocals()
        return self.thread_locals

    def create_thread(self, main_func: Callable, *args) -> Thread:
        return Process(f'process_{self.thread_count}', main_func, *args, cfg=self.process_configs)

    def get_main_thread(self) -> Thread:
        return Process(None, None, cfg=self.process_configs, main_process=True)


class ScreenPreprocessor(keras.layers.Layer):
    def __init__(self, cfg: Config, **kwargs):
        self.grayscale = cfg['grayscale']
        self.crop_top_left = cfg['crop_top_left']
        self.crop_size = cfg['crop_size']
        self.resize_size = cfg['resize_size']
        self.scale_color = cfg['scale_color']
        self.equalize_background = cfg['equalize_background']
        self.contrast_alpha = cfg['contrast_alpha']

        super().__init__(**kwargs)

    def call(self, screens, **kwargs):
        screens = tf.cast(screens, tf.float32) / 255.0
        if self.grayscale:
            screens = tf.image.rgb_to_grayscale(screens)
        screens = tf.image.crop_to_bounding_box(screens, *self.crop_top_left, *self.crop_size)
        screens_shape = tuple([int(d) for d in screens.shape])
        if screens_shape[1:3] != self.resize_size:
            screens = tf.image.resize(screens, self.resize_size)
        screens_shape = tuple([int(d) for d in screens.shape])
        if self.scale_color:
            if screens_shape[-1] != 1:
                raise AttributeError('cannot scale colored images.')
            axes = [1, 2, 3]
            screens = (screens - tf.reduce_min(screens, axis=axes, keep_dims=True)) / \
                      (tf.reduce_max(screens, axis=axes, keep_dims=True) -
                       tf.reduce_min(screens, axis=axes, keep_dims=True) + 1e-6)
        if self.equalize_background:
            if screens_shape[-1] != 1:
                raise AttributeError('cannot equalize background for colored images.')
            image_size = screens_shape[1] * screens_shape[2]
            color_sums = tf.reduce_sum(tf.cast(screens < .5, tf.float32), axis=[1, 2, 3])
            screens, _ = tf.map_fn(lambda elems:
                                   (tf.where(elems[1] < image_size / 2, 1 - elems[0], elems[0]), elems[1]),
                                   (screens, color_sums))
        if self.contrast_alpha > 0:
            if not screens_shape[-1] != 1:
                raise AttributeError('cannot change contrast of colored images.')
            screens = tf.sigmoid(self.contrast_alpha * (screens - .5))
        return screens


class EncodingRewardPredictor(keras.layers.Layer):
    def __init__(self, screen_encoder: keras.layers.Layer, reward_decoder: keras.layers.Layer, **kwarg):
        self.screen_encoder = screen_encoder
        self.reward_decoder = reward_decoder

        super().__init__(**kwarg)

    def call(self, screens, **kwargs):
        return self.reward_decoder(self.screen_encoder(screens))


class SimpleScreenEncoder(keras.layers.Layer):
    def __init__(self, cfg: Config, **kwargs):
        self.padding_type = cfg['padding_type']
        self.filter_nums = cfg['filter_nums']
        self.kernel_sizes = cfg['kernel_sizes']
        self.stride_sizes = cfg['stride_sizes']
        self.maxpool_sizes = cfg['maxpool_sizes']

        super().__init__(**kwargs)

        self.convs = []
        self.maxpools = []

    def build(self, input_shape):
        self.convs = [keras.layers.Conv2D(filters, kernel_size, stride, self.padding_type, activation=tf.nn.elu)
                      for filters, kernel_size, stride in zip(self.filter_nums, self.kernel_sizes, self.stride_sizes)]
        self.maxpools = [keras.layers.Lambda(lambda x: x)
                         if pool_size == 1 else keras.layers.MaxPool2D(pool_size, pool_size, self.padding_type)
                         for pool_size in self.maxpool_sizes]

    def call(self, screens, **kwargs):
        for conv, maxpool in zip(self.convs, self.maxpools):
            screens = maxpool(conv(screens))
        return screens


class SimpleRewardDecoder(keras.layers.Layer):
    def __init__(self, action_type_count: int, reward_categories_count: int, cfg: Config, **kwargs):
        super().__init__(**kwargs)

        self.prediction_shape = tuple(cfg['prediction_shape'])
        self.reward_categories_count = reward_categories_count

        self.action_type_count = action_type_count

        self.last_layer = None

        if self.reward_categories_count != 2:
            raise ValueError('For now only support binary rewards.')

    def build(self, input_shape):
        self.last_layer = keras.layers.Conv2D(self.action_type_count, 1, 1, 'VALID', activation=tf.nn.sigmoid)

    def call(self, parsed_screens, **kwargs):
        parsed_screens = self.last_layer(parsed_screens)
        assert tuple(map(int, (parsed_screens.shape[1:-1]))) == self.prediction_shape
        return parsed_screens


class SimpleRewardPredictor(EncodingRewardPredictor):
    def __init__(self, action_type_count: int, reward_categories_count: int, cfg: Config, **kwargs):
        prediction_shape = cfg['prediction_shape']
        screen_encoder_cfg = cfg['screen_encoder_configs']

        reward_decoder_cfg = {'prediction_shape': prediction_shape}

        super().__init__(SimpleScreenEncoder(screen_encoder_cfg),
                         SimpleRewardDecoder(action_type_count, reward_categories_count, reward_decoder_cfg), **kwargs)


class UNetScreenEncoder(keras.layers.Layer):
    def __init__(self, cfg: Config, **kwargs):
        self.output_layer_names = cfg['output_layer_names']
        self.inner_configs = cfg['inner_configs']
        super().__init__(**kwargs)
        self.encoder = None

    def build(self, input_shape):
        net = keras.applications.MobileNetV2(input_shape=tuple(int(x) for x in input_shape[1:]),
                                             include_top=False, **self.inner_configs)
        self.encoder = keras.Model(inputs=net.input,
                                   outputs=[net.get_layer(name).output for name in self.output_layer_names])

    def call(self, screens, **kwargs):
        return self.encoder(screens)


class UNetRewardDecoder(keras.layers.Layer):
    def __init__(self, action_type_count: int, reward_categories_count: int, cfg: Config, **kwargs):
        self.filter_nums = cfg['filter_nums']
        self.kernel_sizes = cfg['kernel_sizes']
        self.stride_sizes = cfg['stride_sizes']
        self.padding_types = cfg['padding_types']
        self.prediction_shape = tuple(cfg['prediction_shape'])

        super().__init__(**kwargs)

        self.action_type_count = action_type_count
        self.reward_categories_count = reward_categories_count

        self.decoders = None
        self.last_layer = None

        if self.reward_categories_count != 2:
            raise NotImplementedError('For now only support binary rewards.')

    @staticmethod
    def deconv(filters: int, size: int, stride: int, padding: str, activation: str, normalization: bool):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2DTranspose(filters, size, stride, padding, activation=activation,
                                                   kernel_initializer=initializer, use_bias=False))
        if normalization:
            result.add(keras.layers.BatchNormalization())
        return result

    @staticmethod
    def val2list(val, size: int) -> list:
        return val if isinstance(val, list) else [val] * size

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.filter_nums = self.val2list(self.filter_nums, len(input_shape) - 1)
        self.kernel_sizes = self.val2list(self.kernel_sizes, len(input_shape))
        self.stride_sizes = self.val2list(self.stride_sizes, len(input_shape))
        self.padding_types = self.val2list(self.padding_types, len(input_shape))
        self.decoders = [self.deconv(filter, size, stride, padding, 'relu', True) for filter, size, stride, padding in
                         zip(self.filter_nums, self.kernel_sizes[:-1], self.stride_sizes[:-1], self.padding_types[:-1])]
        self.last_layer = self.deconv(self.action_type_count, self.kernel_sizes[-1], self.stride_sizes[-1],
                                      self.padding_types[-1], 'sigmoid', False)

    def call(self, parsed_screens_layers, **kwargs):
        hidden = parsed_screens_layers[0]
        skips = parsed_screens_layers[1:]
        for decoder, skip in zip(self.decoders, skips):
            hidden = decoder(hidden)
            concat = tf.keras.layers.Concatenate()
            hidden = concat([hidden, skip])
        result = self.last_layer(hidden)
        assert tuple(map(int, (result.shape[1:-1]))) == self.prediction_shape
        return result


class UNetRewardPredictor(EncodingRewardPredictor):
    def __init__(self, action_type_count: int, reward_categories_count: int, cfg: Config, **kwargs):
        screen_encoder_cfg = cfg['screen_encoder_configs']
        reward_decoder_cfg = cfg['reward_decoder_configs']
        reward_decoder_cfg['prediction_shape'] = cfg['prediction_shape']

        super().__init__(UNetScreenEncoder(screen_encoder_cfg),
                         UNetRewardDecoder(action_type_count, reward_categories_count, reward_decoder_cfg), **kwargs)


class RandomRewardPredictor(keras.layers.Layer):
    def __init__(self, action_type_count: int, reward_categories_count: int, cfg: Config, **kwargs):
        self.prediction_shape = cfg['prediction_shape']

        super().__init__(**kwargs)

        self.action_type_count = action_type_count
        self.reward_categories_count = reward_categories_count

        if self.reward_categories_count != 2:
            raise NotImplementedError('For now only support binary rewards.')

    def call(self, screens, **kwargs):
        return tf.random.uniform((screens.shape[0], *self.prediction_shape, self.action_type_count))


class BufferLogger(keras.layers.Layer):
    def __init__(self, freq: int, handler: Callable, aggregate: bool, **kwargs):
        super().__init__(**kwargs)

        self.freq = freq
        self.handler = handler
        self.aggregate = aggregate

        self.log_values = []
        self.log_step = None
        self.dependency = None

    def build(self, input_shape):
        self.log_step = self.add_weight(shape=(), trainable=False,
                                        initializer=lambda *args, **kwargs: 0, dtype=tf.int32)
        self.dependency = self.add_weight(shape=(), trainable=False,
                                          initializer=lambda *args, **kwargs: 0, dtype=tf.int32)

    def call(self, inputs, **kwargs):
        self.log_step = tf.assign_add(self.log_step, 1)

        if self.aggregate:
            with tf.control_dependencies([tf.py_func(lambda v: self.log_values.append(v), (inputs,), [])]):
                with tf.control_dependencies([tf.py_func(partial(cond_flush, buffer_logger=self),
                                                         (self.log_step,), ())]):
                    self.dependency = tf.identity(self.dependency)
        else:
            with tf.control_dependencies([tf.py_func(partial(cond_flush, buffer_logger=self),
                                                     (self.log_step, inputs), ())]):
                self.dependency = tf.identity(self.dependency)

        return self.dependency


def cond_flush(step: int, values: np.ndarray = None, buffer_logger: 'BufferLogger' = None) -> None:
    if values is None:
        values = buffer_logger.log_values
    if step % buffer_logger.freq == 0:
        buffer_logger.handler(values)
        buffer_logger.log_values = []


class CollectorLogger(EnvironmentCallbacks):
    def __init__(self, name: str, preprocessed_screen: tf.Tensor, reward_prediction: tf.Tensor,
                 preds_clusterer: 'PredictionClusterer', action_for_screen: Callable,
                 to_preprocessed_coord: Callable, cfg: Config):
        self.scalar_log_frequency = cfg['scalar_log_frequency']
        self.image_log_frequency = cfg['image_log_frequency']
        self.coverage_log_frequency = cfg['coverage_log_frequency']
        self.prediction_overlay_factor = cfg['prediction_overlay_factor']
        self.cluster_color_size = cfg['cluster_color_size']
        self.dir = cfg['dir']
        self.steps_per_new_file = cfg['steps_per_new_file']
        self.log_preprocessed_screen = cfg['log_preprocessed_screen']
        self.log_reward_prediction = cfg['log_reward_prediction']
        self.steps_per_app = cfg['steps_per_app']
        self.chunk_start = cfg['chunk_start']
        self.environment = None

        assert self.coverage_log_frequency % self.scalar_log_frequency == 0

        self.name = name
        self.action_for_screen = action_for_screen
        self.to_preprocessed_coord = to_preprocessed_coord

        self.local_step = 0
        self.rewards = []
        self.activity_count = []
        self.action = None
        self.action_metadata = None
        self.preprocessed_screen = None
        self.prediction = None
        self.clustering = None
        self.scalars = {}
        self.summary_writer = None
        self.summary = tf.Summary()

        preds_clusterer.add_callback(self.on_new_clustering)
        self.dependencies = [BufferLogger(self.image_log_frequency,
                                          self.on_new_preprocessed_screen, False)(preprocessed_screen[0])]
        if self.log_reward_prediction:
            self.dependencies += [
                BufferLogger(self.image_log_frequency, self.on_new_prediction, False)(reward_prediction[0]),
                BufferLogger(self.scalar_log_frequency,
                             partial(self.on_new_scalar,
                                     'Predictions/Mean'), True)(tf.reduce_mean(reward_prediction[0])),
                BufferLogger(self.scalar_log_frequency,
                             partial(self.on_new_scalar,
                                     'Predictions/Std'), True)(tf.math.reduce_std(reward_prediction[0]))
            ]

    def get_chunk(self) -> int:
        return self.local_step // self.steps_per_new_file + self.chunk_start

    def set_environment(self, environment: RelevantActionEnvironment):
        self.environment = environment
        self.environment.add_on_crash_callback(self.on_crash)
        self.environment.phone.add_action_metadata_callback(self.on_action_metadata)

    def get_dependencies(self) -> List[tf.Tensor]:
        return self.dependencies

    def on_new_clustering(self, clickables: List[tf.Tensor], clusters: List[np.ndarray],
                          valid_clusters_nums: List[np.ndarray]) -> None:
        self.clustering = ([x.numpy() for x in clickables], clusters, valid_clusters_nums)

    def on_new_prediction(self, prediction: np.ndarray) -> None:
        self.prediction = prediction

    def on_new_scalar(self, name: str, values: List[np.ndarray]) -> None:
        self.scalars[name] = values

    def on_crash(self) -> None:
        self.log_scalar('Crashes', 0)
        self.log_scalar('Crashes', 1)
        self.log_scalar('Crashes', 0)

    def on_action_metadata(self, metadata: Any) -> None:
        self.action_metadata = metadata

    def on_state_change(self, src_state: np.ndarray, action: Any, dst_state: np.ndarray, reward: float) -> None:
        self.local_step += 1
        self.rewards.append(np.array(reward))
        valid_visited_activities = {x for x in self.environment.phone.visited_activities
                                    if x.startswith(self.environment.get_current_app(step=self.local_step - 1))}
        self.activity_count.append(np.array(len(valid_visited_activities)))
        if self.local_step % self.scalar_log_frequency == 0:
            self.log_scalar('Metrics/Reward', self.rewards)
            if self.environment.phone.maintain_visited_activities:
                self.log_scalar('Metrics/Activity Count', self.activity_count)
                self.log_scalar('Metrics/Activity Count Percentage',
                                np.array(self.activity_count) / len(self.environment.phone.get_app_all_activities(
                                    self.environment.get_current_app(apk=True, step=self.local_step - 1))))
            if self.local_step % self.coverage_log_frequency == 0:
                coverages = self.environment.phone.update_code_coverage(
                    self.environment.get_current_app(apk=True, step=self.local_step - 1),
                    f'{self.name}_{self.get_chunk()}_{self.local_step}')
                if coverages is None:
                    coverages = [np.nan] * 4
                self.log_scalar('Coverage/Class', coverages[0])
                self.log_scalar('Coverage/Method', coverages[1])
                self.log_scalar('Coverage/Block', coverages[2])
                self.log_scalar('Coverage/Line', coverages[3])
            for name in self.scalars:
                self.log_scalar(name, self.scalars[name])
            self.rewards = []
            self.activity_count = []
            self.scalars = {}
        if self.local_step % self.image_log_frequency == 0:
            # assert self.action is None
            self.action = action
            self.log_screen('Screen/Original', src_state, lambda x: x, self.environment.animation_mask)
            if self.log_preprocessed_screen:
                self.log_screen('Screen/Preprocessed', self.preprocessed_screen, self.to_preprocessed_coord)
            if self.log_reward_prediction:
                self.log_predictions(self.prediction, self.clustering)
        self.action = None
        self.preprocessed_screen = None
        self.clustering = None

    def write_summary(self) -> None:
        if self.summary_writer is not None:
            self.summary_writer.add_summary(self.summary, self.local_step)

    def on_wait(self) -> None:
        self.write_summary()
        self.summary = tf.Summary()
        if self.local_step % self.steps_per_new_file == 0:
            chunk = self.get_chunk()
            if self.summary_writer is not None:
                self.summary_writer.close()
            self.summary_writer = tf.summary.FileWriter(f'{self.dir}/{self.name}_chunk_{chunk}')
            print(f'{datetime.now()}: Changed log file to chunk {chunk} for {self.name}.')
        if self.local_step % self.steps_per_app == 0:
            self.log_scalar('Crashes', 0)
            self.log_scalar('Coverage/Class', 0.0)
            self.log_scalar('Coverage/Method', 0.0)
            self.log_scalar('Coverage/Block', 0.0)
            self.log_scalar('Coverage/Line', 0.0)
            self.summary_writer.add_summary(self.summary, self.local_step)
            self.summary = tf.Summary()

    def on_environment_finished(self) -> None:
        self.write_summary()
        self.summary_writer.close()

    def on_new_preprocessed_screen(self, screen: np.ndarray) -> None:
        # assert self.preprocessed_screen is None
        self.preprocessed_screen = (screen * 255).astype(np.uint8)

    def log_screen(self, name: str, screen: np.ndarray, point_transformer: Callable, mask: bool = None) -> None:
        screen = screen.copy()
        if screen.shape[-1] == 1:
            screen = np.concatenate([screen] * 3, axis=-1)
        if mask is not None:
            screen[mask == 0] = [100, 255, 0]
        action_p = point_transformer(self.action_for_screen(self.action[:2])).astype(np.int32)
        if self.action[-1] == 0:
            screen[max(0, action_p[0] - 3):action_p[0] + 3, max(0, action_p[1] - 3):action_p[1] + 3] = [255, 0, 0]
        elif self.action[-1] == 1:
            diff = self.action_metadata
            screen[action_p[0]:max(0, action_p[0] + diff):diff // abs(diff), max(0, action_p[1] - 3):action_p[1] + 3] \
                = [255, 0, 0]
            screen[max(0, action_p[0] - 3):action_p[0] + 3, max(0, action_p[1] - 3):action_p[1] + 3] = [0, 0, 255]
        elif self.action[-1] == 2:
            diff = self.action_metadata
            screen[max(0, action_p[0] - 3):action_p[0] + 3, action_p[1]:max(0, action_p[1] + diff):diff // abs(diff)] \
                = [255, 0, 0]
            screen[max(0, action_p[0] - 3):action_p[0] + 3, max(0, action_p[1] - 3):action_p[1] + 3] = [0, 0, 255]
        elif self.action[-1] == 3:
            text = self.action_metadata
            screen[max(0, action_p[0] - 3):action_p[0] + 3, max(0, action_p[1] - 3):action_p[1] + 3] = [0, 0, 255]
        else:
            raise NotImplementedError('Unsupported action.')

        self.log_image(name, screen)

    def log_predictions(self, pred: np.ndarray,
                        clusterings: Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]) -> None:
        type_count = pred.shape[-1]
        new_size = self.preprocessed_screen.shape[:2]
        final_pred_size = (new_size[0] * int(type_count ** .5), new_size[1] * (type_count // int(type_count ** .5)))
        final_pred = np.zeros((*final_pred_size, 3), dtype=np.uint8)
        original_pred = np.uint8(cm.viridis(pred)[:, :, :, :3] * 255)
        for type in range(type_count):
            pred = original_pred[:, :, type, :]
            if clusterings is not None or self.prediction_overlay_factor > 0:
                prev_size = pred.shape[:2]
                pred = Image.fromarray(pred)
                pred = pred.resize(new_size)
                pred = Image.blend(pred, Image.fromarray(self.preprocessed_screen), self.prediction_overlay_factor)
                pred = np.array(pred)
                if clusterings is not None:
                    clustering = tuple(x[type] for x in clusterings)
                    cluster_colors = cm.Reds(np.linspace(0, 1, len(clustering[2])))[:, :3] * 255
                    for cluster in zip(*clustering[:2]):
                        cl_ind = np.where(clustering[2] == cluster[1])[0]
                        if len(cl_ind) > 0:
                            y, x = tuple(cluster[0][:2] * new_size // prev_size)
                            pred[y:y + self.cluster_color_size, x:x + self.cluster_color_size] \
                                = cluster_colors[cl_ind[0]]
            type_x = type // (final_pred_size[1] // new_size[1])
            type_y = type % (final_pred_size[1] // new_size[1])
            final_pred[type_x * new_size[0]:(type_x + 1) * new_size[0], type_y * new_size[1]:
                                                                        (type_y + 1) * new_size[1]] = pred
        self.log_image('Predictions', final_pred)

    def log_image(self, name: str, image: np.ndarray) -> None:
        self.summary.value.add(tag=name, image=get_image_summary(image))

    def log_scalar(self, name: str, values: Union[float, int, np.ndarray, List[Union[np.ndarray, int, float]]]) -> None:
        self.summary.value.add(tag=name, simple_value=np.mean(values))


def get_image_summary(image: np.ndarray) -> tf.Summary.Image:
    bio = BytesIO()
    Image.fromarray(image).save(bio, format="png")
    res = tf.Summary.Image(encoded_image_string=bio.getvalue(), height=image.shape[0], width=image.shape[1])
    bio.close()
    return res


def index_to_action(index: tf.Tensor, preds: tf.Tensor) -> tf.Tensor:
    shape = preds.shape[1:]
    y = tf.cast(index // np.prod(shape[1:]), tf.int32)
    x = tf.cast((index // shape[2]) % shape[1], tf.int32)
    type = tf.cast(index % shape[2], tf.int32)
    return tf.transpose(tf.concat([[y], [x], [type]], axis=0))


def linear_normalizer(logits: tf.Tensor, axis=None) -> tf.Tensor:
    return logits / (tf.reduce_sum(logits, axis=axis) + keras.backend.epsilon())


def most_probable_weighted_policy_user(logits: tf.Tensor) -> tf.Tensor:
    if prediction_normalizer is None:
        return tf.argmax(tf.distributions.Multinomial(1.0, logits=logits).sample(), axis=-1)
    return tf.argmax(tf.distributions.Multinomial(1.0, probs=prediction_normalizer(logits, axis=-1)).sample(), axis=-1)


def better_reward_to_action(preds: tf.Tensor) -> tf.Tensor:
    preds = preds * action_prob_coeffs_tensor
    preds_f = tf.reshape(preds, (-1, np.prod(preds.shape[1:])))
    return index_to_action(most_probable_weighted_policy_user(preds_f), preds)


def worse_reward_to_action(preds: tf.Tensor) -> tf.Tensor:
    return better_reward_to_action(-preds + 1)


def least_certain_reward_to_action(preds: tf.Tensor) -> tf.Tensor:
    return better_reward_to_action(.5 - tf.abs(-preds + .5))


def most_certain_reward_to_action(preds: tf.Tensor) -> tf.Tensor:
    return better_reward_to_action(tf.abs(-preds + .5))


def random_reward_to_action(preds: tf.Tensor) -> tf.Tensor:
    return better_reward_to_action(tf.ones_like(preds, dtype=tf.float32))


class PredictionClusterer:
    def __init__(self, cfg: Config):
        self.start_clickable_threshold = cfg['start_clickable_threshold']
        self.clickable_threshold_speed = cfg['clickable_threshold_speed']
        self.clickable_threshold_speed_step = cfg['clickable_threshold_speed_step']
        self.speed_steps_per_clickable_threshold_reset = cfg['speed_steps_per_clickable_threshold_reset']
        self.distance_threshold = cfg['distance_threshold']
        self.cluster_count_threshold = cfg['cluster_count_threshold']

        self.callbacks = []
        self.steps = 0

    def add_callback(self, callback: Callable) -> None:
        self.callbacks.append(callback)

    def __call__(self, preds: tf.Tensor) -> tf.Tensor:
        clickable_threshold = self.start_clickable_threshold - self.clickable_threshold_speed * \
                              ((self.steps // self.clickable_threshold_speed_step) %
                               self.speed_steps_per_clickable_threshold_reset)
        self.steps += 1
        if preds.shape[0] > 1:
            raise NotImplementedError('cluster reward is not implemented for batch size > 1.')
        preds_old = preds
        type_count = preds.shape[-1]

        all_clickables = []
        all_clusters = []
        all_valid_clusters_nums = []
        for type in range(type_count):
            all_clickables.append([])
            all_clusters.append([])
            all_valid_clusters_nums.append([])
            preds = preds_old[0, :, :, type]
            clickables = tf.cast(tf.where(preds > clickable_threshold), tf.int32)
            if len(clickables) == 0 or len(clickables) == 1:
                all_clickables[-1] = clickables
                continue
            clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=self.distance_threshold,
                                                compute_full_tree=True, linkage='single')
            clusters = clusterer.fit_predict(clickables)
            clusters_nums, clusters_counts = np.unique(clusters, axis=0, return_counts=True)
            valid_clusters_nums = clusters_nums[clusters_counts >= self.cluster_count_threshold]
            if len(valid_clusters_nums) == 0:
                all_clickables[-1] = clickables
                continue
            all_clickables[-1] = clickables
            all_clusters[-1] = clusters
            all_valid_clusters_nums[-1] = valid_clusters_nums

        if sum(map(len, all_clusters)) == 0:
            return random_reward_to_action(preds_old)

        valid_types = [i for i in range(len(all_clusters)) if len(all_clusters[i]) > 0]
        valid_type_coeffs = np.array([action_prob_coeffs[i]
                                      for i in range(len(all_clusters)) if len(all_clusters[i]) > 0])
        chosen_type = np.random.choice(valid_types, 1, p=valid_type_coeffs / np.sum(valid_type_coeffs))[0]
        valid_clusters_nums = all_valid_clusters_nums[chosen_type]
        clickables = all_clickables[chosen_type]
        clusters = all_clusters[chosen_type]
        chosen_cluster_num = np.random.choice(valid_clusters_nums, 1)
        chosen_clickables = tf.boolean_mask(clickables, clusters == chosen_cluster_num)
        chosen_clickable_i = most_probable_weighted_policy_user(tf.gather_nd(preds, chosen_clickables))
        chosen_clickable = tf.concat([tf.gather(chosen_clickables, chosen_clickable_i), [chosen_type]], axis=-1)
        for callback in self.callbacks:
            callback(all_clickables, all_clusters, all_valid_clusters_nums)
        return tf.expand_dims(chosen_clickable, axis=0)


def combine_prediction_to_actions(prediction_to_actions: List[Callable], probs: List[float]) -> Callable:
    assert abs(sum(probs) - 1) < 1e-6

    def to_actions(preds: tf.Tensor) -> tf.Tensor:
        return np.random.choice(prediction_to_actions, 1, False, probs)[0](preds)

    return lambda inp: tf.py_function(to_actions, [inp], tf.int32)


def prediction_sampler(predictions: tf.Tensor, actions: tf.Tensor) -> tf.Tensor:
    return tf.expand_dims(tf.gather_nd(predictions, actions, batch_dims=1), axis=-1)


def remove_logs(log_dir: str, reset_logs: bool) -> None:
    while reset_logs and os.path.isdir(log_dir) and len(os.listdir(log_dir)) > 0 and \
            len(os.listdir(f'{log_dir}/{os.listdir(log_dir)[0]}')) > 0:
        for agent_dir in os.listdir(log_dir):
            try:
                for f in os.listdir(f'{log_dir}/{agent_dir}'):
                    os.unlink(f'{log_dir}/{agent_dir}/{f}')
            except FileNotFoundError:
                pass


def transform_linearly(coord: np.ndarray, ratio: np.ndarray, offset: np.ndarray, dtype=None) -> np.ndarray:
    res = coord * ratio + offset
    if dtype is not None:
        res = res.astype(dtype)
    return res


# maybe log some of these
def distort_episode(episode: Episode, distort_color_probability: float) -> Episode:
    color_order = np.arange(3)
    if np.random.rand() < distort_color_probability:
        while np.all(color_order == np.arange(3)):
            color_order = np.random.permutation(3)
    state = episode.state.copy()[:, :, color_order]
    # note that I am not distorting the result here. If I use it I have to distort it too.
    return Episode(state, episode.action.copy(), episode.reward.copy(), episode.result.copy())


def control_dependencies(inputs) -> tf.Tensor:
    x, dependencies = inputs
    with tf.control_dependencies(dependencies):
        x = tf.identity(x)
    return x


def linear_combination(funcs: List[Callable], coeffs: List[float] = None) -> Callable:
    return lambda *inps: sum([c * f(inps) for c, f in zip(coeffs, funcs)])


def preds_variance_regularizer(preds: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.reduce_mean(tf.math.reduce_variance(preds, axis=[1, 2]), axis=-1))


def create_agent(id: int, agent_num: int, agent_name: str, is_learner: bool, is_tester: bool,
                 agent_option_probs: List[float], agent_clusterer_cfg_name: str,
                 weights_file: str) -> Union[DataCollectionAgent, LearningAgent]:
    environment_configs = cfg['environment_configs']
    learner_configs = cfg['learner_configs']
    collector_configs = cfg['collector_configs']
    screen_preprocessor_configs = cfg['screen_preprocessor_configs']
    phone_configs = cfg['phone_configs']
    collector_logger_configs = cfg['collector_logger_configs']
    dummy_mode = cfg['dummy_mode']
    monkey_client_mode = cfg['monkey_client_mode']
    data_file_dir = cfg['data_file_dir']
    logs_dir = cfg['logs_dir']
    collectors_apks_path = cfg['collectors_apks_path']
    testers_apks_path = cfg['testers_apks_path']
    collectors_clone_script = cfg['collectors_clone_script']
    testers_clone_script = cfg['testers_clone_script']
    distort_color_probability = cfg['distort_color_probability']
    reward_predictor = cfg['reward_predictor']
    prediction_shape = cfg['prediction_shape']
    variance_reg_coeff = cfg['variance_reg_coeff']
    l1_reg_coeff = cfg['l1_reg_coeff']
    use_logger = cfg['use_logger']
    screen_shape = phone_configs['screen_shape']
    adb_path = phone_configs['adb_path']
    scroll_min_value = phone_configs['scroll_min_value']
    scroll_max_value = phone_configs['scroll_max_value']
    scroll_event_count = phone_configs['scroll_event_count']
    action_type_count = environment_configs['action_type_count']
    steps_per_app = environment_configs['steps_per_app']
    batch_size = learner_configs['batch_size'] if is_learner else 1
    screen_preprocessor_resize_size = screen_preprocessor_configs['resize_size']
    screen_preprocessor_crop_top_left = screen_preprocessor_configs['crop_top_left']
    screen_preprocessor_crop_size = screen_preprocessor_configs['crop_size']
    reward_predictor_configs = cfg[f'{reward_predictor[1]}_reward_predictor_configs']

    environment_configs['pos_reward'] = pos_reward
    environment_configs['neg_reward'] = neg_reward
    environment_configs['steps_per_episode'] = 1
    environment_configs['crop_top_left'] = screen_preprocessor_crop_top_left
    environment_configs['crop_size'] = screen_preprocessor_crop_size
    phone_configs['crop_top_left'] = screen_preprocessor_crop_top_left
    phone_configs['crop_size'] = screen_preprocessor_crop_size
    phone_configs['apks_path'] = testers_apks_path if is_tester else collectors_apks_path
    phone_configs['clone_script_path'] = testers_clone_script if is_tester else collectors_clone_script
    collector_configs['file_dir'] = data_file_dir
    if is_tester:
        collector_configs['max_file_size'] = 0
    learner_configs['file_dir'] = data_file_dir
    learner_configs['logs_dir'] = logs_dir
    collector_logger_configs['dir'] = logs_dir
    collector_logger_configs['steps_per_app'] = steps_per_app
    reward_predictor_configs['prediction_shape'] = prediction_shape
    monkey_client_configs = {'adb_path': adb_path, 'scroll_min_value': scroll_min_value,
                             'scroll_max_value': scroll_max_value, 'scroll_event_count': scroll_event_count}

    screen_preprocessor_resize_size_a = np.array(screen_preprocessor_resize_size)
    screen_preprocessor_crop_top_left_a = np.array(screen_preprocessor_crop_top_left)
    screen_preprocessor_crop_size_a = np.array(screen_preprocessor_crop_size)
    prediction_shape_a = np.array(prediction_shape)

    def action_pos_to_screen_pos(action_p: np.ndarray, dtype=None) -> np.ndarray:
        return transform_linearly(action_p + .5, screen_preprocessor_crop_size_a / prediction_shape_a,
                                  screen_preprocessor_crop_top_left_a, dtype)

    example_episode = Episode(np.zeros((*screen_shape, 3), np.uint8), np.zeros(3, np.int32),
                              np.zeros((), np.bool), np.zeros((*screen_shape, 3), np.uint8))

    screen_input = keras.layers.Input(example_episode.state.shape, batch_size, name='state',
                                      dtype=example_episode.state.dtype)

    screen_preprocessor = ScreenPreprocessor(screen_preprocessor_configs, name='screen_preprocessor')
    reward_predictor = eval(reward_predictor[0])(action_type_count, 2, reward_predictor_configs,
                                                 name='reward_predictor')

    predictions = reward_predictor(screen_preprocessor(screen_input))

    if is_learner:
        regs, coeffs = [], []
        if variance_reg_coeff != 0:
            regs.append(preds_variance_regularizer)
            coeffs.append(-variance_reg_coeff)
        if l1_reg_coeff != 0:
            regs.append(keras.regularizers.l1(l1_reg_coeff))
            coeffs.append(1)
        if len(regs) > 0:
            reward_predictor.activity_regularizer = linear_combination(regs, coeffs)
        action_input = keras.layers.Input(example_episode.action.shape, batch_size, name='action',
                                          dtype=example_episode.action.dtype)
        input = (screen_input, action_input)
        action = keras.layers.Lambda(lambda elems: prediction_sampler(elems[0], elems[1]),
                                     name='action_sampler')((predictions, action_input))
    else:
        input = screen_input
        built_prediction_to_action_options = [prediction_to_action_options[0](agent_clusterer_cfg_name)] + \
                                             prediction_to_action_options[1:]
        action = keras.layers.Lambda(
            combine_prediction_to_actions(built_prediction_to_action_options, agent_option_probs),
            name='action_predictor')(predictions)

        if use_logger:
            logger = CollectorLogger(f'{agent_name}_{"tester" if is_tester else "collector"}{id}',
                                     screen_preprocessor.output, reward_predictor.output,
                                     built_prediction_to_action_options[0], action_pos_to_screen_pos,
                                     lambda p: transform_linearly(p - screen_preprocessor_crop_top_left_a,
                                                                  screen_preprocessor_resize_size_a /
                                                                  screen_preprocessor_crop_size_a, np.array([0, 0])),
                                     collector_logger_configs)

            action = keras.layers.Lambda(control_dependencies,
                                         name='log_dependency_controller')((action, logger.get_dependencies()))

    model = keras.Model(inputs=input, outputs=action)

    if weights_file is not None:
        model.load_weights(weights_file, by_name=True)
    if is_learner:
        model.compile(optimizer, keras.losses.BinaryCrossentropy())

    def action2pos(action: np.ndarray) -> Tuple[int, int, int]:
        pos = action_pos_to_screen_pos(action[:2], np.int32)
        return pos[1], pos[0], action[2]

    def create_environment(collector: DataCollectionAgent) -> Environment:
        if monkey_client_mode:
            env = RelevantActionMonkeyClient(collector, action2pos, 3000 + agent_num,
                                             5554 + 2 * agent_num, monkey_client_configs)
            return env
        else:
            phone_type = DummyPhone if dummy_mode else Phone
            env = RelevantActionEnvironment(collector, phone_type(('tester' if is_tester else 'collector') + str(id),
                                                                  5554 + 2 * agent_num, phone_configs),
                                            action2pos, environment_configs)
            if use_logger:
                env.add_callback(logger)
                logger.set_environment(env)
            return env

    if is_learner:
        return LearningAgent(id, model,
                             partial(distort_episode, distort_color_probability=distort_color_probability),
                             learner_configs)
    else:
        return DataCollectionAgent(id, model, example_episode, create_environment, collector_configs)


def parse_specs_to_probs_and_ops(specs: Dict, max_len: int) -> List:
    return sum([[([spec[1][i] if len(spec[1]) > i else (1 - sum(spec[1])) / (max_len - len(spec[1]))
                   for i in range(max_len)], *spec[2:])] * spec[0]
                for spec in specs], [])


optimizer = 'adam'
pos_reward = 1
neg_reward = 0
with open('configs.yaml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
collectors = cfg['collectors']
testers = cfg['testers']
reset_logs = cfg['reset_logs']
coordinator_configs = cfg['coordinator_configs']
clusterer_configs = cfg['clusterer_configs']
logs_dir = cfg['logs_dir']
weights_file = cfg['weights_file']
prediction_normalizer_name = cfg['prediction_normalizer']
collector_version_start = cfg['collector_configs']['version_start']
shuffle_apps = cfg['environment_configs']['shuffle_apps']
action_prob_coeffs = cfg['action_prob_coeffs']

coordinator_configs['collector_version_start'] = collector_version_start

prediction_normalizer = None if prediction_normalizer_name is None else eval(prediction_normalizer_name)
action_prob_coeffs_tensor = tf.convert_to_tensor([[[action_prob_coeffs]]])

for clusterer_cfg_name in clusterer_configs:
    if clusterer_cfg_name == 'default':
        continue
    for clusterer_cfg_attr in clusterer_configs['default']:
        if clusterer_cfg_attr not in clusterer_configs[clusterer_cfg_name]:
            clusterer_configs[clusterer_cfg_name][clusterer_cfg_attr] = \
                clusterer_configs['default'][clusterer_cfg_attr]

prediction_to_action_options = [
    lambda cfg_name: PredictionClusterer(clusterer_configs[cfg_name if cfg_name is not None else 'default']),
    better_reward_to_action, worse_reward_to_action, most_certain_reward_to_action,
    least_certain_reward_to_action, random_reward_to_action]
collector_option_probs_and_ops = parse_specs_to_probs_and_ops(collectors, len(prediction_to_action_options))
tester_option_probs_and_ops = parse_specs_to_probs_and_ops(testers, len(prediction_to_action_options))

tf.disable_v2_behavior()
os.environ["KMP_AFFINITY"] = "verbose"

if __name__ == '__main__':
    remove_logs(logs_dir, reset_logs)
    collector_creators = [partial(create_agent, i, i, probs_and_ops[1] if len(probs_and_ops) > 1 else '',
                                  False, False, probs_and_ops[0], probs_and_ops[2] if len(probs_and_ops) > 2 else None,
                                  weights_file[probs_and_ops[3]] if len(probs_and_ops) > 3 else None)
                          for i, probs_and_ops in enumerate(collector_option_probs_and_ops)]
    tester_creators = [partial(create_agent, i, i + len(collector_creators),
                               probs_and_ops[1] if len(probs_and_ops) > 1 else '', False, True,
                               probs_and_ops[0], probs_and_ops[2] if len(probs_and_ops) > 2 else None,
                               weights_file[probs_and_ops[3]] if len(probs_and_ops) > 3 else None)
                       for i, probs_and_ops in enumerate(tester_option_probs_and_ops)]
    learner_creator = partial(create_agent, *2 * (len(collector_creators) + len(tester_creators),), 'learner',
                              True, False, None, None, weights_file['learner'])
    coord = ProcessBasedCoordinator(collector_creators, learner_creator, tester_creators, cfg['coordinator_configs'])
    coord.start()

# metrics
# add prints to error callback of agents
