import glob
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, List, Dict, Tuple, Callable, Optional, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow_core.python.keras.callbacks import LambdaCallback

from environment import EnvironmentCallbacks, EnvironmentController, Environment
from parallelism import Thread, Process
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


class TestingAgent(DataCollectionAgent):
    def __init__(self, id: int, model: keras.Model, example_episode: Episode,
                 create_environment: Callable[['DataCollectionAgent'], Environment], cfg: Config):
        self.learn = cfg['learn']
        self.weights_file = cfg['weights_file']
        self.weight_reset_frequency = cfg['weight_reset_frequency']
        self.version_window = cfg['version_window']
        past_rewards_window = cfg['past_rewards_window']
        self.past_rewards_threshold = cfg['past_rewards_threshold']
        self.loss_threshold = cfg['loss_threshold']

        max_episodes = cfg['max_episodes']
        max_file_size = cfg['max_file_size']
        file_dir = cfg['file_dir']
        meta_save_frequency = max_file_size
        version_start = 0

        collection_cfg = {
            'max_episodes': max_episodes,
            'max_file_size': max_file_size if self.learn else 0,
            'meta_save_frequency': meta_save_frequency,
            'file_dir': file_dir,
            'version_start': version_start
        }

        super().__init__(id, model, example_episode, create_environment, collection_cfg)

        self.steps = 0
        self.most_recent_weights = None
        self.next_file_valid = True
        self.first_valid_version = 0
        self.is_learning = False
        self.weight_reset_callbacks = []
        self.learning_request_callback = None
        self.past_rewards = deque(maxlen=past_rewards_window)

        self.add_on_file_completed_callbacks(self.on_file_completed)

    def reset_file(self, new_file: bool = True):
        super().reset_file(new_file)
        self.next_file_valid = True

    def set_learning_request_callback(self, callback: Callable) -> Callable:
        self.learning_request_callback = callback

        def done_callback():
            self.is_learning = False

        return done_callback

    def on_file_completed(self, id: int, file_version: int):
        if self.next_file_valid:
            past_rewards_sum = sum(self.past_rewards) / len(self.past_rewards)
            print(f'{datetime.now()}: past {len(self.past_rewards)} rewards sum '
                  f'in tester {self.id} is {past_rewards_sum}.')
            if past_rewards_sum < self.past_rewards_threshold and not self.is_learning:
                self.is_learning = True
                print(f'{datetime.now()}: sending learning request in tester {self.id}.')
                self.learning_request_callback(self.id, list(range(max(
                    file_version + 1 - self.version_window, self.first_valid_version),
                    file_version + 1)), self.loss_threshold)
        else:
            self.first_valid_version = file_version + 1

    def update_weights(self, weights: List[tf.Tensor]):
        super().update_weights(weights)
        if self.learn:
            self.most_recent_weights = weights

    def add_on_weight_reset_callbacks(self, callback: Callable) -> None:
        self.weight_reset_callbacks.append(callback)

    def reset_weights(self):
        if not self.learn:
            return
        if self.steps > 0:
            self.next_file_valid = False
        print(f'{datetime.now()}: resetting weights for tester {self.id} with weights from ', end='')
        self.past_rewards.clear()
        if self.most_recent_weights is not None:
            print('global learner')
            for callback in self.weight_reset_callbacks:
                callback(self.id)
            self.update_weights(self.most_recent_weights)
        elif self.weights_file is not None:
            print('file')
            for callback in self.weight_reset_callbacks:
                callback(self.id, self.weights_file)
            self.model.load_weights(self.weights_file, by_name=True)

    def on_episode_start(self, state: np.ndarray) -> None:
        if self.learn and self.weight_reset_frequency is not None and self.steps % self.weight_reset_frequency == 0:
            self.reset_weights()
        super().on_episode_start(state)

    def on_state_change(self, src_state: np.ndarray, action: Any, dst_state: np.ndarray, reward: float) -> None:
        self.past_rewards.append(reward)
        super().on_state_change(src_state, action, dst_state, reward)

    def on_episode_end(self, premature: bool) -> None:
        self.steps += 1
        super().on_episode_end(premature)


class EarlyStoppingByLossVal(keras.callbacks.Callback):
    def __init__(self, monitor, value, stop_predicate: Callable, batch_end_callback: Callable = None):
        super().__init__()
        self.monitor = monitor
        self.value = value
        self.stop_predicate = stop_predicate
        self.batch_end_callback = batch_end_callback

    def on_batch_end(self, batch, logs=None):
        self.batch_end_callback()
        if self.stop_predicate():
            self.model.stop_training = True

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            raise RuntimeError(f'Early stopping requires {self.monitor} available!')

        if current < self.value:
            self.model.stop_training = True


class LearningAgent:
    def __init__(self, id: int, model: keras.Model, iic_distorter: Optional[Callable], cfg: Config):
        self.file_dir = cfg['file_dir']
        self.shuffle = cfg['shuffle']
        self.correct_distributions = cfg['correct_distributions']
        self.augmenting_correction = cfg['augmenting_correction']
        self.strict_correction = cfg['strict_correction']
        self.batch_size = cfg['batch_size']
        self.epochs_per_version = cfg['epochs_per_version']
        self.data_portion_per_epoch = cfg['data_portion_per_epoch']
        self.save_dir = cfg['save_dir']
        self.validation_dir = cfg['validation_dir']

        self.id = id
        # plot the model (maybe here or where it's created)
        self.model = model
        self.iic_distorter = iic_distorter
        self.stop_learning_callback = None
        self.is_learning = False

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
        # i can also not have this and instead use weights in keras

        if len(total_reward_indices) == 1:
            return np.array(-1), -1
        if len(total_reward_indices) != 2:
            raise NotImplementedError('Oops. Currently not supporting non-binary rewards.')
        less_represented_reward = list(total_reward_indices.keys())[np.argmin([len(total_reward_indices[reward])
                                                                               for reward in total_reward_indices])]
        augmented_size = abs(np.subtract(*[len(total_reward_indices[reward]) for reward in total_reward_indices]))
        return less_represented_reward, augmented_size

    @staticmethod
    def read_episode_files(directory: str, version: Union[int, List[int]]) -> \
            Tuple[List[EpisodeFile], List[int], List[Dict[np.ndarray, List[int]]], Episode]:
        if not isinstance(version, list):
            version = [version]

        example_episode = None
        episode_files = []
        file_sizes = []
        file_reward_indices_list = []
        meta_files = []
        for v in version:
            meta_files += glob.glob(f'{directory}/{v}/*.meta')
        for meta_file in meta_files:
            meta = load_obj(meta_file)
            example_episode = meta['example'] if example_episode is None else \
                LearningAgent.get_general_example(example_episode, meta['example'])
            episode_files.append(EpisodeFile(meta_file[:-5], meta['max_size'], meta['example'], 'r'))
            file_sizes.append(meta['size'])
            file_reward_indices_list.append(meta['reward_indices'])

        return episode_files, file_sizes, file_reward_indices_list, example_episode

    def create_training_data(self, directory: str, version: Union[int, List[int]]) -> Tuple[Optional[Callable], int]:
        episode_files, file_sizes, file_reward_indices_list, example_episode = self.read_episode_files(directory,
                                                                                                       version)

        if len(episode_files) == 0:
            return None, 0

        total_reward_indices = self.merge_reward_indices_list(file_reward_indices_list)
        if self.correct_distributions:
            less_represented_reward, augmented_size = self.correct_distribution(total_reward_indices)
            more_represented_reward = [x for x in list(total_reward_indices.keys()) if x != less_represented_reward][0]
        else:
            augmented_size = 0

        if augmented_size == -1:
            if self.strict_correction:
                return None, 0
            augmented_size = 0

        total_size = sum(file_sizes)
        if self.augmenting_correction:
            training_size = total_size + augmented_size
        else:
            training_size = total_size - augmented_size

        if self.correct_distributions and augmented_size != 0:
            poses1 = np.random.choice(np.array(total_reward_indices[more_represented_reward], dtype='int, int'),
                                      training_size // 2, replace=False)
            inds = np.array(total_reward_indices[less_represented_reward], dtype='int, int')
            rem_size = training_size // 2
            arrs = []
            while rem_size >= len(inds):
                arrs.append(inds)
                rem_size -= len(inds)
            if rem_size > 0:
                arrs.append(np.random.choice(inds, rem_size, replace=False))
            poses2 = np.concatenate(arrs)
            positions = np.concatenate([poses1, poses2])
        else:
            positions = np.concatenate([np.array(inds, dtype='int, int') for inds in total_reward_indices.values()])
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
                    if self.iic_distorter is not None:
                        x['state2'] = x['state'].copy()
                        x['action2'] = x['action'].copy()
                        x['result2'] = x['result'].copy()
                        y2 = np.zeros((batch_size, 1), dtype=np.int32)

                    for i in range(batch_size):
                        position = positions[(current_positions_i + i) % training_size]
                        file_i = position[0]
                        data_i = position[1]
                        episode = episode_files[file_i].get(data_i)
                        x['state'][i] = episode.state
                        x['action'][i] = episode.action
                        x['result'][i] = episode.result
                        y[i][0] = episode.reward
                        if self.iic_distorter is not None:
                            episode2, mask, mask2 = self.iic_distorter(episode)
                            x['state2'][i] = episode2.state
                            x['action2'][i] = episode2.action
                            x['result2'][i] = episode2.result
                            if 'iic_mask' not in x:
                                x['iic_mask'] = np.zeros((batch_size, *mask.shape), dtype=np.float32)
                                x['iic_mask2'] = np.zeros((batch_size, *mask.shape), dtype=np.float32)
                            x['iic_mask'][i] = mask
                            x['iic_mask2'][i] = mask2
                            y2[i][0] = episode2.reward

                    if self.iic_distorter is None:
                        yield x, y
                    else:
                        yield x, (y, y2)

                    current_positions_i = (current_positions_i + self.batch_size) % training_size

        return generator, max(training_size, self.batch_size)

    def stop_if_learning(self, callback: Callable) -> None:
        if self.is_learning:
            self.stop_learning_callback = callback

    # add logs
    def learn(self, version: Union[int, List[int]], loss_threshold: int = None,
              batch_end_callback: Callable = None) -> None:
        generator, data_size = self.create_training_data(self.file_dir, version)
        if self.validation_dir is not None:
            validation_generator, validation_data_size = self.create_training_data(self.validation_dir, version)
        else:
            validation_generator = None
            validation_data_size = 0
        if generator is None or (self.validation_dir is not None and validation_generator is None):
            print(f'{datetime.now()}: In learner {self.id}, the experience version {version} '
                  f'is not expressive enough to learn/validate from.')
        else:
            print(f'{datetime.now()}: starting learning for experience version {version} in learner {self.id}')
            data = generator()
            validation_data = None if validation_generator is None else validation_generator()
            data_size = int(data_size * self.data_portion_per_epoch)
            steps_per_epoch = int(data_size * self.epochs_per_version / self.batch_size / int(self.epochs_per_version))
            validation_steps = int(validation_data_size / self.batch_size)

            lambda_callback = LambdaCallback(on_batch_end=lambda epoch, logs: print())
            callbacks = [lambda_callback]
            if self.save_dir is not None:
                checkpoint_callback = keras.callbacks.ModelCheckpoint(
                    f'{self.save_dir}/{version[-1] if isinstance(version, list) else version}-' +
                    '{epoch:02d}-loss_{loss:.2f}-val-loss_{val_loss:.2f}.hdf5',
                    monitor='val_loss', save_best_only=False, save_weights_only=True,
                    save_freq='epoch')
                callbacks.append(checkpoint_callback)
            if loss_threshold is not None:
                stop_callback = EarlyStoppingByLossVal('loss', loss_threshold,
                                                       lambda: self.stop_learning_callback is not None,
                                                       batch_end_callback)
                callbacks.append(stop_callback)

            self.stop_learning_callback = None
            self.is_learning = True
            self.model.fit(data, validation_data=validation_data, validation_steps=validation_steps,
                           epochs=int(self.epochs_per_version), steps_per_epoch=steps_per_epoch,
                           callbacks=callbacks)
            self.is_learning = False
            if self.stop_learning_callback is not None:
                self.stop_learning_callback()
                self.stop_learning_callback = None

            del data


class ThreadLocals:
    def __init__(self):
        self.thread = None
        self.collector = None
        self.new_weight = None
        self.new_tester_weight = None

    def pop_and_run_next(self, *local_args, wait=False) -> None:
        self.thread.pop_and_run_next(*local_args, wait=wait)


class Coordinator(ABC, EnvironmentCallbacks):
    def __init__(self, collector_creators: List[Callable[[], DataCollectionAgent]],
                 learner_creator: Callable[[], LearningAgent],
                 tester_creators: List[Union[int, Callable[[], TestingAgent]]],
                 tester_learner_creators: List[Callable[[], Union[None, LearningAgent]]], cfg: Config):
        self.collector_version_start = cfg['collector_version_start']
        self.train = cfg['train']
        self.pre_training = cfg['pre_training']
        self.collect_before_pre_training = cfg['collect_before_pre_training']
        self.sync_weight = cfg['sync_weight']

        self.collector_creators = collector_creators
        self.learner_creator = learner_creator
        if len(tester_creators) > 0:
            self.tester_ids, self.tester_creators = zip(*tester_creators)
        else:
            self.tester_ids, self.tester_creators = [], []
        self.tester_learner_creators = tester_learner_creators

        self.learner = None
        self.learner_thread_run_queue = deque()
        self.tester_learners = []
        self.file_completions = defaultdict(list)
        self.environment_completion_count = 0
        self.is_tester = None
        self.weight_reset_requested = []
        self.tester_reset_weight_file = []
        self.tester_in_learning = []
        self.learning_done_callback = None

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
        self.is_tester = isinstance(collector, TestingAgent)
        if self.is_tester:
            self.learning_done_callback = collector.set_learning_request_callback(self.on_tester_learning_request)
            collector.add_on_weight_reset_callbacks(self.on_tester_weight_reset)
        else:
            collector.add_on_file_completed_callbacks(self.on_collector_file_completed)
        collector.environment.add_callback(self)
        self.get_thread_locals().collector = collector
        self.get_thread_locals().thread = thread
        self.get_thread_locals().pop_and_run_next(self, wait=not self.collect_before_pre_training)
        collector.start()

    def on_episode_end(self, premature: bool) -> None:
        self.get_thread_locals().pop_and_run_next(self)
        self.local_update_collector_weight()
        if self.is_tester:
            self.local_update_tester_weight()

    def on_environment_finished(self) -> None:
        self.learner_thread.add_to_run_queue(Coordinator.record_environment_completion)

    def record_environment_completion(self):
        self.environment_completion_count += 1

    def local_set_new_weight(self, new_weight: List[tf.Tensor]) -> None:
        self.get_thread_locals().new_weight = new_weight

    def local_set_new_tester_weight(self, new_weight: List[tf.Tensor]) -> None:
        self.get_thread_locals().new_tester_weight = new_weight

    # make these functions with function decorator for coolness :D
    def local_update_collector_weight(self):
        locals = self.get_thread_locals()
        if locals.new_weight is not None:
            locals.collector.update_weights(locals.new_weight)
            print(f'{datetime.now()}: collector {self.get_thread_locals().collector.id} synced weights.')
            locals.new_weight = None

    def local_update_tester_weight(self):
        locals = self.get_thread_locals()
        if locals.new_tester_weight is not None:
            locals.collector.update_weights(locals.new_tester_weight)
            print(f'{datetime.now()}: tester {self.get_thread_locals().collector.id} synced weights with its learner.')
            locals.new_tester_weight = None
            self.call_learning_done_callback()

    def send_to_workers(self, func: Callable, *args) -> None:
        for collector_thread in self.collector_threads:
            collector_thread.add_to_run_queue(func, *args)
        for tester_thread in self.tester_threads:
            tester_thread.add_to_run_queue(func, *args)

    def send_to_tester(self, id: int, func: Callable, *args) -> None:
        self.tester_threads[self.tester_ids.index(id)].add_to_run_queue(func, *args)

    def sync_weights(self):
        if not self.sync_weight:
            return
        print(f'{datetime.now()}: sending weights to workers.')
        self.send_to_workers(Coordinator.local_set_new_weight, self.learner.get_weights())

    def sync_tester_weight(self, id: int) -> None:
        print(f'{datetime.now()}: sending weights to tester {id}.')
        self.send_to_tester(id, Coordinator.local_set_new_tester_weight,
                            self.tester_learners[self.tester_ids.index(id)].get_weights())

    def dummy(self):
        return

    def record_collector_file_completion(self, id: int, version: int) -> None:
        self.file_completions[version].append(True)
        if self.train and len(self.file_completions[version]) == len(self.collector_creators):
            self.learner.learn(version)
            self.sync_weights()

    def tester_learning_batch_end_callback(self, id: int) -> None:
        while True:
            func, args = self.learner_thread.pop_next()
            if func is None:
                return
            if isinstance(func, partial) and func.func is Coordinator.tester_learner_weight_reset and \
                    func.keywords['id'] == id:
                func(self, *args)
                return
            else:
                self.learner_thread_run_queue.append((func, args))

    def reset_tester_learner_weights(self, id: int) -> None:
        tester_index = self.tester_ids.index(id)
        print(f'{datetime.now()}: resetting tester learner weights for {id}.')
        if self.tester_reset_weight_file[tester_index] is None:
            self.tester_learners[tester_index].model.set_weights(self.learner.model.get_weights())
        else:
            self.tester_learners[tester_index].model.load_weights(self.tester_reset_weight_file[tester_index],
                                                                  by_name=True)

    def learn_for_tester(self, id: int, version: List[int], loss_threshold: float) -> None:
        tester_index = self.tester_ids.index(id)
        self.tester_in_learning[tester_index] = True
        self.tester_learners[tester_index].learn(version, loss_threshold,
                                                 partial(self.tester_learning_batch_end_callback, id=id))
        if self.weight_reset_requested[tester_index]:
            self.reset_tester_learner_weights(id)
            self.send_to_tester(id, Coordinator.call_learning_done_callback)
        else:
            self.sync_tester_weight(id)
        self.weight_reset_requested[tester_index] = False

    def call_learning_done_callback(self) -> None:
        self.learning_done_callback()

    def on_collector_file_completed(self, id: int, version: int) -> None:
        self.learner_thread.add_to_run_queue(
            partial(Coordinator.record_collector_file_completion, id=id, version=version))

    def on_tester_learning_request(self, id: int, version: List[int], loss_threshold: float) -> None:
        self.learner_thread.add_to_run_queue(partial(Coordinator.learn_for_tester, id=id,
                                                     version=version, loss_threshold=loss_threshold))

    def on_tester_weight_reset(self, id: int, weight_file: str = None) -> None:
        self.learner_thread.add_to_run_queue(partial(Coordinator.tester_learner_weight_reset,
                                                     id=id, weight_file=weight_file))

    def tester_learner_weight_reset(self, id: int, weight_file: str = None) -> None:
        tester_index = self.tester_ids.index(id)
        self.tester_reset_weight_file[tester_index] = weight_file

        def func():
            self.weight_reset_requested[tester_index] = True

        tester_learner = self.tester_learners[tester_index]
        if self.tester_in_learning[tester_index]:
            print(f'{datetime.now()}: sending stop request in learner {tester_learner.id}.')
            tester_learner.stop_if_learning(func)
        else:
            self.reset_tester_learner_weights(id)

    def start(self):
        self.learner_thread = self.get_main_thread()
        self.collector_threads = [self.create_thread(self.start_collector, c_creator)
                                  for c_creator in self.collector_creators]
        self.tester_threads = [self.create_thread(self.start_collector, t_creator)
                               for t_creator in self.tester_creators]
        [c_thread.run() for c_thread in self.collector_threads]
        [t_thread.run() for t_thread in self.tester_threads]
        self.tester_learners = [learner_creator() for learner_creator in self.tester_learner_creators]
        self.weight_reset_requested = [False] * len(self.tester_learners)
        self.tester_reset_weight_file = [None] * len(self.tester_learners)
        self.tester_in_learning = [False] * len(self.tester_learners)
        self.learner = self.learner_creator()
        if self.pre_training:
            self.learner.learn(list(range(self.collector_version_start)))
        if self.sync_weight:
            self.sync_weights()
        else:
            print(f'{datetime.now()}: sending dummy to workers.')
            self.send_to_workers(Coordinator.dummy)
        while self.environment_completion_count < len(self.collector_creators) + len(self.tester_creators):
            if len(self.learner_thread_run_queue) > 0:
                func, args = self.learner_thread_run_queue.popleft()
                func(self, *args)
            else:
                self.learner_thread.pop_and_run_next(self)
            time.sleep(1)


class ProcessBasedCoordinator(Coordinator):
    def __init__(self, collector_creators: List[Callable[[], DataCollectionAgent]],
                 learner_creator: Callable[[], LearningAgent],
                 tester_creators: List[Union[int, Callable[[], TestingAgent]]],
                 tester_learner_creators: List[Callable[[], Union[None, LearningAgent]]], cfg: Config):
        self.process_configs = cfg['process_configs']

        super().__init__(collector_creators, learner_creator, tester_creators, tester_learner_creators, cfg)

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
