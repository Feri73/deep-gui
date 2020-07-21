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
from parallelism import ThreadLocals, Thread, Process
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
    def __init__(self, id: int, model: keras.Model, learn_model: Optional[keras.Model], example_episode: Episode,
                 create_environment: Callable[['DataCollectionAgent'], Environment],
                 iic_distorter: Optional[Callable], cfg: Config):
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
        shuffle = cfg['shuffle']
        correct_distributions = cfg['correct_distributions']
        augmenting_correction = cfg['augmenting_correction']
        strict_correction = cfg['strict_correction']
        batch_size = cfg['batch_size']
        epochs_per_version = cfg['epochs_per_version']
        meta_save_frequency = max_file_size
        version_start = 0
        save_dir = None
        validation_dir = None

        collection_cfg = {
            'max_episodes': max_episodes,
            'max_file_size': max_file_size if self.learn else 0,
            'meta_save_frequency': meta_save_frequency,
            'file_dir': file_dir,
            'version_start': version_start
        }
        learning_cfg = {
            'file_dir': file_dir,
            'shuffle': shuffle,
            'correct_distributions': correct_distributions,
            'augmenting_correction': augmenting_correction,
            'strict_correction': strict_correction,
            'batch_size': batch_size,
            'epochs_per_version': epochs_per_version,
            'save_dir': save_dir,
            'validation_dir': validation_dir
        }

        super().__init__(id, model, example_episode, create_environment, collection_cfg)

        self.steps = 0
        self.most_recent_weights = None
        self.next_file_valid = True
        self.first_valid_version = 0
        self.past_rewards = deque(maxlen=past_rewards_window)

        self.add_on_file_completed_callbacks(self.on_file_completed)

        if self.learn:
            self.learning_agent = LearningAgent(id, learn_model, iic_distorter, learning_cfg)

    def reset_file(self, new_file: bool = True):
        super().reset_file(new_file)
        self.next_file_valid = True

    def on_file_completed(self, id: int, file_version: int):
        if self.next_file_valid:
            past_rewards_sum = sum(self.past_rewards) / len(self.past_rewards)
            print(f'{datetime.now()}: past rewards sum in tester {self.id} is {past_rewards_sum}.')
            if past_rewards_sum < self.past_rewards_threshold:
                self.learning_agent.learn(list(range(max(file_version + 1 - self.version_window,
                    self.first_valid_version), file_version + 1)), self.loss_threshold)
        else:
            self.first_valid_version = file_version + 1

    def update_weights(self, weights: List[tf.Tensor]):
        super().update_weights(weights)
        if self.learn:
            self.most_recent_weights = weights

    def reset_weights(self):
        if self.steps > 0:
            self.next_file_valid = False
        print(f'{datetime.now()}: resetting weights for tester {self.id} with weights from ', end='')
        self.past_rewards.clear()
        if self.most_recent_weights is not None:
            print('global learner')
            self.update_weights(self.most_recent_weights)
        elif self.weights_file is not None:
            print('file')
            self.model.load_weights(self.weights_file, by_name=True)

    def on_episode_start(self, state: np.ndarray) -> None:
        if self.learn and self.weight_reset_frequency is not None and self.steps % self.weight_reset_frequency == 0:
            self.reset_weights()
        self.steps += 1
        super().on_episode_start(state)

    def on_state_change(self, src_state: np.ndarray, action: Any, dst_state: np.ndarray, reward: float) -> None:
        self.past_rewards.append(reward)
        super().on_state_change(src_state, action, dst_state, reward)


class EarlyStoppingByLossVal(keras.callbacks.Callback):
    def __init__(self, monitor, value):
        super().__init__()
        self.monitor = monitor
        self.value = value

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
        self.save_dir = cfg['save_dir']
        self.validation_dir = cfg['validation_dir']

        self.id = id
        # plot the model (maybe here or where it's created)
        self.model = model
        self.iic_distorter = iic_distorter

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

    # add logs
    def learn(self, version: Union[int, List[int]], loss_threshold: int=None) -> None:
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
                stop_callback = EarlyStoppingByLossVal('loss', loss_threshold)
                callbacks.append(stop_callback)

            self.model.fit(data, validation_data=validation_data, validation_steps=validation_steps,
                           epochs=int(self.epochs_per_version), steps_per_epoch=steps_per_epoch,
                           callbacks=callbacks)
            del data


class Coordinator(ABC, EnvironmentCallbacks):
    def __init__(self, collector_creators: List[Callable[[], DataCollectionAgent]],
                 learner_creator: Callable[[], LearningAgent],
                 tester_creators: List[Callable[[], TestingAgent]], cfg: Config):
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

    def start_collector(self, collector_creator: Callable[[], DataCollectionAgent],
                        is_tester: bool, thread: Thread) -> None:
        collector = collector_creator()
        if not is_tester:
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
        self.collector_threads = [self.create_thread(self.start_collector, c_creator, False)
                                  for c_creator in self.collector_creators]
        self.tester_threads = [self.create_thread(self.start_collector, t_creator, True)
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


