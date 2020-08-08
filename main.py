import os
import copy
import random
from datetime import datetime
from functools import partial
from io import BytesIO
from typing import Callable, List, Any, Tuple, Union, Dict

import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import yaml
from PIL import Image

import readouts
from environment import EnvironmentCallbacks, Environment
from phone import DummyPhone, Phone
# noinspection PyUnresolvedReferences
from predictors import ScreenPreprocessor, SimpleRewardPredictor, UNetRewardPredictor, RandomRewardPredictor
from readouts import PredictionClusterer, better_reward_to_action, worse_reward_to_action, \
    most_certain_reward_to_action, least_certain_reward_to_action, random_reward_to_action
from relevant_action import RelevantActionEnvironment
from relevant_action_monkey_client import RelevantActionMonkeyClient
from single_state_categorical_reward import DataCollectionAgent, LearningAgent, Episode, TestingAgent, \
    ProcessBasedCoordinator
from tf_utils import BufferLogger
from utils import Config


class CollectorLogger(EnvironmentCallbacks):
    def __init__(self, name: str, preprocessed_screen: tf.Tensor, reward_prediction: tf.Tensor,
                 preds_clusterer: PredictionClusterer, action_for_screen: Callable,
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
            print(f'{datetime.now()}: Changed log file to chunk {chunk} for {self.name}. '
                  f'Current app is {self.environment.get_current_app()}')
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
                pred = pred.resize((new_size[1], new_size[0]))
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


def linear_normalizer(logits: tf.Tensor, axis=None) -> tf.Tensor:
    return logits / (tf.reduce_sum(logits, axis=axis) + keras.backend.epsilon())


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


def get_mask(transformed_shape: np.ndarray, mask_top_left: np.ndarray, mask_bottom_right: np.ndarray,
             transform_coord: Callable) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    top_left = transform_coord(mask_top_left)
    bottom_right = transform_coord(mask_bottom_right)
    tl_diff = np.maximum(0, top_left) - top_left
    br_diff = bottom_right - np.minimum(transformed_shape - 1, bottom_right)
    return top_left, bottom_right, tl_diff, br_diff


def distort_episode_shift(episode: Episode, mask_shape: tuple, shift_max_value: np.ndarray,
                          action_pos_to_screen_pos: Callable, screen_pos_to_action_pos: Callable) \
        -> Tuple[Episode, np.ndarray, np.ndarray]:
    mask_shape = np.array(mask_shape)
    state_img_shape = np.array(episode.state.shape[:2])
    state2 = np.zeros(episode.state.shape, episode.state.dtype)
    shift_neg_max = np.maximum(-episode.action[:2], -shift_max_value)
    shift_pos_max = np.minimum(state_img_shape - 1 - episode.action[:2], shift_max_value)

    screen_pos = action_pos_to_screen_pos(episode.action[:2])

    while True:
        action2 = episode.action.copy()
        shift_val = np.array([np.random.randint(shift_neg_max[0], shift_pos_max[0]),
                              np.random.randint(shift_neg_max[1], shift_pos_max[1])])
        screen_pos2 = screen_pos + shift_val
        action2[:2] = screen_pos_to_action_pos(screen_pos2)
        if np.any(action2[:2] < 0) or np.any(action2[:2] >= mask_shape[:2]):
            continue
        mask_top_left = np.maximum(0, -shift_val)
        mask2_top_left = np.maximum(0, shift_val)
        mask_bottom_right = np.minimum(state_img_shape, state_img_shape - shift_val)
        mask2_bottom_right = np.minimum(state_img_shape, state_img_shape + shift_val)
        state2[mask2_top_left[0]:mask2_bottom_right[0], mask2_top_left[1]:mask2_bottom_right[1]] = \
            episode.state[mask_top_left[0]:mask_bottom_right[0], mask_top_left[1]:mask_bottom_right[1]]

        top_left, bottom_right, tl_diff, br_diff = \
            get_mask(mask_shape[:2], mask_top_left, mask_bottom_right, screen_pos_to_action_pos)
        top_left2, bottom_right2, tl_diff2, br_diff2 = \
            get_mask(mask_shape[:2], mask2_top_left, mask2_bottom_right, screen_pos_to_action_pos)
        top_left = np.ceil(top_left + np.maximum(tl_diff, tl_diff2)).astype(np.int32)
        top_left2 = np.ceil(top_left2 + np.maximum(tl_diff, tl_diff2)).astype(np.int32)
        bottom_right = np.ceil(bottom_right - np.maximum(br_diff, br_diff2)).astype(np.int32)
        bottom_right2 = np.ceil(bottom_right2 - np.maximum(br_diff, br_diff2)).astype(np.int32)

        diff = bottom_right - top_left
        diff2 = bottom_right2 - top_left2
        bottom_right = top_left + np.minimum(diff, diff2)
        bottom_right2 = top_left2 + np.minimum(diff, diff2)

        mask = np.zeros(mask_shape)
        mask[max(0, top_left[0]): bottom_right[0], max(0, top_left[1]): bottom_right[1]] = 1
        mask2 = np.zeros(mask_shape)
        mask2[max(0, top_left2[0]): bottom_right2[0], max(0, top_left2[1]): bottom_right2[1]] = 1

        assert np.sum(mask) == np.sum(mask2)
        # note that I am not distorting the result here. If I use it I have to distort it too.
        return Episode(state2, action2, episode.reward.copy(), episode.result), mask, mask2


# maybe log some of these
def distort_episode_color(episode: Episode, mask_shape: tuple) -> Tuple[Episode, np.ndarray, np.ndarray]:
    color_order = np.arange(3)
    while np.all(color_order == np.arange(3)):
        color_order = np.random.permutation(3)
    state = episode.state.copy()[:, :, color_order]
    mask = np.ones(mask_shape)
    # note that I am not distorting the result here. If I use it I have to distort it too.
    return Episode(state, episode.action.copy(), episode.reward.copy(), episode.result), mask, mask


def combine_distort_episode(func_probs: List[Tuple[Callable, float]], mask_shape: tuple) -> Callable:
    def distort(episode: Episode) -> Episode:
        for func, p in func_probs:
            if random.uniform(0, 1) < p:
                episode = func(episode, mask_shape)
        return episode

    return distort


def control_dependencies(inputs) -> tf.Tensor:
    x, dependencies = inputs
    with tf.control_dependencies(dependencies):
        x = tf.identity(x)
    return x


def linear_combination(funcs: List[Callable], coeffs: List[float] = None) -> Callable:
    return lambda *inps: sum([c * f(*inps) for c, f in zip(coeffs, funcs)])


def preds_variance_regularizer(preds: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(tf.math.reduce_variance(preds, axis=[1, 2, 3]))


# if I use anything other than spawn, this might stop working because I am changing the same dictionary reference
#   in all agents
def create_agent(id: int, agent_num: int, agent_name: str, is_learner: bool, is_tester: bool,
                 agent_option_probs: List[float], agent_clusterer_cfg_name: str,
                 weights_file: str) -> Union[DataCollectionAgent, LearningAgent]:
    cfg = copy.deepcopy(globals()['cfg'])
    environment_configs = cfg['environment_configs']
    learner_configs = cfg['learner_configs']
    collector_configs = cfg['collector_configs']
    tester_configs = cfg['tester_configs']
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
    reward_predictor = cfg['reward_predictor']
    prediction_shape = cfg['prediction_shape']
    variance_reg_coeff = cfg['variance_reg_coeff']
    l1_reg_coeff = cfg['l1_reg_coeff']
    iic_coeff = cfg['iic_coeff']
    iic_distorter_probabilities = cfg['iic_distorter_probabilities']
    distort_shift_max_value = cfg['distort_shift_max_value']
    use_logger = cfg['use_logger']
    screen_shape = phone_configs['screen_shape']
    adb_path = phone_configs['adb_path']
    scroll_min_value = phone_configs['scroll_min_value']
    scroll_max_value = phone_configs['scroll_max_value']
    scroll_event_count = phone_configs['scroll_event_count']
    action_type_count = environment_configs['action_type_count']
    steps_per_app = environment_configs['steps_per_app']
    screenshots_interval = environment_configs['screenshots_interval']
    global_equality_threshold = environment_configs['global_equality_threshold']
    calculate_reward = environment_configs['calculate_reward']
    screen_preprocessor_resize_size = screen_preprocessor_configs['resize_size']
    screen_preprocessor_crop_top_left = screen_preprocessor_configs['crop_top_left']
    screen_preprocessor_crop_size = screen_preprocessor_configs['crop_size']
    reward_predictor_configs = cfg[f'{reward_predictor[1]}_reward_predictor_configs']
    learn_in_tester = tester_configs['learn']
    learning_rate = tester_configs['learning_rate']

    environment_configs['pos_reward'] = pos_reward
    environment_configs['neg_reward'] = neg_reward
    environment_configs['steps_per_episode'] = 1
    environment_configs['crop_top_left'] = screen_preprocessor_crop_top_left
    environment_configs['crop_size'] = screen_preprocessor_crop_size
    if is_tester and learn_in_tester:
        environment_configs['calculate_reward'] = True
        calculate_reward = True
    phone_configs['crop_top_left'] = screen_preprocessor_crop_top_left
    phone_configs['crop_size'] = screen_preprocessor_crop_size
    phone_configs['apks_path'] = testers_apks_path if is_tester else collectors_apks_path
    phone_configs['clone_script_path'] = testers_clone_script if is_tester else collectors_clone_script
    collector_configs['file_dir'] = data_file_dir
    learner_configs['file_dir'] = data_file_dir
    tester_configs['weights_file'] = weights_file
    tester_configs['file_dir'] = tester_configs['file_dir'] + '/tester' + str(id)
    tester_learner_configs = tester_configs['learner_configs']
    batch_size = 1 if not is_learner else tester_learner_configs['batch_size'] \
        if is_tester else learner_configs['batch_size']
    tester_learner_configs['file_dir'] = tester_configs['file_dir']
    tester_learner_configs['shuffle'] = True
    tester_learner_configs['save_dir'] = None
    tester_learner_configs['validation_dir'] = None
    tester_learner_configs['data_portion_per_epoch'] = 1
    if is_tester and monkey_client_mode:
        tester_configs['weight_reset_frequency'] = None
    collector_logger_configs['dir'] = logs_dir
    collector_logger_configs['steps_per_app'] = steps_per_app
    reward_predictor_configs['prediction_shape'] = prediction_shape
    monkey_client_configs = {'adb_path': adb_path, 'scroll_min_value': scroll_min_value,
                             'scroll_max_value': scroll_max_value, 'scroll_event_count': scroll_event_count,
                             'crop_top_left': screen_preprocessor_crop_top_left,
                             'crop_size': screen_preprocessor_crop_size, 'pos_reward': pos_reward,
                             'neg_reward': neg_reward, 'screenshots_interval': screenshots_interval,
                             'global_equality_threshold': global_equality_threshold,
                             'calculate_reward': calculate_reward, 'screen_shape': screen_shape}

    screen_preprocessor_resize_size_a = np.array(screen_preprocessor_resize_size)
    screen_preprocessor_crop_top_left_a = np.array(screen_preprocessor_crop_top_left)
    screen_preprocessor_crop_size_a = np.array(screen_preprocessor_crop_size)
    prediction_shape_a = np.array(prediction_shape)

    def to_preprocessed_coord(p):
        return transform_linearly(p - screen_preprocessor_crop_top_left_a,
                                  screen_preprocessor_resize_size_a /
                                  screen_preprocessor_crop_size_a, np.array([0, 0]))

    def screen_pos_to_action_pos(p: np.ndarray) -> np.ndarray:
        return transform_linearly(to_preprocessed_coord(p),
                                  prediction_shape_a / screen_preprocessor_resize_size_a,
                                  np.array([0, 0]))

    def action_pos_to_screen_pos(action_p: np.ndarray, dtype=None) -> np.ndarray:
        return transform_linearly(action_p, screen_preprocessor_crop_size_a / prediction_shape_a,
                                  screen_preprocessor_crop_top_left_a, dtype)

    example_episode = Episode(np.zeros((*screen_shape, 3), np.uint8), np.zeros(3, np.int32),
                              np.zeros((), np.bool), np.zeros((*screen_shape, 3), np.uint8))

    screen_preprocessor = ScreenPreprocessor(screen_preprocessor_configs, name='screen_preprocessor')

    regs, coeffs = [], []
    if is_learner:
        if variance_reg_coeff != 0:
            regs.append(preds_variance_regularizer)
            coeffs.append(-variance_reg_coeff)
        if l1_reg_coeff != 0:
            regs.append(keras.regularizers.l1(l1_reg_coeff))
            coeffs.append(1)

    reward_predictor = eval(reward_predictor[0])(action_type_count, 2,
                                                 reward_predictor_configs, name='reward_predictor',
                                                 activity_regularizer=None if len(regs) == 0
                                                 else linear_combination(regs, coeffs))

    screen_input = keras.layers.Input(example_episode.state.shape, batch_size, name='state',
                                      dtype=example_episode.state.dtype)
    predictions = reward_predictor(screen_preprocessor(screen_input))

    if is_learner:
        action_sampler = keras.layers.Lambda(lambda elems: prediction_sampler(elems[0], elems[1]),
                                             name='action_sampler')
        action_input = keras.layers.Input(example_episode.action.shape, batch_size, name='action',
                                          dtype=example_episode.action.dtype)
        reward = action_sampler((predictions, action_input))
        learn_model_input = (screen_input, action_input)

        if iic_coeff != 0:
            screen_input2 = keras.layers.Input(example_episode.state.shape, batch_size, name='state2',
                                               dtype=example_episode.state.dtype)
            predictions2 = reward_predictor(screen_preprocessor(screen_input2))
            mask = keras.layers.Input(predictions.shape[1:], batch_size, name='iic_mask', dtype=tf.float32)
            mask2 = keras.layers.Input(predictions2.shape[1:], batch_size, name='iic_mask2', dtype=tf.float32)

            def iic_loss(both_preds):
                preds1 = both_preds[:, :, :, :, 0]
                preds2 = both_preds[:, :, :, :, 1]
                shape1 = np.prod(preds1.get_shape().as_list()[1:])
                shape2 = np.prod(preds2.get_shape().as_list()[1:])
                return iic_coeff * tf.reduce_sum(
                    keras.losses.binary_crossentropy(tf.reshape(preds1 * mask, (-1, shape1)),
                                                     tf.reshape(preds2 * mask2, (-1, shape2))))

            iic_layer = keras.layers.Lambda(
                lambda both_preds: tf.concat([tf.expand_dims(x, axis=-1) for x in both_preds], axis=-1),
                activity_regularizer=iic_loss)
            both_preds = iic_layer((predictions, predictions2))

            predictions2 = keras.layers.Lambda(control_dependencies)((predictions2, [both_preds]))

            action_input2 = keras.layers.Input(example_episode.action.shape, batch_size, name='action2',
                                               dtype=example_episode.action.dtype)
            reward2 = action_sampler((predictions2, action_input2))
            learn_model_input = (*learn_model_input, screen_input2, action_input2, mask, mask2)
            learn_model_output = (reward, reward2)
        else:
            learn_model_output = reward

        learn_model = keras.Model(inputs=learn_model_input, outputs=learn_model_output)
    else:
        built_prediction_to_action_options = [prediction_to_action_options[0](agent_clusterer_cfg_name)] + \
                                             prediction_to_action_options[1:]
        action = keras.layers.Lambda(
            combine_prediction_to_actions(built_prediction_to_action_options, agent_option_probs),
            name='action_predictor')(predictions)

        if use_logger:
            logger = CollectorLogger(f'{agent_name}_{"tester" if is_tester else "collector"}{id}',
                                     screen_preprocessor.output, reward_predictor.output,
                                     built_prediction_to_action_options[0], action_pos_to_screen_pos,
                                     to_preprocessed_coord, collector_logger_configs)

            action = keras.layers.Lambda(control_dependencies,
                                         name='log_dependency_controller')((action, logger.get_dependencies()))

        input = screen_input
        output = action
        model = keras.Model(inputs=input, outputs=output)

    if weights_file is not None:
        if is_learner:
            learn_model.load_weights(weights_file, by_name=True)
        else:
            model.load_weights(weights_file, by_name=True)
    if is_learner:
        if is_tester:
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer = 'adam'
        learn_model.compile(optimizer, keras.losses.BinaryCrossentropy())

    def action2pos(action: np.ndarray) -> Tuple[int, int, int]:
        pos = action_pos_to_screen_pos(action[:2], np.int32)
        return pos[1], pos[0], action[2]

    tester_agent_ref = []

    def create_environment(collector: DataCollectionAgent) -> Environment:
        if monkey_client_mode:
            env = RelevantActionMonkeyClient(collector, action2pos, 3000 + agent_num, 5554 + 2 * agent_num,
                                             5000 + agent_num if is_tester else None,
                                             lambda d: tester_agent_ref[0].reset_weights() if d == b'rw\n' else None,
                                             monkey_client_configs)
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
        iic_distorter = combine_distort_episode(
            list(zip([distort_episode_color, partial(distort_episode_shift,
                                                     shift_max_value=distort_shift_max_value,
                                                     action_pos_to_screen_pos=action_pos_to_screen_pos,
                                                     screen_pos_to_action_pos=screen_pos_to_action_pos)],
                     iic_distorter_probabilities)), tuple([int(x) for x in predictions.shape[1:]]))
        iic_distorter = None if iic_coeff == 0 else iic_distorter

    if is_learner:
        return LearningAgent(id, learn_model, iic_distorter, tester_learner_configs if is_tester else learner_configs)
    elif is_tester:
        agent = TestingAgent(id, model, example_episode, create_environment, tester_configs)
        if monkey_client_mode:
            tester_agent_ref.append(agent)
        return agent
    else:
        return DataCollectionAgent(id, model, example_episode, create_environment, collector_configs)


def parse_specs_to_probs_and_ops(specs: Dict, max_len: int) -> List:
    return sum([[([spec[1][i] if len(spec[1]) > i else (1 - sum(spec[1])) / (max_len - len(spec[1]))
                   for i in range(max_len)], *spec[2:])] * spec[0]
                for spec in specs], [])


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

readouts.prediction_normalizer = None if prediction_normalizer_name is None else eval(prediction_normalizer_name)
readouts.action_prob_coeffs = action_prob_coeffs

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
    tester_creators = list(zip(range(len(tester_option_probs_and_ops)),
                               [partial(create_agent, i, i + len(collector_creators),
                                        probs_and_ops[1] if len(probs_and_ops) > 1 else '', False, True,
                                        probs_and_ops[0], probs_and_ops[2] if len(probs_and_ops) > 2 else None,
                                        weights_file[probs_and_ops[3]] if len(probs_and_ops) > 3 else None)
                                for i, probs_and_ops in enumerate(tester_option_probs_and_ops)]))
    tester_learner_creators = [partial(create_agent, i, i + len(collector_creators) + len(tester_creators),
                                       (probs_and_ops[1] if len(probs_and_ops) > 1 else '') + '_learner', True, True,
                                       None, None,
                                       weights_file[probs_and_ops[3]] if len(probs_and_ops) > 3 else None)
                               for i, probs_and_ops in enumerate(tester_option_probs_and_ops)]
    learner_creator = partial(create_agent, *2 * (len(collector_creators) + 2 * len(tester_creators),), 'learner',
                              True, False, None, None, weights_file['learner'])
    coord = ProcessBasedCoordinator(collector_creators, learner_creator, tester_creators,
                                    tester_learner_creators, cfg['coordinator_configs'])
    coord.start()
