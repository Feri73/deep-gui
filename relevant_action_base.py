import os
import time
from io import BytesIO
from typing import Tuple, Optional, Callable

import matplotlib.cm as cm
import numpy as np
import scipy.misc
import tensorflow as tf
from PIL import Image, ImageDraw

from actor_critic import A2C
from coordinators import MultiCoordinatorCallbacks
from reinforcement_learning import RLCoordinator, Episode
from relevant_action import RelevantActionEnvironment
from screen_parsing import ScreenEncoder, PolicyGenerator, ValueEstimator
from tf1 import TF1RLAgent, LazyGradient
from utils import Config


class PolicyUser:
    def __init__(self, policy_user: Callable):
        self.policy_user = policy_user

    def __call__(self, policy: tf.Tensor) -> tf.Tensor:
        self.policy = policy
        return self.policy_user(policy)


class Agent(TF1RLAgent, MultiCoordinatorCallbacks):
    def __init__(self, id: int, coordinator: Optional[RLCoordinator], optimizer: tf.train.Optimizer,
                 policy_user: Callable[[tf.Tensor], tf.Tensor], action2pos: Callable, action_shape: Tuple[int, int],
                 value: Optional[float], sess: tf.Session, scope: str, input_shape: Tuple[int, ...],
                 trainable: bool, cfg: Config, ):
        self.action2pos = action2pos

        self.screen_new_shape = tuple(cfg['screen_new_shape'])
        self.action_shape = action_shape
        crop_top_left = cfg['crop_top_left']
        crop_size = cfg['crop_size']
        action_type_count = cfg['action_type_count']
        summary_path = cfg['summary_path']
        load_model = cfg['load_model']
        self.log_screen = cfg['log_screen']
        self.log_new_screen = cfg['log_new_screen']
        self.log_policy = cfg['log_policy']
        self.grid_logs = cfg['grid_logs']
        self.save_to_path = cfg['save_to_path']
        save_max_keep = cfg['save_max_keep']
        self.debug_mode = cfg['debug_mode']
        self.steps_per_log = cfg['steps_per_log']
        self.target_updates_per_save = cfg['target_updates_per_save']
        self.local_change_size = cfg['local_change_size']
        self.crop_top_left = cfg['crop_top_left']
        self.crop_size = cfg['crop_size']
        self.contrast_alpha = cfg['contrast_alpha']
        self.padding_type = cfg['padding_type']
        self.value_estimator_use_range = cfg['value_estimator_use_range']
        self.pos_reward = cfg['pos_reward']
        self.neg_reward = cfg['neg_reward']
        conv_kernel_sizes = cfg['conv_kernel_sizes']
        conv_filter_nums = cfg['conv_filter_nums']
        conv_stride_sizes = cfg['conv_stride_sizes']
        conv_maxpool_sizes = cfg['conv_maxpool_sizes']
        deconv_kernel_sizes = cfg['deconv_kernel_sizes']
        deconv_filter_nums = cfg['deconv_filter_nums']
        deconv_output_shapes = cfg['deconv_output_shapes']

        self.target_updates = 0
        self.unlogged_gradients = []

        self.mean_episode_reward = []
        self.mean_loss = []
        self.mean_lengths = []
        self.logs = ()

        self.log_step = 0
        self.last_env_state = None
        self.last_action = None

        action_tensor_shape = (*action_shape, action_type_count)

        with tf.variable_scope(scope):
            screen_encoder = ScreenEncoder(crop_top_left, crop_size, self.screen_new_shape, self.contrast_alpha,
                                           self.padding_type, conv_kernel_sizes, conv_filter_nums,
                                           conv_stride_sizes, conv_maxpool_sizes)
            policy_user = PolicyUser(policy_user)
            rl_model = A2C(screen_encoder, PolicyGenerator(action_tensor_shape, self.padding_type, deconv_kernel_sizes,
                                                           deconv_filter_nums, deconv_output_shapes),
                           ValueEstimator(value, (self.neg_reward, self.pos_reward)
                                          if self.value_estimator_use_range else None, self.padding_type),
                           policy_user, action_tensor_shape, cfg)

        self.summary_writer = tf.summary.FileWriter(f'{summary_path}/{scope}')

        super(Agent, self).__init__(id, rl_model, coordinator, scope, sess, input_shape, optimizer, cfg, trainable)

        if trainable:
            self.output_logs_e += (tf.linalg.global_norm(self.trainable_weights),
                                   # this re-clipping is a problem when inherent is true
                                   tf.linalg.global_norm(self.clip_gradient(self.output_gradients)),
                                   tf.reduce_max(policy_user.policy))

            self.log_names = ('Loss/Policy Loss', 'Loss/Value Loss', 'Loss/Entropy', 'Loss/Advantage',
                              'Model/Weights', 'Loss/Gradients', 'Loss/Policy Max')

            if self.log_new_screen:
                self.new_screen_log_index = len(self.output_logs_e)
                self.output_logs_e += (screen_encoder.get_processed_screen(),)
            if self.log_policy:
                self.policy_log_index = len(self.output_logs_e)
                self.output_logs_e += (policy_user.policy,)
        # add mechanism so that i can have flexible logging and therefore can have policy log for untrainable

        # this should not be only trainable weights
        self.saver = tf.train.Saver(var_list=self.trainable_weights, max_to_keep=save_max_keep)
        if load_model:
            # test this
            # also, this does not work for optimizer parameters because they will be re-initialized. do sth about it
            self.saver.restore(sess, tf.train.get_checkpoint_state(
                self.save_to_path.split('/', 1)[0]).model_checkpoint_path)

    def set_generated_gradient_target(self, target: 'TF1RLAgent'):
        super().set_generated_gradient_target(target)
        if self.debug_mode:
            self.op_apply_dynamic_gradients = [self.op_apply_dynamic_gradients,
                                               tf.print(tf.timestamp(), f'dynamic gradient apply from #{self.id}')]
            self.op_apply_static_gradients = [self.op_apply_static_gradients,
                                              tf.print(tf.timestamp(), f'static gradient apply from #{self.id}')]

    # when in this function there is an error, it only throws the error for the first time
    #   (but it does not add any summary anyway). why?
    def on_episode_gradient_computed(self, episode: Episode, gradient: LazyGradient) -> None:
        self.unlogged_gradients += [(gradient, episode, self.log_step)]
        self.log_step += 1
        for gradient_i, (gradient, episode, step) in enumerate([g for g in self.unlogged_gradients]):
            if not gradient.has_logs():
                break
            self.unlogged_gradients.remove((gradient, episode, step))
            # assumption is that batch size is 1 in episode
            self.mean_episode_reward += [sum(map(lambda x: x[0], episode.rewards_tb[1:]))]
            self.mean_loss += [gradient.loss]
            self.mean_lengths += [len(episode) - 1]
            self.logs = tuple(map(lambda x: [x], gradient.logs_e)) if len(self.logs) == 0 else \
                tuple(map(lambda x, y: x + [y], self.logs, gradient.logs_e))
            summary = tf.Summary()
            if step % self.steps_per_log == 0:
                # use callbacks here
                summary.value.add(tag='Episode/Mean Episode Reward', simple_value=np.mean(self.mean_episode_reward))
                summary.value.add(tag='Episode/Lengths', simple_value=np.mean(self.mean_lengths))
                summary.value.add(tag='Loss/Mean Loss', simple_value=np.mean(self.mean_loss))
                for log_name, log_val in zip(self.log_names, self.logs):
                    summary.value.add(tag=f'{log_name}', simple_value=np.mean(log_val))
                self.mean_episode_reward = []
                self.mean_loss = []
                self.mean_lengths = []
                self.logs = ()

            if self.last_env_state is not None:
                global_diff = np.linalg.norm(RelevantActionEnvironment.crop_state(self, episode.states_tb[0][0])
                                             - RelevantActionEnvironment.crop_state(self, self.last_env_state))
                local_diff = np.linalg.norm(
                    RelevantActionEnvironment.
                    crop_to_local(self, RelevantActionEnvironment.crop_state(self, episode.states_tb[0][0]),
                                  self.last_action)
                    - RelevantActionEnvironment.
                    crop_to_local(self, RelevantActionEnvironment.crop_state(self, self.last_env_state),
                                  self.last_action))
            # measure how much time these logs take, if too much, justt disable them for
            #   everything except the test agent (the final agent)
            if self.log_screen:
                env_state = episode.states_tb[0][0] * 255.0
                action = self.action2pos(episode.actions_tb[1][0], original=True)
                if self.grid_logs:
                    env_state = grid_image(env_state, env_state.shape[:-1] // np.array(self.action_shape), (0, 0, 0))
                env_state = show_local_border(env_state, (self.local_change_size,) * 2, action)
                env_state[max(0, action[1] - 5):action[1] + 5, max(0, action[0] - 5):action[0] + 5] = [255, 0, 0]
                if self.last_env_state is not None:
                    env_state = self.add_diff_texts(env_state, global_diff, local_diff)
                summary.value.add(tag='episodes', image=get_image_summary(env_state))
            if self.log_new_screen:
                env_state = gradient.logs_e[self.new_screen_log_index][0, 0] * 255.0
                if env_state.shape[-1] == 1:
                    env_state = np.concatenate([env_state] * 3, axis=-1)
                action = self.action2pos(episode.actions_tb[1][0], original=False)
                if self.grid_logs:
                    env_state = grid_image(env_state, np.array(self.screen_new_shape) // np.array(self.action_shape),
                                           (0, 0, 0))
                env_state = show_local_border(env_state,
                                              (self.local_change_size * self.screen_new_shape[0] //
                                               episode.states_tb[0][0].shape[0],
                                               self.local_change_size * self.screen_new_shape[1] //
                                               episode.states_tb[0][0].shape[1]), action)
                env_state[action[1], action[0], :] = [255, 0, 0]
                if self.last_env_state is not None:
                    env_state = self.add_diff_texts(env_state, global_diff, local_diff)
                summary.value.add(tag='processed episode', image=get_image_summary(env_state))
            # this is only click policy. make it more general (IDK how tho :|)
            if self.log_policy:
                policy = np.reshape(gradient.logs_e[self.policy_log_index][0, 0], self.action_shape)
                policy = cm.viridis(policy)[:, :, :3] * 255.0
                summary.value.add(tag='policy', image=get_image_summary(policy))
            # i can also visualize layers of CNN

            self.summary_writer.add_summary(summary, step)
            self.summary_writer.flush()

            self.last_env_state = episode.states_tb[0][0]
            self.last_action = self.action2pos(episode.actions_tb[1][0], original=True)

    def add_diff_texts(self, image: np.ndarray, global_diff: float, local_diff: float) -> np.ndarray:
        image = image.copy()
        image = Image.fromarray(np.uint8(image))
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), f'global: {global_diff}', (255, 0, 0))
        draw.text((0, 16), f'local: {local_diff}', (0, 255, 0))
        return np.array(image)

    def log_episode(self, episode: Episode) -> None:
        self.log_step += 1
        self.mean_episode_reward += [sum(map(lambda x: x[0], episode.rewards_tb[1:]))]
        summary = tf.Summary()
        if self.log_step % self.steps_per_log == 0:
            # use callbacks here
            summary.value.add(tag='Episode/Mean Episode Reward', simple_value=np.mean(self.mean_episode_reward))
            self.mean_episode_reward = []
        env_state = episode.states_tb[0][0] * 255.0
        action = self.action2pos(episode.actions_tb[1][0], original=True)
        if self.grid_logs:
            env_state = grid_image(env_state, env_state.shape[:-1] // np.array(self.action_shape), (0, 0, 0))
        env_state[action[1], action[0], :] = [255, 0, 0]
        summary.value.add(tag='episodes', image=get_image_summary(env_state))
        self.summary_writer.add_summary(summary, self.log_step)
        self.summary_writer.flush()

    def on_update_target(self, learner_id: int) -> None:
        self.target_updates += 1
        if self.target_updates % self.target_updates_per_save == 0:
            while True:
                try:
                    self.saver.save(self.session, self.save_to_path)
                    break
                except Exception:
                    time.sleep(1)

    def on_episode_end(self, premature: bool) -> None:
        if self.debug_mode:
            print(f'episode ended in #{self.id}')
        if not self.trainable:
            self.log_episode(self.episode.value)
        super(type(self), self).on_episode_end(premature)

    def on_episode_start(self, env_state) -> None:
        if self.debug_mode:
            print(f'episode started in #{self.id}')
        super(type(self), self).on_episode_start(env_state)


def get_image_summary(image: np.ndarray) -> tf.Summary.Image:
    bio = BytesIO()
    scipy.misc.toimage(image).save(bio, format="png")
    res = tf.Summary.Image(encoded_image_string=bio.getvalue(), height=image.shape[0], width=image.shape[1])
    bio.close()
    return res


def policy2pos(action: np.ndarray, screen_new_shape: Tuple[int, ...],
               crop_size: Tuple[int, int], crop_top_left: Tuple[int, int],
               action_shape: Tuple[int, int], original=True) -> Tuple[int, int, int]:
    action = int(action)
    type = action // np.prod(action_shape)
    y = int(((action // action_shape[1]) % action_shape[0] + .5) * (screen_new_shape[0] // action_shape[0]))
    x = int((action % action_shape[1] + .5) * (screen_new_shape[1] // action_shape[1]))
    if original:
        y, x = tuple((np.array(crop_size) / np.array(screen_new_shape) *
                      np.array([y, x]) + np.array(crop_top_left)).astype(np.int32))
    return x, y, type


# this works incorrectly for original screen because it does not take crop into account
def grid_image(image: np.ndarray, grid_size: Tuple[int, int], color: Tuple[int, int, int]) -> np.ndarray:
    image = image.copy()
    for i in range(0, image.shape[0], grid_size[0]):
        image[i] = color
    for j in range(0, image.shape[1], grid_size[1]):
        image[:, j] = color
    return image


def show_local_border(image: np.ndarray, local_size: Tuple[int, int], action: Tuple[int, int, int]) -> np.ndarray:
    image = image.copy()
    image[max(0, action[1] - local_size[1] // 2):action[1] + local_size[1] // 2,
    max(0, action[0] - local_size[0] // 2)] = [0, 255, 0]
    image[max(0, action[1] - local_size[1] // 2):action[1] + local_size[1] // 2,
    min(image.shape[1] - 1, action[0] + local_size[0] // 2)] = [0, 255, 0]
    image[max(0, action[1] - local_size[1] // 2),
    max(0, action[0] - local_size[0] // 2):action[0] + local_size[0] // 2] = [0, 255, 0]
    image[min(image.shape[0] - 1, action[1] + local_size[1] // 2),
    max(0, action[0] - local_size[0] // 2):action[0] + local_size[0] // 2] = [0, 255, 0]
    return image


def most_probable_weighted_policy_user(policy: tf.Tensor) -> tf.Tensor:
    return tf.argmax(tf.distributions.Multinomial(1.0, probs=policy).sample(), axis=-1)


def least_probable_weighted_policy_user(policy: tf.Tensor) -> tf.Tensor:
    probs = 1 / policy
    probs = probs / tf.reduce_sum(probs, axis=-1)
    return tf.argmax(tf.distributions.Multinomial(1.0, probs=probs).sample(), axis=-1)


def least_certain_weighted_policy_user(policy: tf.Tensor) -> tf.Tensor:
    probs = 1 / tf.abs(.5 - policy)
    probs = probs / tf.reduce_sum(probs, axis=-1)
    return tf.argmax(tf.distributions.Multinomial(1.0, probs=probs).sample(), axis=-1)


# for using this, i have to change the v based on whether the most certain is >.5 or <.5
def most_certain_weighted_policy_user(policy: tf.Tensor) -> tf.Tensor:
    probs = tf.abs(.5 - policy)
    probs = probs / tf.reduce_sum(probs, axis=-1)
    return tf.argmax(tf.distributions.Multinomial(1.0, probs=probs).sample(), axis=-1)


def random_policy_user(policy: tf.Tensor) -> tf.Tensor:
    return tf.argmax(
        tf.distributions.Multinomial(1.0, probs=tf.ones_like(policy) /
                                                tf.cast(tf.shape(policy)[-1], tf.float32)).sample(), axis=-1)


def remove_summaries(summary_path: str, reset_summary: bool) -> None:
    while reset_summary and os.path.isdir(summary_path) and len(os.listdir(summary_path)) > 0 and \
            len(os.listdir(f'{summary_path}/{os.listdir(summary_path)[0]}')) > 0:
        for agent_dir in os.listdir(summary_path):
            try:
                for f in os.listdir(f'{summary_path}/{agent_dir}'):
                    os.unlink(f'{summary_path}/{agent_dir}/{f}')
            except FileNotFoundError:
                pass
