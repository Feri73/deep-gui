import contextlib
from typing import Optional, Tuple, Callable

import numpy as np
import random
from datetime import datetime
from functools import partial

import tensorflow as tf
import yaml

import phone
from coordinators import UnSyncedMultiprocessRLCoordinator, MultithreadRLCoordinator, MultiCoordinatorCallbacks
from reinforcement_learning import RLCoordinator
from relevant_action import RelevantActionEnvironment
import relevant_action_base as base
from screen_parsing import ScreenEncoder, PolicyGenerator
from single_step_binary_reward_dqn import SSBRDQN
from tf1 import TF1RLAgent
from utils import Config


class LogCallbacks(MultiCoordinatorCallbacks):
    def log(self, log: str) -> None:
        print(log)

    def on_update_learner(self, learner_id: int) -> None:
        print(f'{datetime.now()}: learner #{learner_id} got updated.')

    def on_update_target(self, learner_id: int) -> None:
        print(f'{datetime.now()}: target updated by #{learner_id}.')


class Agent(base.Agent):
    def __init__(self, id: int, coordinator: Optional[RLCoordinator], optimizer: tf.train.Optimizer,
                 q_user: Callable[[tf.Tensor], tf.Tensor], action2pos: Callable, action_shape: Tuple[int, int],
                 sess: tf.Session, scope: str, input_shape: Tuple[int, ...], trainable: bool, cfg: Config):
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
            q_generator = PolicyGenerator(action_tensor_shape, self.padding_type, deconv_kernel_sizes,
                                          deconv_filter_nums, deconv_output_shapes, tf.nn.sigmoid)
            q_user = base.PolicyUser(q_user)
            rl_model = SSBRDQN(lambda x: q_generator(screen_encoder(
                (tf.expand_dims(x, axis=1), None, None, None)))[0][:, 0], q_user, action_tensor_shape, cfg)

        self.summary_writer = tf.summary.FileWriter(f'{summary_path}/{scope}')

        TF1RLAgent.__init__(self, id, rl_model, coordinator, scope, sess, input_shape, optimizer, cfg, trainable)

        if trainable:
            self.output_logs_e += (tf.linalg.global_norm(self.trainable_weights),
                                   tf.linalg.global_norm(self.clip_gradient(self.output_gradients)),
                                   tf.reduce_mean(q_user.policy),
                                   tf.math.reduce_std(q_user.policy))

            self.log_names = ('Loss/Loss', 'Loss/High Reward Weight', 'Loss/High Reward Current Size',
                              'Loss/High Reward Previous', 'Model/Weights', 'Loss/Gradients',
                              'Model/Q Mean', 'Model/Q Std')

            if self.log_new_screen:
                self.new_screen_log_index = len(self.output_logs_e)
                self.output_logs_e += (screen_encoder.get_processed_screen(),)
            if self.log_policy:
                self.policy_log_index = len(self.output_logs_e)
                self.output_logs_e += (q_user.policy,)

        self.saver = tf.train.Saver(var_list=self.trainable_weights, max_to_keep=save_max_keep)
        if load_model:
            self.saver.restore(sess, tf.train.get_checkpoint_state(
                self.save_to_path.split('/', 1)[0]).model_checkpoint_path)

    def on_episode_end(self, premature: bool) -> None:
        if self.debug_mode:
            print(f'episode ended in #{self.id}')
        if not self.trainable:
            self.log_episode(self.episode.value)
        super(base.Agent, self).on_episode_end(premature)

    def on_episode_start(self, env_state) -> None:
        if self.debug_mode:
            print(f'episode started in #{self.id}')
        super(base.Agent, self).on_episode_start(env_state)

def most_probable_weighted_q_user(q: tf.Tensor) -> tf.Tensor:
    return base.most_probable_weighted_policy_user(tf.nn.softmax(q))


def least_probable_weighted_q_user(q: tf.Tensor) -> tf.Tensor:
    return base.most_probable_weighted_policy_user(tf.nn.softmax(1 - q))


def least_certain_weighted_q_user(q: tf.Tensor) -> tf.Tensor:
    return base.most_probable_weighted_policy_user(tf.nn.softmax(.5 - tf.abs(.5 - q)))


def most_certain_weighted_q_user(q: tf.Tensor) -> tf.Tensor:
    return base.most_probable_weighted_policy_user(tf.nn.softmax(tf.abs(.5 - q)))


def random_q_user(q: tf.Tensor) -> tf.Tensor:
    return tf.argmax(
        tf.distributions.Multinomial(1.0, probs=tf.ones_like(q) /
                                                tf.cast(tf.shape(q)[-1], tf.float32)).sample(), axis=-1)


log_callbacks = LogCallbacks()


def create_agent(agent_id, is_target, coord):
    session = sess or tf.Session().__enter__()

    q_user = q_users[agent_id % len(q_users)]
    optimizer = random.choice(optimizers)() if learning_rate is None \
        else random.choice(optimizers)(learning_rate=learning_rate)
    print(f'creating agent with q_user={q_user.__name__}, optimizer={optimizer.__class__.__name__}')
    agent = Agent(agent_id, coord, optimizer, q_user, action2pos, action_shape, session,
                  'global' if is_target else f'agent_{agent_id}', input_shape, True, cfg)
    if is_target:
        environment = None
    else:
        environment = RelevantActionEnvironment(agent, Phone(f'device{agent_id}', 5554 + 2 * agent_id, cfg),
                                                action2pos, cfg)
        environment.add_callback(agent)
    if debug_mode and (is_target or isinstance(coord, UnSyncedMultiprocessRLCoordinator)):
        coord.add_callback(log_callbacks)
    if is_target:
        coord.add_callback(agent)
        agent.set_generated_gradient_target(agent)
    return environment, agent


with open('setting.yaml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

dummy_mode = cfg['dummy_mode']
inter_op_core_count = cfg['inter_op_core_count']
intra_op_core_count = cfg['intra_op_core_count']
agents_count = cfg['agents_count']
screen_shape = tuple(cfg['screen_shape'])
screen_new_shape = tuple(cfg['screen_new_shape'])
action_shape = tuple(cfg['action_shape'])
crop_top_left = cfg['crop_top_left']
crop_size = cfg['crop_size']
learning_rate = cfg['learning_rate']
multiprocessing = cfg['multiprocessing']
reset_summary = cfg['reset_summary']
summary_path = cfg['summary_path']
debug_mode = cfg['debug_mode']
value_estimator_values = cfg['value_estimator_values']

cfg['maintain_visited_activities'] = False
cfg['shuffle'] = True

q_users = [
    (most_probable_weighted_q_user, value_estimator_values[0]),
    (least_certain_weighted_q_user, value_estimator_values[1]),
    (random_q_user, value_estimator_values[2]),
    (most_certain_weighted_q_user, value_estimator_values[3]),
    (least_probable_weighted_q_user, value_estimator_values[4])
]
q_users = [p[0] for p in q_users if p[1] is None or p[1] < np.inf]
optimizers = [tf.train.AdamOptimizer, tf.train.RMSPropOptimizer]

if dummy_mode:
    Phone = phone.DummyPhone
else:
    Phone = phone.Phone
# make these two optional
tf.config.threading.set_inter_op_parallelism_threads(inter_op_core_count)
tf.config.threading.set_intra_op_parallelism_threads(intra_op_core_count)
tf.disable_v2_behavior()
# should i set batch size to None or 1?
input_shape = (*screen_shape, 3)

action2pos = partial(base.policy2pos, screen_new_shape=screen_new_shape, crop_size=crop_size,
                     crop_top_left=crop_top_left, action_shape=action_shape)


@contextlib.contextmanager
def dummy_context():
    yield None


with dummy_context() if multiprocessing else tf.Session() as sess:
    if __name__ == '__main__':
        base.remove_summaries(summary_path, reset_summary)

        learning_agent_creators = [partial(create_agent, i, False) for i in range(agents_count)]
        final_agent_creator = partial(create_agent, len(learning_agent_creators), True)

        if multiprocessing:
            coord = UnSyncedMultiprocessRLCoordinator(learning_agent_creators, final_agent_creator, False, cfg)
        else:
            coord = MultithreadRLCoordinator(learning_agent_creators, final_agent_creator, cfg)
        coord.start_learning()
