import contextlib
import os
import random
import shutil
from datetime import datetime
from functools import partial
from io import BytesIO
from typing import Tuple, Optional, Callable

import matplotlib.cm as cm
import numpy as np
import scipy.misc
import tensorflow as tf
import yaml

import phone
from actor_critic import A2C
from coordinators import UnSyncedMultiprocessRLCoordinator, MultithreadRLCoordinator, MultiCoordinatorCallbacks
from reinforcement_learning import RLCoordinator, Episode
from relevant_action import RelevantActionEnvironment
from screen_parsing import ScreenEncoder, PolicyGenerator, ValueEstimator
from tf1 import TF1RLAgent, LazyGradient
from utils import Config


# sometimes because of high computation load, it the click becomes long click
# i think if my batch_size is 1, there is a risk of overfit
# if everything is fucked up, online remove the device and create a new one
# why many python learn_relevant_action.py ?
# set the affinity and see if it helps performance
# change the inter and intra op parallelism and see if it helps performance
# for assessing changes in the screen, if wait until i reach stable screen (no change for 2 consecutive reads)
#   and then continue. this way i can prevent the fact that wait_time is not enough for some situations
# include a way to interact with the headless emulators on demand
# have a monitoring panel, where e.g. i can add new agents or stop existing ones (can be part of
#   coordinator and therefore part of the library) or i can change params of the task online
# add a proper way to abort the learning (right now ctrl+c does not work)
# test to make sure the action is correctly done and logged (what i see in
#   tensorboard is exactly where the agent clicked)
# even most of this file can be a part of framework for plug and play style (rapidly set a network and see results)
# i have a lot of common code in here and test_doom.py think if these can be factorized
# when putting this on github as a framework, add some examples, like non-deep learning, or synchronized a2c
#   (basically batched a2c, where i have multiple envs but one network, and use it in a batched way)
# generally read about off-policy and why it does not work
# ideas:
#     constant (0 gradient) v instead of learned v: means comparing against the best scenaroi rather than the prediction
#           , which is suboptimal
#     for off-policy to work, give all actions beforehand to a RNN and then add it to the initial state. for on policy
#           set this value to, say, 0.
# code in correct python style use object, property, etc.
# how can i use tensorflow capabilities to make running the episodes completely in C and real parallel
#     like specify the graph first (call the mode in loop, then do the logs, etc.) and then just simply run it.
# the way i calculate gradient 2 episodes after the original episode, there is a weight update in between which maybe
#     harmful
# make sure to check vairbales are created only inside the scopes. which means i have to check all variables in the
#     global space in a test (and even with assert before running)
# is it possible to prevent retracing when shape changes?
#   (if i generate the input inside @tf.function it does not retrace)
# apparently i can use tf.function on all functions regardless of if it is tf control flow or now. use this to ditch
#   multiprocessing (did i mean retracing by this?)--> test it tho first (in a simple test file)
# do some of the stuff that people do to prevent high variance and instability in policy gradient
# look at all warnings
# v values can change during training --> steps_per_gradient_update can affect this
# maybe i can cache the trainable weights for speed
# add the logs for the final model as well
# note that the optimizers ar eof no use except for the final agent
# one idea is to have multiple agents that have different architectures with shared components, e.g. some use continuous
#     and some discrete actions
# check emulator sometimes it resets (generally, have a mechanism for resetting a worker when something fails)
# add action resolution parameter
# add an off policy buffer and add similar actions
# create a dependency yaml file
# add different policyUsers
# plot the network architecture
# plot images and scenarios in tensorboard
# save different version of the model (with different names, probably identifying the accuracy at the moment of saving)
# set name for each of the network elements
# how to run emulator on a specific core
# look at the todos and comments in the other project (code)
# test the network independent of the task (only on a couple of image in a trivial dataset that i create to test)
# maybe actor critic is not the best option here. look into q learning and REINFORCE


class PolicyUser:
    def __init__(self, policy_user: Callable):
        self.policy_user = policy_user

    def __call__(self, policy: tf.Tensor) -> tf.Tensor:
        self.policy = policy
        return self.policy_user(policy)


class Agent(TF1RLAgent, MultiCoordinatorCallbacks):
    def __init__(self, id: int, coordinator: Optional[RLCoordinator], optimizer: tf.train.Optimizer,
                 policy_user: Callable[[tf.Tensor], tf.Tensor], action2pos: Callable, action_shape: Tuple[int, int],
                 value: Optional[float], sess: tf.Session, scope: str, input_shape: Tuple[int, ...], cfg: Config):
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
        self.save_to_path = cfg['save_to_path']
        save_max_keep = cfg['save_max_keep']
        self.debug_mode = cfg['debug_mode']
        self.steps_per_log = cfg['steps_per_log']
        self.target_updates_per_save = cfg['target_updates_per_save']

        self.target_updates = 0
        self.unlogged_gradients = []

        self.log_names = ('Loss/Policy Loss', 'Loss/Value Loss', 'Loss/Entropy', 'Loss/Advantage',
                          'Model/Weights', 'Loss/Gradients', 'Loss/Policy Max')

        self.mean_episode_reward = []
        self.mean_loss = []
        self.mean_lengths = []
        self.logs = ()

        self.log_step = 0

        action_tensor_shape = (*action_shape, action_type_count)

        with tf.variable_scope(scope):
            screen_encoder = ScreenEncoder(crop_top_left, crop_size, self.screen_new_shape,
                                           conv_kernel_sizes, conv_filter_nums, conv_stride_sizes)
            policy_user = PolicyUser(policy_user)
            rl_model = A2C(screen_encoder, PolicyGenerator(action_tensor_shape, deconv_kernel_sizes,
                                                           deconv_filter_nums, deconv_output_shapes),
                           ValueEstimator(value), policy_user, action_tensor_shape, cfg)

        self.summary_writer = tf.summary.FileWriter(f'{summary_path}/agent{id}')

        super(Agent, self).__init__(id, rl_model, coordinator, scope, sess, input_shape, optimizer, cfg)

        self.output_logs_e += (tf.linalg.global_norm(self.trainable_weights),
                               tf.linalg.global_norm(self.clip_gradient(self.output_gradients)),
                               tf.reduce_max(policy_user.policy))
        if self.log_new_screen:
            self.new_screen_log_index = len(self.output_logs_e)
            self.output_logs_e += (screen_encoder.get_processed_screen(),)
        if self.log_policy:
            self.policy_log_index = len(self.output_logs_e)
            self.output_logs_e += (policy_user.policy,)

        self.saver = tf.train.Saver(var_list=self.trainable_weights, max_to_keep=save_max_keep)
        if load_model:
            # test this
            # also, this does not work for optimizer parameters because they will be re-initialized. do sth about it
            self.saver.restore(sess, tf.train.get_checkpoint_state(self.save_to_path).model_checkpoint_path)

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

            # measure how much time these logs take, if too much, justt disable them for
            #   everything except the test agent (the final agent)
            if self.log_screen:
                env_state = episode.states_tb[0][0] * 255.0
                action = action2pos(episode.actions_tb[1][0], original=True)
                env_state = grid_image(env_state, env_state.shape[:-1] // np.array(self.action_shape), (0, 0, 0))
                env_state[action[1], action[0], :] = [255, 0, 0]
                summary.value.add(tag='episodes', image=get_image_summary(env_state))
            if self.log_new_screen:
                env_state = gradient.logs_e[self.new_screen_log_index][0, 0] * 255.0
                action = action2pos(episode.actions_tb[1][0], original=False)
                env_state = grid_image(env_state, np.array(self.screen_new_shape) // np.array(self.action_shape),
                                       (0, 0, 0))
                env_state[action[1], action[0], :] = [255, 0, 0]
                summary.value.add(tag='processed episode', image=get_image_summary(env_state))
            # this is only click policy. make it more general (IDK how tho :|)
            if self.log_policy:
                policy = np.reshape(gradient.logs_e[self.policy_log_index][0, 0], self.action_shape)
                policy = cm.viridis(policy)[:, :, :3] * 255.0
                summary.value.add(tag='policy', image=get_image_summary(policy))
            # i can also visualize layers of CNN

            self.summary_writer.add_summary(summary, step)
            self.summary_writer.flush()

    def log(self, log: str) -> None:
        print(log)

    def on_update_learner(self, learner_id: int) -> None:
        print(f'{datetime.now()}: learner #{learner_id} got updated.')

    def on_update_target(self, learner_id: int) -> None:
        print(f'{datetime.now()}: target updated by #{learner_id}.')
        self.target_updates += 1
        if self.target_updates % self.target_updates_per_save == 0:
            self.saver.save(self.session, self.save_to_path)

    def on_episode_end(self, premature: bool) -> None:
        if self.debug_mode:
            print(f'episode ended in #{self.id}')
        super().on_episode_end(premature)

    def on_episode_start(self, env_state) -> None:
        if self.debug_mode:
            print(f'episode started in #{self.id}')
        super().on_episode_start(env_state)


def get_image_summary(image: np.ndarray) -> tf.Summary.Image:
    bio = BytesIO()
    scipy.misc.toimage(image).save(bio, format="png")
    res = tf.Summary.Image(encoded_image_string=bio.getvalue(), height=image.shape[0], width=image.shape[1])
    bio.close()
    return res


def action2pos(action: np.ndarray, screen_new_shape: Tuple[int, ...],
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


def grid_image(image: np.ndarray, grid_size: Tuple[int, int], color: Tuple[int, int, int]) -> np.ndarray:
    image = image.copy()
    for i in range(0, image.shape[0], grid_size[0]):
        image[i] = color
    for j in range(0, image.shape[1], grid_size[1]):
        image[:, j] = color
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
    return tf.argmax(tf.distributions.Multinomial(1.0, probs=tf.ones_like(policy) / tf.shape(policy)[-1]).sample(),
                     axis=-1)


policy_users = [(most_probable_weighted_policy_user, None)]
optimizers = [tf.train.AdamOptimizer, tf.train.RMSPropOptimizer]


def create_agent(agent_id, is_target, coord):
    session = sess or tf.Session().__enter__()

    policy_user, value = policy_users[agent_id % len(policy_users)]
    optimizer = random.choice(optimizers)() if learning_rate is None \
        else random.choice(optimizers)(learning_rate=learning_rate)
    print(f'creating agent with policy_user={policy_user.__name__}, optimizer={optimizer.__class__.__name__}')
    agent = Agent(agent_id, coord, optimizer, policy_user, action2pos, action_shape, value, session,
                  'global' if is_target else f'agent_{agent_id}', input_shape, cfg)
    if is_target:
        environment = None
    else:
        environment = RelevantActionEnvironment(agent, Phone(f'device{agent_id}', 5554 + 2 * agent_id, cfg),
                                                action2pos, cfg)
        environment.add_callback(agent)
    if debug_mode and (is_target is True or isinstance(coord, UnSyncedMultiprocessRLCoordinator)):
        coord.add_callback(agent)
    if is_target:
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
conv_kernel_sizes = cfg['conv_kernel_sizes']
conv_filter_nums = cfg['conv_filter_nums']
conv_stride_sizes = cfg['conv_stride_sizes']
deconv_kernel_sizes = cfg['deconv_kernel_sizes']
deconv_filter_nums = cfg['deconv_filter_nums']
deconv_output_shapes = cfg['deconv_output_shapes']
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

action2pos = partial(action2pos, screen_new_shape=screen_new_shape, crop_size=crop_size,
                     crop_top_left=crop_top_left, action_shape=action_shape)


@contextlib.contextmanager
def dummy_context():
    yield None


with dummy_context() if multiprocessing else tf.Session() as sess:
    if __name__ == '__main__':
        while reset_summary and len(os.listdir(f'{summary_path}/agent0')) > 0:
            for agent_i in range(agents_count):
                try:
                    for f in os.listdir(f'{summary_path}/agent{agent_i}'):
                        os.unlink(f'{summary_path}/agent{agent_i}/{f}')
                except FileNotFoundError:
                    pass

        learning_agent_creators = [partial(create_agent, i, False) for i in range(agents_count)]
        final_agent_creator = partial(create_agent, len(learning_agent_creators), True)

        if multiprocessing:
            coord = UnSyncedMultiprocessRLCoordinator(learning_agent_creators, final_agent_creator, False, cfg)
        else:
            coord = MultithreadRLCoordinator(learning_agent_creators, final_agent_creator, cfg)
        coord.start_learning()
