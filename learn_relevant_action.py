import contextlib
import os
import random
from datetime import datetime
from functools import partial

import tensorflow as tf
import yaml

import phone
from coordinators import UnSyncedMultiprocessRLCoordinator, MultithreadRLCoordinator, MultiCoordinatorCallbacks
from relevant_action import RelevantActionEnvironment
import relevant_action_base as base

# one problem is the exploration during training of relevant actions. if the exploration is not good, i need to
#   find a way. one way is to have a very large batch size. another way is to use the state_finding algorithm
#   (see the presentation) to find out if an action i take brings me to an already seen state, and do not epxlore it
#   again.
# when i have force_app_on_top set to True, i dont think the current approach brings the app to front if the front
#   app is in the same task as the goal app
# note that for different actions, different neural networks are required (like ,for click probably i need smaller CNN
#     kernel than scroll)
# add action for back button.
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


policy_users = [(base.most_probable_weighted_policy_user, 1.0)]
optimizers = [tf.train.AdamOptimizer, tf.train.RMSPropOptimizer]


class LogCallbacks(MultiCoordinatorCallbacks):
    def log(self, log: str) -> None:
        print(log)

    def on_update_learner(self, learner_id: int) -> None:
        print(f'{datetime.now()}: learner #{learner_id} got updated.')

    def on_update_target(self, learner_id: int) -> None:
        print(f'{datetime.now()}: target updated by #{learner_id}.')


log_callbacks = LogCallbacks()


def create_agent(agent_id, is_target, coord):
    session = sess or tf.Session().__enter__()

    policy_user, value = policy_users[agent_id % len(policy_users)]
    optimizer = random.choice(optimizers)() if learning_rate is None \
        else random.choice(optimizers)(learning_rate=learning_rate)
    print(f'creating agent with policy_user={policy_user.__name__}, optimizer={optimizer.__class__.__name__}')
    # here the target needs to be untrainable, in that the gradient callbacks should not be called ont it, but i cannot
    #   pass false for "trainable", because i need the gradient applying graph nodes ... what to do?
    agent = base.Agent(agent_id, coord, optimizer, policy_user, action2pos, action_shape, value, session,
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

cfg['maintain_visited_activities'] = False
cfg['shuffle'] = True
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
