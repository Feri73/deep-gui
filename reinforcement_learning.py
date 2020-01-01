# do not use eager execution when possible (use tf.function)
# make this a library --> general rl library in tf 2 , maybe even make it in a way that tf parts of it are separated
#     and its core is platform-independent
import time
from abc import ABC, abstractmethod
# look everywhere to see if i am indirectly modifying stuff (instead of functional style)
# instead of weights, use trainable weights (both in replacing and getting and generally everywhere)
# 2 thigs to do --> check openai implementation to see if they clip the add_gradients
# correct the way i synchronize the workers and global network so that i update the workers before they go any further
#     in both coordinators
# check it i am using tapes correctly --> (gradients are not accumulated from one episode to another)
# check the implementation of asynchronous coordinators to see of i am updating weights
#   and computing gradients correctly
#   also see why the weights and gradients and everything keeps increasing
# then define a simple task (test) and make sure it works
# and also think about the fact that a2c the way i implementde it only gets trained on episodes where reward is 0
#   (is it? i remember the situations where it would get stuck because and reward was 0 so it would never learn)
#   if the above is true, then 1. i can use it to create better policyUsers 2. maybe if no paper says this,
#   i can make a theory out of it
# eager mode is very slow!!!
from typing import Tuple, Any, Optional, List, Union

import numpy as np

from environment import EnvironmentCallbacks, EnvironmentController
from utils import Config, MemVariable, MemList

ModelStates = tuple


class Episode:
    def __init__(self, id: int):
        self.id = id
        self.finished = False
        self.states_tb = []
        self.actions_tb = []
        self.rewards_tb = []

    def add_step(self, state_b, action_b, reward_b: List[float]):
        self.states_tb += [state_b]
        self.actions_tb += [action_b]
        self.rewards_tb += [reward_b]

    def __len__(self) -> int:
        return len(self.states_tb)

    def __getitem__(self, item: int) -> Tuple[Any, Any, Any]:
        return self.states_tb[item], self.actions_tb[item], self.rewards_tb[item]


class RLCoordinator(ABC):
    @abstractmethod
    def start_learning(self) -> None:
        pass

    @abstractmethod
    def add_gradient(self, agent_id: int, gradient) -> None:
        pass


# this should be in a better file and maybe should be in included in RLAgent
class RLModel(ABC):
    @abstractmethod
    def calc_next_action(self, env_states_bt, bef_actions_bt, bef_rewards_bt,
                         bef_states_eb: ModelStates) -> Tuple[Any, ModelStates]:
        pass

    @abstractmethod
    def calc_loss(self, actions_bt, rewards_bt, model_states_ebt: ModelStates, finished_b) -> Tuple[Any, tuple]:
        pass

    @abstractmethod
    def get_default_action(self):
        pass

    @abstractmethod
    def get_default_reward(self) -> float:
        pass

    @abstractmethod
    def get_default_states(self) -> ModelStates:
        pass


# rethink the functions (like iteration count should be input of start learning, etc.) or names and meaning of
#   classes (like RLCoordinator is only about learning, while RLAgent plays too!)
# this should inherit from Model
# one way instead of adding gradients is to archive as many gradients as we want, and then loop on them and apply them
# remember, if i set max_steps, this can be more efficient
# i need much better nameing in this, tf1, and a2c. i should explicitly specify the if a value is a bef_ value or not
#   in other words, e.g. if an action is the input or the output to the env_state
# one option to add is to make gradients based on weight, so i can save weights if i know i'm gonna change it, compute
#   (and even apply) gradient based on that weight, and then change back (even sometimes no change back cuz it's gonna
#   get updated to the master's weights)
class RLAgent(EnvironmentCallbacks, EnvironmentController):
    def __init__(self, id: int, rl_model: RLModel, coordinator: Optional[RLCoordinator], cfg: Config):
        self.id = id
        self.rl_model = rl_model
        self.coordinator = coordinator

        self.default_action = self.rl_model.get_default_action()
        self.default_reward = self.rl_model.get_default_reward()
        self.default_model_states = self.rl_model.get_default_states()

        self.episodes_per_gradient_update = cfg['episodes_per_gradient_update']
        self.episodes_per_agent = cfg['episodes_per_agent']
        self.max_steps_per_episode = cfg['max_steps_per_episode'] or np.inf
        self.late_gradient = cfg['late_gradient']

        self.current_episode_num = 0
        self.in_on_wait = False

        self.episode = MemVariable(lambda: None)
        self.model_states_teb = MemVariable(lambda: [tuple([state] for state in self.default_model_states)])
        self.episode_vars = MemList([self.episode, self.model_states_teb])

        self.total_gradient = None

    def on_episode_gradient_computed(self, episode: Episode, gradient) -> None:
        pass

    # this is not exactly what arthur juliani does. cuz i do not continue anymore
    # why if i set max_steps to a lower value, the average of episode lengths also becomes lower?
    def should_continue_episode(self) -> bool:
        # check
        return len(self.episode.value) <= self.max_steps_per_episode

    def should_start_episode(self) -> bool:
        res = self.current_episode_num < self.episodes_per_agent
        # i should not make these calls here. I should have a callback that says everything is done, and call there.
        if not res:
            self.add_last_gradient()
            self.compute_last_gradient()
            self.coordinator.add_gradient(self.id, self.total_gradient)
        return res

    def on_episode_start(self, env_state) -> None:
        self.current_episode_num += 1
        self.episode.value = Episode(self.current_episode_num)
        self.episode.value.add_step([env_state], [self.default_action], [self.default_reward])

    def get_next_action(self, state) -> Any:
        action_b, model_states_eb = self.calc_next_action(*self.episode.value[-1], self.model_states_teb.value[-1])
        # generally, in functions that say get_*, i should not change anything. i should think of another method of
        #   saving model_states
        self.model_states_teb.value += [model_states_eb]
        return action_b[0]

    def compute_last_gradient(self):
        if self.episode_vars.has_archive():
            gradient = self.calc_gradient(self.episode.last_value(), self.model_states_teb.last_value())
            self.on_episode_gradient_computed(self.episode.last_value(), gradient)
            self.total_gradient = gradient + self.total_gradient
            self.episode_vars.reset_archive()

    def add_last_gradient(self):
        if self.total_gradient is not None and \
                self.current_episode_num % self.episodes_per_gradient_update == int(self.late_gradient):
            self.coordinator.add_gradient(self.id, self.total_gradient)
            self.total_gradient = None

    def on_wait(self) -> None:
        if not self.late_gradient:
            return
        self.in_on_wait = True
        self.compute_last_gradient()
        # add another option to remove this if
        if len(self.episode.value) == self.max_steps_per_episode:
            self.add_last_gradient()
        self.in_on_wait = False

    def on_state_change(self, src_state, action, dst_state, reward) -> None:
        self.episode.value.add_step([dst_state], [action], [reward])

    def on_episode_end(self, premature: bool) -> None:
        self.episode.value.finished = not premature
        self.episode_vars.archive()
        if not self.late_gradient:
            self.compute_last_gradient()
        self.add_last_gradient()

    # remember, some of the children of this class (e.g. tf2 where it uses multiple tapes) need to override this
    #   and add their own procedures
    def on_error(self) -> None:
        self.episode_vars.reset_value()
        if self.in_on_wait and self.episode_vars.has_archive():
            self.episode_vars.reset_archive()

    @abstractmethod
    def calc_next_action(self, env_state_b, action_b, reward_b,
                         model_states_eb: ModelStates) -> Tuple[Any, ModelStates]:
        pass

    @abstractmethod
    def set_generated_gradient_target(self, target: 'RLAgent'):
        pass

    @abstractmethod
    def calc_gradient(self, episode: Episode, states_teb: List[ModelStates]):
        pass

    @abstractmethod
    def apply_gradient(self, gradients) -> None:
        pass

    # here, target means reference! the one like whom we want to become
    @abstractmethod
    def add_replacement_target(self, target: 'RLAgent') -> None:
        pass

    @abstractmethod
    def replace_parameter(self, target: Union['RLAgent', Any]) -> None:
        pass

    # is there a better way than including all these functions because i KNOW that there is a multiprocess that cannot
    #   transfer rlagent over pickle?
    @abstractmethod
    def get_parameter(self):
        pass
