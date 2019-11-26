import multiprocessing as mp
import threading as td
from queue import Empty
from datetime import datetime
from typing import Tuple, List, Callable

from environment import Environment
from reinforcement_learning import RLCoordinator, RLAgent
from utils import Config, add_gradients

RLAgentCreator = Callable[[RLCoordinator], Tuple[Environment, RLAgent]]


# maybe the name should be asynchronous, and then we have two subclasses, multithread and multiprocess
class MultiprocessRLCoordinator(RLCoordinator):
    def __init__(self, learning_agent_creators: List[RLAgentCreator], final_agent_creator: RLAgentCreator,
                 input_shape: tuple, config: Config):
        self.send_queue = None
        self.receive_queue = None
        self.agent = None
        self.learning_agent_creators = learning_agent_creators
        self.final_agent_creator = final_agent_creator
        self.input_shape = input_shape
        self.gradient_queue_size = config['gradient_queue_size']
        self.steps_per_save = config['steps_per_save']
        self.save_to_path = config['save_to_path']

    def add_gradient(self, agent_id, gradient):
        self.send_queue.put(gradient)
        new_weights = None
        try:
            while True:
                new_weights = self.receive_queue.get_nowait()
        except Empty:
            pass
        if new_weights is not None:
            print(f'{datetime.now()}: agent #{agent_id} is updating weights')
            # neater api for setting weights
            self.agent.rl_model.set_weights(new_weights)

    def run_agent(self, send_queue: mp.Queue, receive_queue: mp.Queue, agent_i: int) -> None:
        self.send_queue = send_queue
        self.receive_queue = receive_queue
        environment, self.agent = self.learning_agent_creators[agent_i](self)
        environment.start()
        send_queue.put('done')
        while True:
            fin = self.receive_queue.get()
            if fin == True:
                break

    # do i have to specify device here (like cpu:0 or :1)
    def start_learning(self):
        queues = []
        for agent_i in range(len(self.learning_agent_creators)):
            queues += [(mp.Queue(self.gradient_queue_size), mp.Queue(self.gradient_queue_size))]
        processes = []
        for agent_i in range(len(self.learning_agent_creators)):
            processes += [mp.Process(name=f'learning agent #{agent_i}', target=self.run_agent,
                                     args=(*queues[agent_i], agent_i))]
            processes[-1].start()
        _, final_agent = self.final_agent_creator(self)
        if not final_agent.is_built():
            final_agent.build_model(self.input_shape)

        update_count = 0
        all_queues = [q for q in queues]
        while len(queues) > 0:
            updated = False
            for receive_queue, send_queue in [q for q in queues]:
                # or maybe i can sum all gradients and then apply once
                gradient = 0
                try:
                    while True:
                        tmp = receive_queue.get_nowait()
                        if tmp == 'done':
                            queues.remove((receive_queue, send_queue))
                            send_queue.put(True)
                            break
                        else:
                            gradient = add_gradients(tmp, gradient)
                except Empty:
                    pass
                if gradient != 0:
                    print(f'{datetime.now()}: applying gradients')
                    final_agent.apply_gradient(gradient)
                    updated = True
            if updated:
                if update_count > 0 and update_count % (self.steps_per_save * len(self.learning_agent_creators)) == 0:
                    print(f'{datetime.now()}: saving ...')
                    # neater api
                    final_agent.rl_model.save_weights(self.save_to_path)
                update_count += 1
                print(f'{datetime.now()}: sending updated weights')
                for _, send_queue in [q for q in queues]:
                    # neater api for getting weights
                    send_queue.put(final_agent.rl_model.get_weights())
        for q1, q2 in all_queues:
            q1.close()
            q2.close()
        for p in processes:
            p.join()


class MultithreadRLCoordinator(RLCoordinator):
    def __init__(self, learning_agent_creators: List[RLAgentCreator], final_agent_creator: RLAgentCreator,
                 input_shape: tuple, config: Config):
        self.learning_agent_creators = learning_agent_creators
        self.final_agent_creator = final_agent_creator
        self.input_shape = input_shape
        self.steps_per_agent = config['steps_per_agent']
        self.steps_per_save = config['steps_per_save']
        self.save_to_path = config['save_to_path']
        self.agents = []
        self.final_agent = None
        self.update_count = 0

    def add_gradient(self, agent_id, gradient):
        self.final_agent.apply_gradient(gradient)
        self.agents[agent_id][1].replace_weights(self.final_agent)
        if agent_id == 0:
            if self.update_count > 0 and self.update_count % self.steps_per_save == 0:
                print(f'{datetime.now()}: saving ...')
                # neater api
                self.agents[agent_id][1].rl_model.save_weights(self.save_to_path)
            self.update_count += 1

    def run_agent(self, agent_i: int) -> None:
        self.agents[agent_i][0].start()

    # do i have to specify device here (like cpu:0 or :1)
    def start_learning(self):
        _, self.final_agent = self.final_agent_creator(self)
        if not self.final_agent.is_built():
            self.final_agent.build_model(self.input_shape)
        processes = []
        for agent_i in range(len(self.learning_agent_creators)):
            self.agents += [self.learning_agent_creators[agent_i](self)]
            processes += [td.Thread(name=f'learning agent #{agent_i}', target=self.run_agent, args=(agent_i,))]
            processes[-1].start()
        for p in processes:
            p.join()
