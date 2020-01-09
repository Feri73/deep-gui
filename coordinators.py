import multiprocessing
import signal
import threading as td
from datetime import datetime
from queue import Empty, Full
from typing import Tuple, List, Callable

from environment import Environment
from reinforcement_learning import RLCoordinator, RLAgent
from utils import Config

RLAgentCreator = Callable[[RLCoordinator], Tuple[Environment, RLAgent]]


# do i really needs these? i mean, by overwriting rlAgent I can have the two on_update functionalities
class MultiCoordinatorCallbacks:
    # when i use this i should not include datetime
    # also, make this log to get lambda rather than text, so that i don't have to generate text id no callback is set
    def log(self, log: str) -> None:
        pass

    def on_update_learner(self, learner_id: int) -> None:
        pass

    def on_update_target(self, learner_id: int) -> None:
        pass


class UnSyncedMultiprocessRLCoordinator(RLCoordinator):
    def __init__(self, learning_agent_creators: List[RLAgentCreator],
                 final_agent_creator: RLAgentCreator, block: bool, config: Config):
        self.send_queue = None
        self.receive_queue = None
        self.agent = None
        self.learning_agent_creators = learning_agent_creators
        self.final_agent_creator = final_agent_creator
        self.block = block
        self.gradient_queue_size = config['gradient_queue_size']
        self.callbacks = []

    def add_callback(self, callback: MultiCoordinatorCallbacks) -> None:
        self.callbacks += [callback]

    def add_gradient(self, agent_id, gradient):
        self.log(f'{datetime.now()}: agent #{agent_id} is sending gradients - {self.send_queue.qsize()}')
        self.send_queue.put(gradient)
        new_weights = None
        self.log(f'{datetime.now()}: agent #{agent_id} is contacting coordinator - {self.receive_queue.qsize()}')
        queue_size = 0
        if not self.block:
            queue_size = self.receive_queue.qsize()
        while self.block or queue_size > 0:
            try:
                new_weights = self.receive_queue.get(block=self.block)
                queue_size -= 1
                if self.block:
                    break
            except Empty:
                pass
        if new_weights is not None:
            self.log(f'{datetime.now()}: agent #{agent_id} is updating weights')
            self.agent.replace_parameter(new_weights)
            self.on_update_learner(self.agent.id)

    def run_agent(self, send_queue: multiprocessing.Queue, receive_queue: multiprocessing.Queue, agent_i: int) -> None:
        signal.signal(signal.SIGINT, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt))
        self.send_queue = send_queue
        self.receive_queue = receive_queue
        environment, self.agent = self.learning_agent_creators[agent_i](self)
        new_weights = self.receive_queue.get()
        self.agent.replace_parameter(new_weights)
        self.send_queue.put(f'id: {self.agent.id}')
        environment.start()
        self.log(f'{datetime.now()}: agent #{self.agent.id}\'s environment finished')
        send_queue.put('done')
        while True:
            fin = self.receive_queue.get()
            if fin == True:
                break

    # do i have to specify device here (like cpu:0 or :1)
    def start_learning(self):
        mp = multiprocessing.get_context('spawn')
        queues = []
        for agent_i in range(len(self.learning_agent_creators)):
            queues += [(mp.Queue(1 if self.block else self.gradient_queue_size),
                        mp.Queue(1 if self.block else self.gradient_queue_size))]
        processes = []
        for agent_i in range(len(self.learning_agent_creators)):
            processes += [mp.Process(name=f'learning agent #{agent_i}', target=self.run_agent,
                                     args=(*queues[agent_i], agent_i))]
            processes[-1].start()
        _, final_agent = self.final_agent_creator(self)

        for queue_i, (_, send_queue) in enumerate(queues):
            send_queue.put(final_agent.get_parameter())

        agent_ids = [-1 for _ in queues]
        for queue_i, (receive_queue, _) in enumerate(queues):
            agent_ids[queue_i] = int(receive_queue.get().split(': ')[1])

        all_queues = [q for q in queues]
        while len(queues) > 0:
            updated = False
            tmp_agent_ids = [a_id for a_id in agent_ids]
            for queue_i, (receive_queue, send_queue) in enumerate([q for q in queues]):
                # or maybe i can sum all gradients and then apply once
                gradient = None
                agent_id = tmp_agent_ids[queue_i]
                # consider self.block both here and in the similar loop in add_gradient
                queue_size = 0
                if not self.block:
                    queue_size = receive_queue.qsize()
                while self.block or queue_size > 0:
                    try:
                        tmp = receive_queue.get(block=self.block)
                        queue_size -= 1
                        if tmp == 'done':
                            queues.remove((receive_queue, send_queue))
                            agent_ids.remove(agent_id)
                            send_queue.put(True)
                            break
                        else:
                            gradient = tmp + gradient
                        if self.block:
                            break
                    except Empty:
                        pass
                        # self.log(f'{datetime.now()}: False empty while reading from #{agent_id}')
                if gradient is not None:
                    self.log(f'{datetime.now()}: applying gradients from #{agent_id} - {receive_queue.qsize()}')
                    final_agent.apply_gradient(gradient)
                    self.on_update_target(agent_id)
                    updated = True
            if updated:
                self.log(f'{datetime.now()}: starting sending updates')
                for queue_i, (_, send_queue) in enumerate([q for q in queues]):
                    self.log(f'{datetime.now()}: sending updated '
                             f'weights to #{agent_ids[queue_i]} - {send_queue.qsize()}')
                    # test this (not blocking when sending update) with obvious tasks (like ajuliani) to see if it
                    #   creates any problem
                    try:
                        send_queue.put(final_agent.get_parameter(), block=self.block)
                    except Full:
                        pass
        for p in processes:
            p.join()
        for q1, q2 in all_queues:
            q1.close()
            q2.close()

    def log(self, log: str) -> None:
        for callback in self.callbacks:
            callback.log(log)

    def on_update_learner(self, learner_id: int) -> None:
        for callback in self.callbacks:
            callback.on_update_learner(learner_id)

    def on_update_target(self, learner_id: int) -> None:
        for callback in self.callbacks:
            callback.on_update_target(learner_id)


# logs here should also go to tensorboard

# add the early stopping for synced multiprocess as well

# explicitly say multiprocessing should not be fork-based
# maybe the name should be asynchronous, and then we have two subclasses, multithread and multiprocess
class SyncedMultiprocessRLCoordinator(RLCoordinator):
    def __init__(self, learning_agent_creators: List[RLAgentCreator],
                 final_agent_creator: RLAgentCreator, config: Config):
        self.send_queue = None
        self.receive_queue = None
        self.agent = None
        self.learning_agent_creators = learning_agent_creators
        self.final_agent_creator = final_agent_creator
        self.callbacks = []

    def add_callback(self, callback: MultiCoordinatorCallbacks) -> None:
        self.callbacks += [callback]

    def add_gradient(self, agent_id, gradient):
        self.log(f'{datetime.now()}: agent #{agent_id} is sending gradients - {self.send_queue.qsize()}')
        self.send_queue.put(gradient)
        self.log(f'{datetime.now()}: agent #{agent_id} is contacting coordinator - {self.receive_queue.qsize()}')
        new_weights = self.receive_queue.get()
        self.log(f'{datetime.now()}: agent #{agent_id} is updating weights')
        # neater api for setting weights
        self.agent.replace_parameter(new_weights)
        self.on_update_learner(self.agent.id)

    def run_agent(self, send_queue: multiprocessing.Queue, receive_queue: multiprocessing.Queue, agent_i: int) -> None:
        self.send_queue = send_queue
        self.receive_queue = receive_queue
        environment, self.agent = self.learning_agent_creators[agent_i](self)
        new_weights = self.receive_queue.get()
        self.agent.replace_parameter(new_weights)
        self.send_queue.put(f'id: {self.agent.id}')
        environment.start()
        send_queue.put('done')
        while True:
            fin = self.receive_queue.get()
            if fin == True:
                break

    # do i have to specify device here (like cpu:0 or :1)
    def start_learning(self):
        mp = multiprocessing.get_context('spawn')
        queues = []
        for agent_i in range(len(self.learning_agent_creators)):
            queues += [(mp.Queue(1), mp.Queue(1))]
        processes = []
        for agent_i in range(len(self.learning_agent_creators)):
            processes += [mp.Process(name=f'learning agent #{agent_i}', target=self.run_agent,
                                     args=(*queues[agent_i], agent_i))]
            processes[-1].start()
        _, final_agent = self.final_agent_creator(self)

        for queue_i, (_, send_queue) in enumerate(queues):
            send_queue.put(final_agent.get_parameter())

        agent_ids = [-1 for _ in queues]
        for queue_i, (receive_queue, _) in enumerate(queues):
            agent_ids[queue_i] = int(receive_queue.get().split(': ')[1])

        all_queues = [q for q in queues]
        while len(queues) > 0:
            for queue_i, (receive_queue, send_queue) in enumerate([q for q in queues]):
                try:
                    # or maybe i can sum all gradients and then apply once
                    agent_id = agent_ids[queue_i]
                    # consider self.block both here and in the similar loop in add_gradient
                    msg = receive_queue.get_nowait()
                    if msg == 'done':
                        queues.remove((receive_queue, send_queue))
                        del agent_ids[queue_i]
                        send_queue.put(True)
                        break
                    else:
                        gradient = msg
                        self.log(f'{datetime.now()}: applying gradients from #{agent_id} - {receive_queue.qsize()}')
                        final_agent.apply_gradient(gradient)
                        self.on_update_target(agent_id)
                        self.log(f'{datetime.now()}: sending updated '
                                 f'weights to #{agent_ids[queue_i]} - {send_queue.qsize()}')
                        send_queue.put(final_agent.get_parameter())
                except Exception as ex:
                    print(f'An exception happened: {ex}\nignoring it')
        for q1, q2 in all_queues:
            q1.close()
            q2.close()
        for p in processes:
            p.join()

    def log(self, log: str) -> None:
        for callback in self.callbacks:
            callback.log(log)

    def on_update_learner(self, learner_id: int) -> None:
        for callback in self.callbacks:
            callback.on_update_learner(learner_id)

    def on_update_target(self, learner_id: int) -> None:
        for callback in self.callbacks:
            callback.on_update_target(learner_id)


# explicitly say multithreading should not be fork-based
# maybe here i can control the speed of agents (e.g. if an agent has seen much more episodes, make it sleep a bit)
class MultithreadRLCoordinator(RLCoordinator):
    def __init__(self, learning_agent_creators: List[RLAgentCreator],
                 final_agent_creator: RLAgentCreator, config: Config):
        self.learning_agent_creators = learning_agent_creators
        self.final_agent_creator = final_agent_creator
        self.agents = []
        self.final_agent = None
        self.callbacks = []

    def add_callback(self, callback: MultiCoordinatorCallbacks) -> None:
        self.callbacks += [callback]

    def add_gradient(self, agent_id, gradient):
        self.final_agent.apply_gradient(gradient)
        self.on_update_target(agent_id)
        self.agents[agent_id][1].replace_parameter(self.final_agent)
        self.on_update_learner(agent_id)

    def run_agent(self, agent_i: int) -> None:
        self.agents[agent_i][1].replace_parameter(self.final_agent)
        self.agents[agent_i][0].start()

    # do i have to specify device here (like cpu:0 or :1)
    def start_learning(self):
        _, self.final_agent = self.final_agent_creator(self)
        processes = []
        for agent_i in range(len(self.learning_agent_creators)):
            self.agents += [self.learning_agent_creators[agent_i](self)]
            self.agents[-1][1].add_replacement_target(self.final_agent)
            self.agents[-1][1].set_generated_gradient_target(self.final_agent)
            processes += [td.Thread(name=f'learning agent #{agent_i}', target=self.run_agent, args=(agent_i,))]
        # both here and in multiprocess_coord, i should have a callback for when all agents are created
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    def log(self, log: str) -> None:
        for callback in self.callbacks:
            callback.log(log)

    def on_update_learner(self, learner_id: int) -> None:
        for callback in self.callbacks:
            callback.on_update_learner(learner_id)

    def on_update_target(self, learner_id: int) -> None:
        for callback in self.callbacks:
            callback.on_update_target(learner_id)
