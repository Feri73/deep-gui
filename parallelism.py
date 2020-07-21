import multiprocessing
from abc import ABC, abstractmethod
from queue import Empty
from typing import Callable, Optional

from utils import Config


class ThreadLocals:
    def __init__(self):
        self.thread = None
        self.collector = None
        self.new_weight = None

    def pop_and_run_next(self, *local_args, wait=False) -> None:
        self.thread.pop_and_run_next(*local_args, wait=wait)


class Thread(ABC):
    @abstractmethod
    def add_to_run_queue(self, func: Callable, *args) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        pass

    # this can only be accessed from the thread associated to this object
    @abstractmethod
    def pop_and_run_next(self, *local_args, wait=False) -> None:
        pass


# one problem with current framework is that each Thread has only 1 queue
class Process(Thread):
    def __init__(self, name: Optional[str], main_func: Optional[Callable], *args,
                 cfg: Config, main_process: bool = False):
        type = cfg['type']
        queue_size = cfg['queue_size']

        mp = multiprocessing.get_context(type)

        if main_process:
            assert name is None and main_func is None and len(args) == 0
        else:
            self.process = mp.Process(name=name, target=main_func, args=(*args, self))
        self.queue = mp.Queue(queue_size)

    def add_to_run_queue(self, func: Callable, *args) -> None:
        self.queue.put((func, args))

    def run(self) -> None:
        self.process.start()

    def pop_and_run_next(self, *local_args, wait=False) -> None:
        try:
            if wait:
                func, args = self.queue.get()
            else:
                func, args = self.queue.get_nowait()
            func(*local_args, *args)
        except Empty:
            pass
