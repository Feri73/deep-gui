import multiprocessing
from abc import ABC, abstractmethod
from queue import Empty
from typing import Callable, Optional, Tuple, Union

from utils import Config


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

    def pop_next(self, wait=False) -> Union[Tuple[Callable, list], Tuple[None, None]]:
        try:
            if wait:
                func, args = self.queue.get()
            else:
                func, args = self.queue.get_nowait()
            return func, args
        except Empty:
            return None, None

    def pop_and_run_next(self, *local_args, wait=False) -> None:
        func, args = self.pop_next(wait)
        if func is not None:
            func(*local_args, *args)
