import os
import pickle
from typing import Callable, Dict, Any, List

Config = Dict[str, Any]


def is_windows() -> bool:
    return os.name == 'nt'


def run_parallel_command(command: str) -> None:
    if is_windows():
        command = f'start /min {command}'
    else:
        command = f'{command} &'
    os.system(command)


class MemVariable:
    def __init__(self, init_value: Callable[[], Any], memory_size: int = 1):
        self.value = init_value()
        self.init_value = init_value
        self.memory_size = memory_size
        self.reset_archive()

    def archive(self) -> None:
        self.memory = ([self.value] + self.memory)[:self.memory_size]
        self.value = self.init_value()

    def last_value(self, index: int = 0) -> Any:
        return self.memory[index]

    def reset_archive(self) -> None:
        self.memory = []

    def has_archive(self) -> bool:
        return len(self.memory) > 0

    def reset_value(self) -> None:
        self.value = self.init_value()


class MemList(MemVariable):
    def __init__(self, content: List[MemVariable]):
        self.content = content
        super().__init__(lambda: [c.value for c in self.content])

    def archive(self) -> None:
        for c in self.content:
            c.archive()
        self.value = self.init_value()

    def last_value(self, index: int = 0) -> List[Any]:
        return [c.memory[index] for c in self.content]

    def reset_archive(self) -> None:
        for c in self.content:
            c.reset_archive()

    def has_archive(self) -> bool:
        return sum([c.has_archive() for c in self.content]) == len(self.content)

    def reset_value(self) -> None:
        for c in self.content:
            c.reset_value()


def dump_obj(obj: Any, file_path: str):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def load_obj(file_path: str) -> Any:
    with open(file_path, 'rb') as f:
        return pickle.load(f)
