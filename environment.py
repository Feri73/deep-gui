from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple

import numpy as np


class Environment(ABC):
    @abstractmethod
    def restart(self) -> None:
        pass

    @abstractmethod
    def is_finished(self) -> bool:
        pass

    # maybe i can change all numpy usage to tensorflow
    @abstractmethod
    def read_state(self) -> np.ndarray:
        pass

    @abstractmethod
    def act(self, action: Any, computation: Callable[[], Any]) -> Tuple[float, Any]:
        pass
