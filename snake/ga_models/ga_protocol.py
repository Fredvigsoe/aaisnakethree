from typing import Protocol
import numpy as np


class GAModel(Protocol):
    def __init__(self):
        pass

    def update(self, obs) -> (int, int):
        pass

    def mutate(self, mutation_rate) -> None:
        pass

    @property
    def DNA(self) -> np.array:
        pass
