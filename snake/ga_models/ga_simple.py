import random
from typing import Protocol, Tuple, List, Sequence
import numpy
import numpy as np
from ga_models.ga_protocol import GAModel
from ga_models.activation import sigmoid, tanh, softmax


class SimpleModel(GAModel):
    def __init__(self, *, dims: Tuple[int, ...]):
        assert len(dims) >= 2, 'Error: dims must be two or higher.'
        self.dims = dims
        self.DNA = []
        for i, dim in enumerate(dims):
            if i < len(dims) - 1:
                self.DNA.append(np.random.rand(dim, dims[i+1]))

    def update(self, obs: Sequence) -> Tuple[int, ...]:
        x = obs
        for i, layer in enumerate(self.DNA):
            if not i == 0:
                x = tanh(x)
            x = x @ layer
        return softmax(x)

    def action(self, obs: Sequence):
        return self.update(obs).argmax()

    def mutate(self, mutation_rate) -> None:
        if random.random() < mutation_rate:
            random_layer = random.randint(0, len(self.DNA) - 1)
            row = random.randint(0, self.DNA[random_layer].shape[0] - 1)
            col = random.randint(0, self.DNA[random_layer].shape[1] - 1)
            self.DNA[random_layer][row][col] = random.uniform(-1, 1)

    def __add__(self, other):
        baby_DNA = []
        for mom, dad in zip(self.DNA, other.DNA):
            if random.random() > 0.5:
                baby_DNA.append(mom)
            else:
                baby_DNA.append(dad)
        baby = type(self)(dims=self.dims)
        baby.DNA = baby_DNA
        return baby

    def DNA(self):
        return self.DNA
