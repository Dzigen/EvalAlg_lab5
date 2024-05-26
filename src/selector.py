from typing import List
from numba.experimental import jitclass 
from numba import int32
import numpy as np

spec = [
    ('pop_size', int32),
    ('cand_dim', int32)
]

@jitclass(spec)
class ChessSelector:
    def __init__(self, population_size: int, dim: int) -> None:
        self.pop_size = population_size
        self.cand_dim = dim

    def filter_population(self, population: List[List[int]], fitnesses: List[float]) -> List[List[int]]:
        sorted_idx = list(map(lambda v: v[0], sorted(list(zip(list(range(len(population))), fitnesses)), key=lambda v: v[1])))

        filtered_pop = np.zeros(shape=(self.pop_size, self.cand_dim), dtype=np.int32)
        for i in range(len(filtered_pop)):
            filtered_pop[i] = np.copy(population[sorted_idx[i]])

        return filtered_pop