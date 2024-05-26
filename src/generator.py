from typing import List, Set, Tuple
import numpy as np
from numba.experimental import jitclass 
from numba import int32, float32     # import the types

spec = [
    ('dim', int32),
    ('unique_genes', int32)
]

@jitclass(spec)
class ChessGenerator:
    def __init__(self, problem_size: int) -> None:
        self.unique_genes = problem_size
        self.dim = problem_size

    def get_population(self, pop_size: int) -> List[List[int]]:
        population = np.zeros(shape=(pop_size, self.dim), dtype=np.int32)
        places = np.arange(self.unique_genes, dtype=np.int32)

        cur_idx = 0
        while cur_idx < pop_size:
            solution = np.random.choice(places, size=self.dim, replace=False)
            
            population[cur_idx] = solution
            cur_idx += 1
            
        return population
    