from typing import List
import numpy as np
from numba.experimental import jitclass 
from numba import float32, int32    # import the types

spec = [
    ('best_result', float32),
    ('best_solution',  int32[:]),
    ('result_age', int32),
    ('distances', float32[:,:]),
    ('dim', int32)
]

@jitclass(spec)
class ChessFitness:
    def __init__(self, dim: int) -> None:
        self.best_result = 100000000
        self.best_solution = np.zeros(dim, dtype=np.int32)
        self.result_age = 0
        self.dim = dim
        
    def calculate_fitness(self, solution: List[int]) -> float:
        diag_collisions = 0

        for i in range(len(solution)-1):
            for j in range(i+1, len(solution)):
                if np.abs(i - j) == np.abs(solution[i] - solution[j]):
                    diag_collisions += 1

        if self.best_result > diag_collisions:
            self.best_result = diag_collisions
            self.best_solution = np.copy(solution)
            self.result_age = 0

        return diag_collisions
