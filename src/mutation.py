from typing import List
import random
import math
import numpy as np
from numba import int32, float32    # import the types
from numba.experimental import jitclass 

spec = [
    ('cand_d', int32),
    ('gene_ids', int32[:]),
    ('mut_mode', int32),
    ('opt_prob', float32),
    ('mutation_cand_percent', float32)
]

@jitclass(spec)
class ChessMutation:
    def __init__(self, dimension: int, mut_mode: int, opt_prob: int, mut_cand_percent: float) -> None:
        self.cand_d = dimension
        self.gene_ids = np.arange(self.cand_d, dtype=np.int32)
        self.mut_mode = mut_mode
        self.opt_prob = opt_prob
        self.mutation_cand_percent = mut_cand_percent

    def shuffle(self, pop):
        shuffled_idxs = np.random.choice(np.arange(len(pop)), size=len(pop), replace=False)

        shufled_pop = np.zeros(shape=(len(pop), self.cand_d), dtype=np.int32)
        for i, idx in enumerate(shuffled_idxs):
            shufled_pop[i] = pop[idx]

        return shufled_pop 

    def apply(self, population: List[List[int]]) -> List[List[int]]:
        mut_amount = math.floor(len(population) * self.mutation_cand_percent)

        if self.mut_mode == 0:
            mutated_pop = np.zeros((len(population) + mut_amount, self.cand_d), dtype=np.int32)
            start_idx = len(population)
        elif self.mut_mode == 1:
            mutated_pop = np.zeros((len(population), self.cand_d), dtype=np.int32)
            start_idx = 0
        else:
            raise ValueError
        
        mutated_pop[:len(population)] = self.shuffle(population)

        for c_idx in range(start_idx, len(mutated_pop)):
            # Сохраняется прошлое состояние генов мутируемой особи
            if self.mut_mode == 0:
                mutated_pop[c_idx] = mutated_pop[c_idx - len(population)]
            # Изменение генов особи без сохранения
            # предыдущего состояния её генов
            elif self.mut_mode == 1:
                pass
            else:
                raise ValueError
 
            #
            l_border_idx, r_border_idx = np.random.choice(self.gene_ids, size=2, replace=False)

            # Инвертируем два гена
            if random.uniform(0, 1) < self.opt_prob:
                l_node, r_node = mutated_pop[c_idx][l_border_idx], mutated_pop[c_idx][r_border_idx]

                mutated_pop[c_idx][l_border_idx] = r_node
                mutated_pop[c_idx][r_border_idx] = l_node

            # Нвертируем весь интервал генов
            else:
                reversed_genes =  mutated_pop[c_idx][r_border_idx::-1] if l_border_idx == 0 else mutated_pop[c_idx][r_border_idx:l_border_idx-1:-1]
                
                if r_border_idx == (self.cand_d - 1):
                    mutated_pop[c_idx][l_border_idx:] = reversed_genes
                else:
                    mutated_pop[c_idx][l_border_idx:r_border_idx+1] = reversed_genes
            
        return mutated_pop
