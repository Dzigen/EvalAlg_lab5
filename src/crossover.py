from typing import List
import numpy as np
from numba.experimental import jitclass 
from numba import int32, float32
import random


spec = [
    ('cand_d', int32),
    ('genes_idx', int32[:]),
    ('pairs_num', int32),
    ('cross_opt_prob', float32),
    ('opt_prob', float32),
    ('genes_vals', int32[:])
]

@jitclass(spec)
class ChessCrossover:
    
    def __init__(self, dimension: int, crossover_pairs: int, cross_opt_prob: float) -> None:
        self.cand_d = dimension
        self.genes_vals = np.arange(self.cand_d, dtype=np.int32)
        self.genes_idx = np.arange(1, self.cand_d-1, dtype=np.int32)
        self.pairs_num = crossover_pairs
        self.opt_prob = cross_opt_prob

    def get_opt_seq(self, seq_len: int) -> List[int]:
        selected_opts = np.zeros(shape=seq_len, dtype=np.int32)
        for i in range(seq_len):
            selected_opts[i] = 1 if random.uniform(0,1 ) > self.opt_prob else 0

        return selected_opts

    def mate(self, population: List[List[int]]) -> List[List[int]]:

        selected_opts = self.get_opt_seq(self.pairs_num)
        #print(selected_opts)

        new_sols_size = (len(selected_opts[selected_opts == 0]) * 2) + (len(selected_opts[selected_opts == 1]) * 1)
        new_solutions = np.zeros(shape=(new_sols_size, self.cand_d), dtype=np.int32)
        #print(new_solutions.shape)
        
        pop_idxs = np.arange(len(population),dtype=np.int32)
        new_sols_idx = 0

        for idx in range(len(selected_opts)):
            idx1, idx2 = np.random.choice(pop_idxs, size=2, replace=False)
            sol1, sol2 = population[idx1], population[idx2]
            
            if selected_opts[idx] == 0:
                delim_idx = np.random.choice(self.genes_idx, size=1, replace=False)[0]
                
                new_sol1 = self.create_new_sol(delim_idx, sol1, sol2)
                new_sol2 = self.create_new_sol(delim_idx, sol2, sol1)

                new_solutions[new_sols_idx] = new_sol1
                new_solutions[new_sols_idx+1] = new_sol2
                new_sols_idx += 2
            
            elif selected_opts[idx] == 1:
                new_sol = np.zeros(shape=self.cand_d, dtype=np.int32)

                # Заполняем гены новой особи в тех позициях,
                # где гены родителей совпадают
                used_genes = []
                not_used_ids = []
                for i in range(self.cand_d):
                    if sol1[i] == sol2[i]:
                        new_sol[i] = sol1[i]
                        used_genes.append(sol1[i])
                    else:
                        not_used_ids.append(i)

                # Случайным образом заполняем оставшиеся гены
                # новой особи
                not_used_genes = np.array([gene for gene in self.genes_vals if gene not in used_genes], dtype=np.int32)
                not_used_genes =np.random.choice(not_used_genes, size=len(not_used_genes), replace=False)
                for i, gene in zip(not_used_ids, not_used_genes):
                    new_sol[i] = gene

                new_solutions[new_sols_idx] = np.copy(new_sol)
                new_sols_idx += 1

            else:
                raise ValueError
        
        return new_solutions
    
    def create_new_sol(self, delim_idx: int, sol1: List[int], sol2: List[int]) -> List[int]:
        new_sol = np.zeros(shape=self.cand_d, dtype=np.int32) 
        new_sol[:delim_idx] = sol1[:delim_idx]

        sol2_part = np.array([gene for gene in sol2[delim_idx:] if gene not in new_sol])
        new_sol[delim_idx:delim_idx + len(sol2_part)] = sol2_part

        sol1_part = np.array([gene for gene in sol1[delim_idx:] if gene not in new_sol])
        new_sol[delim_idx + len(sol2_part):delim_idx + len(sol2_part) + len(sol1_part)] = sol1_part

        return new_sol