from audioop import cross
import random
from typing import Callable
from matplotlib.pyplot import step
import numpy as np

class Test:
    def __init__(self) -> None:
        pass

    def TEST_init_pop(self):
        pass

    def TEST_evaluate_pop(self, pop):
        return self.fitness(pop)

    def TEST_do_terminate(self, pop_eval, gen_count):
        pass

    def TEST_select_parents(self, pop):
        pass

    def TEST_make_offsprings(self, parents):
        pass

    def TEST_select_survivors(self, old_pop, offsprings, pop_eval, offs_eval):
        pass

    def TEST_plot_progress(self):
        pass

    def TEST_plot_end_result(self):
        pass

    def run(self):
        pass


class GA:
    def __init__(self, params, fitness:Callable, survival_selecter:Callable=None) -> None:
        self.params = params
        self.fitness = fitness
        self.survival_selecter = survival_selecter
        # Unpack params 
        self.indiv_len = self.params['indiv_len']
        self.pop_size = self.params['pop_size']
        self.p_c = self.params['p_c']
        self.num_parents = self.params['num_parents']
        self.p_m = self.params['p_m']

    def init_pop(self):
        rand_ints = [random.getrandbits(self.indiv_len) for x in range(self.pop_size)]
        pop = list(map(lambda x: np.binary_repr(x, self.indiv_len), rand_ints))
        return pop

    def evaluate_pop(self, pop):
        return self.fitness(pop, self.params)

    def do_terminate(self, pop_eval, gen_count):
        pass

    def select_parents(self, pop):
        # Stocastic
        pop_fitness = self.evaluate_pop(pop)
        self.fitness_dict = {pop[i]:pop_fitness[i] for i in range(self.pop_size)}
        fitness_sum = sum(pop_fitness)
        weights = pop_fitness / fitness_sum
        print('Weights used to select parents based on normalized fitness:', weights)
        parents = random.choices(pop, weights=weights, k=self.num_parents)
        return parents

    def crossover(self, parents):
        offsprings = []
        for i in range(0, self.num_parents-1, 2):
            #print(i)
            parent1 = parents[i]
            parent2 = parents[i+1]
            crosspoint = None
            for k in range(1, self.indiv_len-1):
                temp = random.choices([1, 0], weights=[self.p_c, 1 - self.p_c])
                #print(temp)
                if temp[0] == 1:
                    crosspoint = k
                    break
                #print(crosspoint)
            if crosspoint:
                child1 = parent1[:crosspoint] + parent2[crosspoint:]
                child2 = parent2[:crosspoint] + parent1[crosspoint:]
                offsprings.extend([child1, child2])
            else:
                offsprings.extend([parent1, parent2])
        return offsprings

    def mutate(self, offsprings:list):
        offsprings_mod = []
        for indiv in offsprings:
            new_indiv = indiv
            for i in range(len(indiv)):
                temp = random.choices([1, 0], weights=[self.p_m, 1 - self.p_m])
                if temp[0] == 1:
                    if new_indiv[i] == '0': 
                        new_indiv = indiv[:i] + '1' + indiv[i+1:]  
                    else: 
                        new_indiv = indiv[:i] + '0' + indiv[i+1:]
            offsprings_mod.append(new_indiv)
        return offsprings_mod

    def make_offsprings(self, parents):
        pass

    def select_survivors(self, old_pop, offsprings, pop_eval, offs_eval):
        if self.survival_selecter:
            return self.survival_selecter( old_pop, offsprings, pop_eval, offs_eval)
        else: 
            pass

    def plot_progress(self):
        pass

    def plot_end_result(self):
        pass

    def run(self):
        pop = self.init_pop() # numpy array (pop_size, 1)
        gen_count = 0
        pop_eval = self.evaluate_pop(pop)
        eval_log = {gen_count: pop_eval}
        while not self.do_terminate(pop_eval, gen_count):
            parents = self.select_parents(pop)
            offsprings = self.make_offsprings(parents)
            offs_eval = self.evaluate_pop(offsprings) 
            pop = self.select_survivors(pop, offsprings, pop_eval, offs_eval)

            gen_count += 1
            pop_eval = self.evaluate_pop(pop)
            eval_log[gen_count] = pop_eval
        
        return pop, eval_log
