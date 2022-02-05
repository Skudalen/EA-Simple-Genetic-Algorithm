import random
from typing import Callable
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

    def init_pop(self):
        self.indiv_len = self.params['indiv_len']
        self.pop_size = self.params['pop_size']
        rand_ints = [random.getrandbits(self.indiv_len) for x in range(self.pop_size)]
        pop = list(map(lambda x: np.binary_repr(x, self.indiv_len), rand_ints))
        return pop

    def evaluate_pop(self, pop):
        return self.fitness(pop, self.params)

    def do_terminate(self, pop_eval, gen_count):
        pass

    def select_parents(self, pop):
        # Stocastic
        self.num_parents = self.params['num_parents']
        pop_fitness = self.evaluate_pop(pop)
        self.fitness_dict = {pop[i]:pop_fitness[i] for i in range(self.pop_size)}
        fitness_sum = sum(pop_fitness)
        weights = pop_fitness / fitness_sum
        print('Weights used to select parents based on normalized fitness:', weights)
        parents = random.choices(pop, weights=weights, k=self.num_parents)
        return parents



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
