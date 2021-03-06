from audioop import cross
import random
from typing import Callable
from matplotlib.pyplot import step
import numpy as np
from ipywidgets import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.special import entr


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
        self.max_gen = self.params['max_gen']

    def init_pop(self):
        rand_ints = [random.getrandbits(self.indiv_len) for x in range(self.pop_size)]
        pop = list(map(lambda x: np.binary_repr(x, self.indiv_len), rand_ints))
        return pop

    def evaluate_pop(self, pop):
        x, fitness, weights = self.fitness(pop, self.params)   # returns x-values list, fitness list, weights list
        return x, fitness, weights

    def do_terminate(self, pop_eval, gen_count):
        term = True if gen_count >= self.max_gen else False
        return term

    def select_parents(self, pop):
        # Stocastic
        _, _, weights = self.evaluate_pop(pop)
        #fitness_sum = sum(pop_fitness)
        #weights = np.divide(pop_fitness, fitness_sum)
        #print('\nWeights used to select parents based on normalized fitness:\n', weights)
        parents = random.choices(pop, weights=weights, k=self.num_parents)
        return parents

    def crossover(self, parents):
        offsprings = []
        for i in range(0, self.num_parents-1, 2):
            parent1 = parents[i]
            parent2 = parents[i+1]
            crosspoint = None
            for k in range(1, self.indiv_len-1):
                temp = random.choices([1, 0], weights=[self.p_c, 1 - self.p_c])
                if temp[0] == 1:
                    crosspoint = k
                    break
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
        offsprings = self.crossover(parents)
        offsprings_mod = self.mutate(offsprings)
        return offsprings_mod

    def select_survivors(self, parents, offsprings, pop_weights, off_weights, is_high_best):
        if self.survival_selecter:
            return self.survival_selecter(parents, offsprings, pop_weights, off_weights, is_high_best)
        else:   # Default: generational survival selection 
            return offsprings

    def plot_sine_generations(self, eval_log):
    
        x_sine = np.linspace(0, 128, 1000)
        y_sine = np.sin(x_sine)

        fig, axs = plt.subplots(figsize=(5,3))
        plt.subplots_adjust(bottom=0.35)
        plt.title("Population plot")
        plt.xlabel("x")
        plt.ylabel("sin(x)")
        plt.xlim(-1, 129)
        plt.ylim(-1.5, 1.5)
        line, = axs.plot(x_sine, y_sine)

        i = 1
        x = eval_log[i][0]
        y = eval_log[i][1]

        dots = axs.scatter(x, y, marker='o', color='orange')

        ax = plt.axes([0.25, 0.1, 0.55, 0.05])
        generation = Slider(ax, label='Generation', valmin=0, valmax=self.params['max_gen'], valstep=1, valinit=i)

        def update(val):
            gen = generation.val
            dots.set_offsets(np.c_[eval_log[gen][0], eval_log[gen][1]])

        generation.on_changed(update)

    def plot_end_result(self):
        pass

    def get_pop_entropy(self, pop):
        char_count = {k:0 for k in range(self.indiv_len)}
        for indiv in pop:
            for i, char in enumerate(list(indiv)):
                if char == '1':
                    char_count[i] += 1
        #[print(key, value) for key, value in char_count.items()]
        probs = [char_count.get(i)/self.pop_size for i in char_count.keys()]
        #entropy = - sum([p * np.log2(p) for p in probs])
        entropy = entr(probs).sum()
        return entropy

    def run(self):
        pop = self.init_pop() # numpy array (pop_size, 1)
        gen_count = 0
        # Store data, gen 0
        x, pop_fitness, pop_weights = self.evaluate_pop(pop)
        entropy = self.get_pop_entropy(pop)
        eval_log = {gen_count: [pop, pop_weights, x, pop_fitness, entropy]}
        # Evolution:
        while not self.do_terminate(pop_fitness, gen_count):
            parents = self.select_parents(pop)
            offsprings = self.make_offsprings(parents)
            _, off_fitness, off_weights = self.evaluate_pop(offsprings)
            pop = self.select_survivors(parents, offsprings, pop_fitness, off_fitness, self.params['is_high_best'])
            gen_count += 1
            # Store data, gen > 0
            x, pop_fitness, pop_weights = self.evaluate_pop(pop)
            entropy = self.get_pop_entropy(pop)
            eval_log[gen_count] = [pop, pop_weights, x, pop_fitness, entropy]

        print('Algorithm succsessfully executed')
        return eval_log
