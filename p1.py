import json
import os
import pandas as pd
from ga import *
import numpy as np
import LinReg

class Prep():

    def __init__(self) -> None:
        pass

    def set_params(self, params:dict):
        with open('data/params.json', 'w') as outfile:
            json.dump(params, outfile)

    def format_dataset(self):
        dataset_path = str(os.getcwd() + '/data/p1_dataset.txt')
        data_df = pd.read_csv(dataset_path, header=None, skiprows=[1993])
        data_df.to_csv('data/data.csv')
        values_df = pd.read_csv(dataset_path, header=None, skiprows=1993)
        values_df.to_csv('data/values.csv')

# HELP FUNCTIONS -----------------------------
def pop_to_real(pop):
    pop_real_val = list(map(lambda x: int(x, 2), pop))
    return pop_real_val



# INPUT FUNCTIONS ----------------------------
def sine_fitness(pop, params):
    max_sine_exp = params['max_sine_exp']
    indiv_len = params['indiv_len']
    scalar = 2 ** (max_sine_exp - indiv_len)
    pop_real_val = np.multiply(pop_to_real(pop), scalar)   # fitting the values into [0,128] bit interval
    #   Get fitness
    pop_fitness = list(map(lambda x: np.sin(x), pop_real_val))
    if params['sine_constraint']:
        pop_fitness = list(map(lambda x: np.sin(x) if x >= 5 and x <= 10 else -1.25, pop_real_val))
    #   Make weights
    min_, max_ = min(pop_fitness), max(pop_fitness)
    weights = list(map(lambda x: (x-min_)/(max_-min_) if (max_-min_) > 0 else 1, pop_fitness))
    weights = np.divide(weights, sum(weights))
    return pop_real_val, pop_fitness, weights

def feature_fitness(pop, params):
    linreg = LinReg.LinReg()
    data_df = pd.read_csv('data/data.csv', index_col=[0])
    values_df = pd.read_csv('data/values.csv', index_col=[0])
    rmse_errors = []
    for indiv in pop:
        x = linreg.get_columns(data_df, indiv)
        y = linreg.get_columns(values_df, indiv)
        feats = y.shape[1]
        x = x.reshape(feats, x.shape[0])
        y = y.reshape(feats, y.shape[0])
        error = linreg.get_fitness(x, y)
        rmse_errors.append(error)
    # Scale from low-best to high-best
    max_error = max(rmse_errors)
    rev_error = list(map(lambda x: (max_error - x)**2, rmse_errors))
    min_, max_ = min(rev_error), max(rev_error)
    weights = list(map(lambda x: (x-min_)/(max_-min_) if (max_-min_) > 0 else 1, rev_error))
    weights = np.divide(weights, sum(weights))
    _ = None
    return _, rmse_errors, weights

def crowding_selection(parents:list, offsprings:list, pop_weights:list, off_weights:list, is_high_best:bool):
    
    def get_diff(a, b):
        c = int(a, 2) ^ int(b, 2)
        return "{0:b}".format(c).count('1')

    pop_size = len(offsprings)
    new_pop = []
    for i in range(0, pop_size-1, 2):
        p1 = parents[i]
        p2 = parents[i+1]
        o1 = offsprings[i]
        o2 = offsprings[i+1]
        if get_diff(p1, o1) + get_diff(p2, o2) < get_diff(p1, o2) + get_diff(p2, o1):
            if is_high_best:
                chosen_1 =  p1 if pop_weights[i] > off_weights[i] else o1
                chosen_2 =  p2 if pop_weights[i+1] > off_weights[i+1] else o1
            else:
                chosen_1 =  p1 if pop_weights[i] < off_weights[i] else o1
                chosen_2 =  p2 if pop_weights[i+1] < off_weights[i+1] else o1
            new_pop.extend([chosen_1, chosen_2])
        else:
            if is_high_best:
                chosen_1 =  p1 if pop_weights[i] > off_weights[i+1] else o2
                chosen_2 =  p2 if pop_weights[i+1] > off_weights[i] else o1
            else:
                chosen_1 =  p1 if pop_weights[i] < off_weights[i+1] else o2
                chosen_2 =  p2 if pop_weights[i+1] < off_weights[i] else o1
            new_pop.extend([chosen_1, chosen_2])
    return new_pop


def main(params):

    algorithm = GA(params, fitness=sine_fitness)
    #algorithm = GA(params, fitness=feature_fitness)
    #algorithm = GA(params, fitness=sine_fitness, survival_selecter=crowding_survival)
    #algorithm = GA(params, fitness=feature_fitness, survival_selecter=crowding_survival)

    #eval_log = algorithm.run()

if __name__ == '__main__':
    
    params = {
        'indiv_len': 10,
        'pop_size': 8,              # Has to be even
        'num_parents':8,            # Has to be <= pop_size
        'p_m': 0.1,
        'p_c': 0.6,
        'max_sine_exp': 7,          # 2^7 -> [0,128]
        'max_gen': 10, 
        'sine_constraint': False
    }

    Prepper = Prep()
    Prepper.format_dataset()    
    Prepper.set_params(params)

    main(params)