import json
import os
import pandas as pd
from ga import GA
import numpy as np

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
        #print(data_df)
        values_df = pd.read_csv(dataset_path, header=None, skiprows=1993)
        values_df.to_csv('data/values.csv')
        #print(values_df)

# HELP FUNCTIONS -----------------------------
def pop_to_real(pop):
    pop_real_val = list(map(lambda x: int(x, 2), pop))
    return pop_real_val

# INPUT FUNCTIONS ----------------------------
def sine_fitness(pop, params):
    max_sine_exp = params['max_sine_exp']
    indiv_len = params['indiv_len']
    scalar = 2 ** (max_sine_exp - indiv_len) 
    pop_real_val = np.power(pop_to_real(pop), scalar)   # fitting the values into [0,128] bit interval
    pop_fitness = list(map(lambda x: np.sin(x), pop_real_val))
    pop_fitness = [ x+1 for x in pop_fitness]
    return pop_fitness

def feature_fitness(pop):
    pass

def crowding_selection(pop):
    pass


def main(params):

    algorithm = GA(params, fitness=sine_fitness)
    #algorithm = GA(params, fitness=feature_fitness)
    #algorithm = GA(params, fitness=sine_fitness, survival_selecter=crowding_survival)
    #algorithm = GA(params, fitness=feature_fitness, survival_selecter=crowding_survival)
    
    # TEST
    

    #pop, eval_log = algorithm.run()

if __name__ == '__main__':
    
    params = {
        'indiv_len': 10,
        'pop_size': 10,     # Has to be even
        'num_parents':2,
        'p_m': 0.3,
        'p_c': 0.6,
        'max_sine_exp': 7
    }

    Prepper = Prep()
    Prepper.format_dataset()    
    Prepper.set_params(params)

    main(params)