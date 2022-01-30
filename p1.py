import json
import os
import pandas as pd

class Prep():

    def __init__(self) -> None:
        pass

    def set_params(self, params:dict):
        with open('params.json', 'w') as outfile:
            json.dump(params, outfile)

    def format_dataset(self):
        dataset_path = str(os.getcwd() + '/data/p1_dataset.txt')
        data_df = pd.read_csv(dataset_path, header=None, skiprows=[1993])
        data_df.to_csv('data/data.csv')
        #print(data_df)
        values_df = pd.read_csv(dataset_path, header=None, skiprows=1993)
        values_df.to_csv('data/values.csv')
        #print(values_df)



def main():

    pass

if __name__ == '__main__':
    
    prepper = Prep()
    prepper.format_dataset()

    params = {
        'indiv_length': 15,
        'pop_size': 100,
        'p_m': 0.1,
        'p_c': 0.6,

    }
    prepper.set_params(params)

    main()