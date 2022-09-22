from pyomo_solver_prices_model import *
import instances_reader 
from instances_reader import  *
from tqdm import tqdm
import pandas as pd

if __name__ == '__main__':

    instances_path_items = [['MNL'],
                            ['2'],
                            ['32'],
                            ["250"],
                            ["900"]
                           ]
    #1: Instantiate the reader object
    reader = instances_reader.Reader()
    for element in itertools.product(*instances_path_items):
        #1: Get the instances path items
        demand,M,T,capacity,setup = element
        
        results = {}
        results['Instance class'] = list()
        results['Instance number'] = list()
        results['OA-objective'] = list()
        results['OA-cpu-time'] = list()

        results_folder_path = f'../Results/Prices_model/{demand}/P_{T}_CH_{M}_cap_{capacity}_setup_{setup}/'
        class_results_path = f'{results_folder_path}P_{T}_CH_{M}_cap_{capacity}_setup_{setup}_prices_results.xlsx'
        for instance_number in tqdm(range(10)):    
           
            #2.1: Read the instance
            instance = reader.read_instance_lingo_format(demand,M,T,capacity,
                                                         setup,instance_number)      
            
            dm, cpu_time = solve_pricing_and_lot_sizing_multi_channel(instance,demand,capacity,setup,instance_number)
            results['Instance class'].append(f'{M,T,capacity,setup}')
            results['Instance number'].append(f'{instance_number+1}')
            results['OA-objective'].append(f'{round(value(dm.obj),2)}')
            results['OA-cpu-time'].append(f'{round(cpu_time,2)}')

            instance_results_path = f'{results_folder_path}Instance_{instance_number+1}_P_{T}_CH_{M}_cap_{capacity}_setup_{setup}_results.xlsx'
            save_model_results(dm,instance_results_path,instance.channels,
                               instance.periods,cpu_time)

        df = pd.DataFrame(data = results)
        df.to_excel(f'{class_results_path}')
            
        
        
