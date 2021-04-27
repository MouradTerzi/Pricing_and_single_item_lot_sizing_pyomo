import itertools
import instance_feasibility 
from instance_feasibility import *
import instances_reader 
from instances_reader import  *

if __name__ == "__main__":
       
    instances_path_items = [['MNL'],['Keller'],['Small'],['2'],['2']]
    
    #1: Instantiate the reader object
    reader = instances_reader.READER()

    for element in itertools.product(*instances_path_items):
        
        #1: Get the instances' path items
        demand_, gen_protocole_,set_, periods_, channels_, = element
        wb_path = f'../Results/{demand_}/{gen_protocole_}/{set_}/keller_set_1_pyomo_results_.xls'
        
        #3: Call the solver for each instance
        for instance_number_ in range(1):    
      
            #3.1: Read the instance
            T, periods, M, channels, capacities, capacity_used, production_costs, holding_costs, setup_costs, big_M, \
            markets_length, min_presence, A, B, LB, UB = reader.read_instance_lingo_format(demand_, gen_protocole_,
                                    set_,channels_,periods_,
                                    instance_number_ + 1)
           
            
            check_instance_feasibility(channels, periods,
                                       min_presence, A, B,
                                       LB, UB)
            