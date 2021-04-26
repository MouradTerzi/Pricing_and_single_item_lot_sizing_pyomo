import pyomo_solver
from pyomo_solver import *
import instances_reader 
from instances_reader import  *


if __name__ == "__main__":
       
    instances_path_items = [['MNL'],['Keller'],['Small'],['2'],['2'],['1']]

    for element in itertools.product(*instances_path_items):
        
        #1: Get the instances' path items
        demand_, gen_protocole_,set_, periods_, channels_, instance_number_ = element
       
        #2: Read the instance
        reader = instances_reader.READER()
        T, periods, M, channels, capacities, capacity_used, production_costs, holding_costs, setup_costs, big_M, \
        markets_length, min_presence, A, B, LB, UB = reader.read_instance_lingo_format(demand_, gen_protocole_,
                                set_,channels_,periods_,
                                instance_number_)
        
        #3: Show the instance
        reader.show_instance_data(T, periods, M, channels, 
                                capacities, capacity_used, 
                                production_costs, holding_costs,
                                setup_costs, big_M, markets_length, 
                                min_presence, A, B, LB, UB )
        
        #4: Solve the instance
        ms = solver_market_share_single_product(T, periods, M, channels, 
                                               capacities, capacity_used, 
                                               production_costs, holding_costs, 
                                               setup_costs, big_M, markets_length, 
                                               min_presence, A, B, LB, UB)
                                               
        save_ms_model_and_results(ms, demand_, gen_protocole_,
                              set_, periods_, channels_, 
                              instance_number_)
