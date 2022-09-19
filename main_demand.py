from pyomo_solver_demand_model import *
import instances_reader 
from instances_reader import  *
from tqdm import tqdm 

if __name__ == "__main__":
    
    instances_path_items = [['MNL'],
                            ['32'],
                            ['2'],
                            ["250"],
                            ["100"]
                           ]
    
    #1: Instantiate the reader object
    reader = instances_reader.Reader()

    for element in itertools.product(*instances_path_items):

        #1: Get the instances path items
        demand, periods_, channels_, capacity, setup = element

        #3: Call the solver for each instance
        for instance_number in tqdm(range(1)):    
           
            #3.1: Read the instance
            T, periods,M,channels,capacities,capacity_used,production_costs,holding_costs,setup_costs, big_M, \
            markets_length, min_presence, A, B, LB, UB, inventory_ubs, message = reader.read_instance_lingo_format(demand,
                                                                                 channels_,periods_,capacity,setup,
                                                                                 instance_number + 1
                                                                                 )                                                                     
            
            ms, cpu_time, exec_time = solver_demand_single_product(T,periods, M, channels,  
                                                                   capacities,capacity_used, 
                                                                   production_costs,holding_costs, 
                                                                   setup_costs,big_M,markets_length,
                                                                   min_presence,A,B,LB,UB,inventory_ubs
                                                                   )
            