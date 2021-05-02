import pyomo_solver
from pyomo_solver import *
import instances_reader 
from instances_reader import  *



if __name__ == "__main__":
    
    instances_path_items = [['MNL'],['Keller'],['2048'],['2'],['DB_BC_BP']]
    set_ = "Large"
    set_number_ = '2'
    
    #1: Instantiate the reader object
    reader = instances_reader.READER()


    for element in itertools.product(*instances_path_items):
        
        #1: Get the instances' path items
        demand_, gen_protocole_, periods_, channels_, demand_params_= element
        #wb_path = f'../Results/{demand_}/{set_}/{gen_protocole_}_set_{set_number_}_pyomo_results.xls'
        
        
        #3: Call the solver for each instance
        for instance_number_ in range(1):    
           
            #3.1: Read the instance
            T, periods, M, channels, capacities, capacity_used, production_costs, holding_costs, setup_costs, big_M, \
            markets_length, min_presence, A, B, LB, UB, inventory_ubs, message = reader.read_instance_lingo_format(demand_, gen_protocole_,
                                    set_, set_number_, channels_,periods_, demand_params_, instance_number_ + 1)

            if message != "Instance not found":
               
                
                #3.2: Solve the instance
                ms, exec_time = solver_market_share_single_product(T, periods, M, channels, 
                                                        set_, demand_, demand_params_, set_number_,
                                                        instance_number_ + 1, gen_protocole_, 
                                                        capacities, capacity_used, production_costs, 
                                                        holding_costs, setup_costs, 
                                                        big_M, markets_length, min_presence,
                                                        A, B, LB, UB, inventory_ubs)
                
            
                
            
            """
            #3.3: Save the model     
            save_ms_model_and_results(ms, demand_, set_, set_number_, 
                                demand_params_, gen_protocole_, periods_, 
                                channels_, instance_number_ + 1)
            """
        
                  
    
            