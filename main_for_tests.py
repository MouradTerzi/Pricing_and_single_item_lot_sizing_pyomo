import pyomo_solver_pricing_model
from pyomo_solver_pricing_model import *
import instances_reader 
from instances_reader import  *
from xlwt import Workbook


if __name__ == "__main__":
<<<<<<< HEAD
    instances_path_items = [['MNL'],['Keller'],['6'],['2']]
    set_number = '3'
    capacity = "700"
    setup = "900"
    production = "discrete"
=======
    
    instances_path_items = [['MNL'],['Keller'],['6'],['2'],['DB_BC_BP']]
    set_ = "Small"
    set_number_ = '2'
>>>>>>> pricing_model
    
    #1: Instantiate the reader object
    reader = instances_reader.READER()
   
    for element in itertools.product(*instances_path_items):
        #1: Get the instances' path items
        demand, gen_protocole, periods_, channels_ = element
        
        #3: Call the solver for each instance
        for instance_number in range(16,17):    
           
            #3.1: Read the instance
            T, periods, M, channels, capacities, capacity_used, production_costs, holding_costs, setup_costs, big_M, \
<<<<<<< HEAD
            markets_length, min_presence, A, B, LB, UB, inventory_ubs, message = reader.read_instance_lingo_format(demand, gen_protocole,
                                                                                 set_number, channels_,periods_, 
                                                                                 capacity, setup, instance_number + 1
                                                                                 )
           
=======
            markets_length, min_presence, A, B, LB, UB, inventory_ubs, message = reader.read_instance_lingo_format(demand_, gen_protocole_,
                                    set_, set_number_, channels_,periods_, demand_params_, instance_number_ + 1)
            
>>>>>>> pricing_model
            reader.show_instance_data(T, periods, M, channels,
                          capacities, capacity_used, production_costs,
                          holding_costs, setup_costs, big_M, 
                          markets_length, min_presence, 
                          A, B, LB, UB)
            
<<<<<<< HEAD
            if message != "Instance not found":
            
                #3.2: Solve the instance
                ms, cpu_time, exec_time = solver_market_share_single_product(T, production, periods, M, channels, 
                                                                            demand, capacity, setup, set_number, 
                                                                            instance_number + 1, gen_protocole, 
                                                                            capacities, capacity_used, production_costs, 
                                                                            holding_costs, setup_costs, big_M, markets_length, 
                                                                            min_presence, A, B, LB, UB, inventory_ubs)
                print(f'Total execution time :{exec_time}')
                print(f'Total profit over the whole horizon is:')
                ms.obj.display()

                #3.3: Save the model    
                """ 
                save_ms_model_and_results(ms, production, demand, set_number, 
                                        gen_protocole, periods_, 
                                        channels_, capacity, setup,
                                        instance_number + 1)
                """
                
=======

            if message != "Instance not found":
               
                
                #3.2: Solve the instance
                prices_model, cpu_time, exec_time = solver_prices_single_product(T, periods, M, channels, 
                                                        set_, demand_, demand_params_, set_number_,
                                                        instance_number_ + 1, gen_protocole_, 
                                                        capacities, capacity_used, production_costs, 
                                                        holding_costs, setup_costs, 
                                                        big_M, markets_length, min_presence,
                                                        A, B, LB, UB, inventory_ubs)
                
               
                
            print("cpu time:",cpu_time)
            print("exec time:",exec_time)
            """
            #3.3: Save the model     
            save_ms_model_and_results(ms, demand_, set_, set_number_, 
                                demand_params_, gen_protocole_, periods_, 
                                channels_, instance_number_ + 1)
            """
        
                  
    
>>>>>>> pricing_model
            