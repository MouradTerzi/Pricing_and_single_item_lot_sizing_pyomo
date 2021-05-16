import pyomo_solver
from pyomo_solver import *
import instances_reader 
from instances_reader import  *
from xlwt import Workbook


if __name__ == "__main__":
    instances_path_items = [['MNL'],['Keller'],['6'],['2']]
    set_number = '3'
    capacity = "700"
    setup = "900"
    production = "discrete"
    
    #1: Instantiate the reader object
    reader = instances_reader.READER()
   
    for element in itertools.product(*instances_path_items):
        #1: Get the instances' path items
        demand, gen_protocole, periods_, channels_ = element
        
        #3: Call the solver for each instance
        for instance_number in range(1):    
           
            #3.1: Read the instance
            T, periods, M, channels, capacities, capacity_used, production_costs, holding_costs, setup_costs, big_M, \
            markets_length, min_presence, A, B, LB, UB, inventory_ubs, message = reader.read_instance_lingo_format(demand, gen_protocole,
                                                                                 set_number, channels_,periods_, 
                                                                                 capacity, setup, instance_number + 1
                                                                                 )
            if message != "Instance not found":
            
                #3.2: Solve the instance
                ms, cpu_time, exec_time = solver_market_share_single_product(T, production, periods, M, channels, 
                                                                            demand, capacity, setup, set_number, 
                                                                            instance_number + 1, gen_protocole, 
                                                                            capacities, capacity_used, production_costs, 
                                                                            holding_costs, setup_costs, big_M, markets_length, 
                                                                            min_presence, A, B, LB, UB, inventory_ubs)
                
              