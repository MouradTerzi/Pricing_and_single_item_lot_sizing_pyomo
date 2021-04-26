import pyomo_solver
from pyomo_solver import *
import instances_reader 
from instances_reader import  *
from xlwt import Workbook


def create_sheet_file_for_periods_channels(wb, demand_, 
                                          gen_protocole_, set_,
                                          periods_, channels_):

    sheet = wb.add_sheet(f'{periods_}_{channels_}')
    sheet.write(0,0,"Periods")
    sheet.write(0,2,"Channels")
    sheet.write(0,4,"Instance number")
    sheet.write(0,6,"MS_Pyomo_Obj")
    sheet.write(0,8,"MS_pyomo_Time (s)")

    return sheet

def save_results_in_excel_file(wb,model,periods_, channels_,
                               instance_number_, exec_time):
    
    sheet.write(instance_number_ + 1, 0, str(periods_))
    sheet.write(instance_number_ + 1, 2, str(channels_))
    sheet.write(instance_number_ + 1, 4, str(instance_number_ + 1))
    
    try:
        sheet.write(instance_number_ + 1, 6, str(value(ms.obj)))
        
    except: 
        sheet.write(instance_number_ + 1, 6, "None")

    sheet.write(instance_number_ + 1, 8, str(exec_time))
    
    return 

if __name__ == "__main__":
       
    instances_path_items = [['MNL'],['Keller'],['Large'],['64'],['2']]
    
    #3: Instantiate the reader object
    reader = instances_reader.READER()
    #1: Create the excel file for the restuls

    wb = Workbook()
    for element in itertools.product(*instances_path_items):
        
        #1: Get the instances' path items
        demand_, gen_protocole_,set_, periods_, channels_, = element
        wb_path = f'../Results/{demand_}/{gen_protocole_}/{set_}/keller_set_1_pyomo_results.xls'
        
        #2: Create the results excel file
        sheet = create_sheet_file_for_periods_channels(wb, demand_, 
                                                      gen_protocole_, set_,
                                                      periods_, channels_)
        
        #4: Call the solver for each instance
        for instance_number_ in range(1):    
      
            #4.1: Read the instance
            T, periods, M, channels, capacities, capacity_used, production_costs, holding_costs, setup_costs, big_M, \
            markets_length, min_presence, A, B, LB, UB = reader.read_instance_lingo_format(demand_, gen_protocole_,
                                    set_,channels_,periods_,
                                    instance_number_ + 1)
           
            #4.2: Solve the instance
            ms, exec_time = solver_market_share_single_product(T, periods, M, channels, 
                                                    set_, demand_, instance_number_ + 1,
                                                    gen_protocole_, capacities, 
                                                    capacity_used, production_costs, 
                                                    holding_costs, setup_costs, 
                                                    big_M, markets_length, min_presence,
                                                    A, B, LB, UB)
        
            #4.3: Save the model     
            save_ms_model_and_results(ms, demand_, gen_protocole_,
                                set_, periods_, channels_, 
                                instance_number_ + 1)
            
            save_results_in_excel_file(wb, ms, periods_, 
                                      channels_, instance_number_, 
                                      exec_time)

                                              
        wb.save(wb_path)   

            