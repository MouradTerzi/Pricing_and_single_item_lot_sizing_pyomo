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
    sheet.write(0,8,"MS_pyomo_cpu_Time(s)")
    sheet.write(0,10,"MS_pyomo_total_exec_Time (s)")

    return sheet

def save_results_in_excel_file(wb,model,periods_, channels_,
                               instance_number_, cpu_time, exec_time):
    
    sheet.write(instance_number_ + 1, 0, str(periods_))
    sheet.write(instance_number_ + 1, 2, str(channels_))
    sheet.write(instance_number_ + 1, 4, str(instance_number_ + 1))
    
    try:
        sheet.write(instance_number_ + 1, 6, str(value(ms.obj)))
        
    except: 
        sheet.write(instance_number_ + 1, 6, "None")

    sheet.write(instance_number_ + 1, 8, str(cpu_time))
    sheet.write(instance_number_ + 1, 10, str(exec_time))
    
    return 

if __name__ == "__main__":
    
    instances_path_items = [['MNL'],['Keller'],['16','32','64','128','256','512','1024'],['2','3','4','5'],['DB_BC_BP']]
    set_ = "Large"
    set_number_ = '2'
    
    #1: Instantiate the reader object
    reader = instances_reader.READER()


    for element in itertools.product(*instances_path_items):

        #1: Create the excel file for each (set, periods, channels)
        wb = Workbook()
        
        #1: Get the instances' path items
        demand_, gen_protocole_, periods_, channels_, demand_params_= element
        wb_path = f'../Results/Market_share_model/{demand_}/{set_}/{gen_protocole_}_P_{periods_}_CH_{channels_}_set_{set_number_}/{gen_protocole_}_P_{periods_}_CH_{channels_}_set_{set_number_}_pyomo_results.xls'
        
        #2: Create the results excel file
        sheet = create_sheet_file_for_periods_channels(wb, demand_, 
                                                      gen_protocole_, set_,
                                                      periods_, channels_)
        
        #3: Call the solver for each instance
        for instance_number_ in range(20):    
           
            #3.1: Read the instance
            T, periods, M, channels, capacities, capacity_used, production_costs, holding_costs, setup_costs, big_M, \
            markets_length, min_presence, A, B, LB, UB, inventory_ubs, message = reader.read_instance_lingo_format(demand_, gen_protocole_,
                                    set_, set_number_, channels_,periods_, demand_params_, instance_number_ + 1)

            if message != "Instance not found":
                
                #3.2: Solve the instance
                ms, cpu_time, exec_time = solver_market_share_single_product(T, periods, M, channels, 
                                                        set_, demand_, demand_params_, 
                                                        set_number_, instance_number_ + 1, gen_protocole_, 
                                                        capacities, capacity_used, production_costs, 
                                                        holding_costs, setup_costs, 
                                                        big_M, markets_length, min_presence,
                                                        A, B, LB, UB, inventory_ubs)
            
                #3.3: Save the model     
                save_ms_model_and_results(ms, demand_, set_, set_number_, 
                                    demand_params_, gen_protocole_, periods_, 
                                    channels_, instance_number_ + 1)
                
                #3.4: Save the model 
                save_results_in_excel_file(wb, ms, periods_, 
                                        channels_, instance_number_, 
                                        cpu_time, exec_time)
                
        wb.save(wb_path)   
    
            