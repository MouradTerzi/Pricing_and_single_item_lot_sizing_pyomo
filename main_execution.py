import pyomo_solver
from pyomo_solver import *
import instances_reader 
from instances_reader import  *
from xlwt import Workbook
#import tqdm
#from tqdm import tqdm 

def get_excel_results_path(production, demand, set_number, 
                           gen_protocole, periods,channels, 
                           capacity, setup):
    
    wb_partial_path = f'../Results/Market_share_model/{production}_production/{demand}/set_{set_number}/{gen_protocole}_P_{periods}_CH_{channels}/'
    if set_number == '2':
        return f'{wb_partial_path}{gen_protocole}_P_{periods}_CH_{channels_}_pyomo_results.xls'
    
    elif set_number == '3':
        return f'{wb_partial_path}cap_{capacity}_setup_{setup}/{gen_protocole}_P_{periods}_CH_{channels_}_pyomo_results_real_X.xls'


def create_sheet_file_for_periods_channels(wb, demand, 
                                          gen_protocole,
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
    
    instances_path_items = [['MNL'],['Keller'],['6'],['4','5']]
    set_number = '3'
    capacity = "700"
    setup = "900"
    production = "discrete"
    
    #1: Instantiate the reader object
    reader = instances_reader.READER()

    for element in itertools.product(*instances_path_items):

        #1: Create the excel file for each (set, periods, channels)
        wb = Workbook()
        
        #1: Get the instances' path items
        demand, gen_protocole, periods_, channels_ = element

        wb_path = get_excel_results_path(production, demand, set_number, gen_protocole,
                                            periods_,channels_, capacity, setup)

        
        #2: Create the results excel file
        sheet = create_sheet_file_for_periods_channels(wb, demand, 
                                                      gen_protocole,
                                                      periods_, channels_)
    
        #3: Call the solver for each instance
        for instance_number in range(10):    
           
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
                                                                            min_presence, A, B, LB, UB, inventory_ubs
                                                                            )
                """
                #3.3: Save the model     
                save_ms_model_and_results(ms, production, demand, set_number, 
                                        gen_protocole, periods_, 
                                        channels_, capacity, setup,
                                        instance_number + 1)
                """

                #3.4: Save the model 
                save_results_in_excel_file(wb, ms, periods_, 
                                        channels_, instance_number, 
                                        cpu_time, exec_time)
   
        wb.save(wb_path)   

            