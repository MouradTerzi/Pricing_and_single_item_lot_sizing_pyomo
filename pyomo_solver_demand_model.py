import  sys 
import numpy as np
import itertools
import time
import timeit
import pandas as pd
import pyomo
from pyomo.environ import *
import xlsxwriter

import instances_reader
from instances_reader import *

"""
    # Sets methods
"""
def channels_periods_set_init(model):
    
    return [(channel, period) for channel in channels for period in periods]

def channels_set_init(model,channel):
    
    return [channels.index(channel) + 1]

def create_instance_sets(model):
    
    """
      1. Periods set 
      2. Channels set
      3. Channels and periods set
      
    """
    model.P = Set(initialize = periods)
    model.CH = Set(channels, initialize = channels_set_init)
    model.CHP= Set(initialize = channels_periods_set_init)
    
    return

"""
    # Logistics Params initialization
"""

def initialize_production_costs(model, period):
    return production_costs[period - 1]

def initialize_holding_costs(model,period):
    return holding_costs[period - 1]

def initialize_setup_costs(model, period):
    return setup_costs[period - 1]

def initialize_capacity_per_period(model, period):
    return capacities[period - 1]

def initialize_capacity_used_per_period(model, period):
    return capacity_used[period - 1]

def logistic_params_initialization(model):
    
    model.capacity_used = Param(model.P, initialize = initialize_capacity_used_per_period, within = PositiveReals)
    model.capacities = Param(model.P, initialize = initialize_capacity_per_period, within = PositiveReals)
    model.Prod_cost = Param(model.P, initialize = initialize_production_costs, within = PositiveReals)
    model.Hold_cost = Param(list(model.P)[:-1], initialize = initialize_holding_costs, within = PositiveReals)
    model.Setup_cost = Param(model.P, initialize = initialize_setup_costs, within = PositiveReals)
    
    return 

"""
   # Markets params initialization
"""
def initialize_minimum_markets_presence(model, channel):
    return min_presence[list(model.CH).index(channel)]

def initialize_markets_length(model, period):
    return markets_length[period - 1]

def initialize_demand_params_a(model, channel, period):
    return A[(channel,period)]

def initialize_demand_params_b(model, channel, period):
    return B[(channel,period)]

def initialize_lbs(model, channel, period):
    return LB[(channel,period)]

def initialize_ubs(model, channel, period):
    return UB[(channel,period)]

def market_data_initialization(model):
    
    model.min_presence = Param(channels, initialize = initialize_minimum_markets_presence, within = PositiveReals)
    model.Mars_len = Param(model.P, initialize = initialize_markets_length, within = PositiveReals)
    model.A = Param(model.CHP, initialize = initialize_demand_params_a, within = PositiveReals)
    model.B = Param(model.CHP, initialize = initialize_demand_params_b, within = NegativeReals)
    model.LB = Param(model.CHP, initialize = initialize_lbs, within = PositiveReals)
    model.UB = Param(model.CHP, initialize = initialize_ubs, within = PositiveReals)
    
    return 

"""
   # Variables creation
"""    

def X_bounds(model, period):
    return 0, capacities[period - 1]/capacity_used[period - 1]

def I_upper_bounds(model, period):
    return (0,inventory_ubs[period - 1])

def decision_variables_creation(model):
    
    model.d_mt = Var(model.CHP, within = PositiveReals, initialize = 0.25)
    model.d_o = Var(model.P, within = PositiveReals, initialize = 0.25)
    model.X = Var(model.P, within = NonNegativeReals, initialize = 0)
    model.I = Var(list(model.P)[:-1], bounds= I_upper_bounds, within = NonNegativeReals, initialize = 0)
    model.Y = Var(model.P, within = Binary)
    
    return 
"""
  # Objective function
"""

def compute_profit_mt(model,channel, period):
    
    di_mt = model.d_mt[(channel,period)]/model.B[(channel, period)]
    g_mt =  log(model.d_mt[(channel,period)]/model.d_o[period]) - model.A[(channel,period)]
    
    return di_mt*g_mt 

def compute_objective_function(model):
    
    model.profit = Expression(expr = sum(compute_profit_mt(model, channel, period) \
    for (channel, period) in model.CHP))
    
    model.total_costs = Expression(expr = summation(model.Prod_cost,model.X) \
    + summation(model.Hold_cost,model.I) + summation(model.Setup_cost,model.Y))
    
    model.obj = Objective(expr = model.profit.expr - model.total_costs.expr, sense = maximize)

    return 
"""
  # Add Business constraints
"""
def minimum_presence_mt(model, channel, period):
    
    return model.min_presence[channel]*sum(model.d_mt[(m,period)] for m in model.CH) \
    <= model.d_mt[(channel,period)]

def sum_demand_equal_to_market_length(model, period):
    return sum([model.d_mt[(m,period)] for m in model.CH]) + model.d_o[period] == model.Mars_len[period]

def add_business_constraints(model):
    
    model.presence_constraints = Constraint(model.CHP, rule = minimum_presence_mt)
    model.demand_equal_to_market_length_constraints = Constraint(model.P, rule = sum_demand_equal_to_market_length)
"""
  # Add Logistic constraints
"""

def production_limit_constraint(model,period):
    return model.X[period] <= model.capacities[period]/model.capacity_used[period]

def inventory_between_two_and_t_1(model,period):   
    return sum(model.d_mt[m,period] for m in model.CH) \
    + model.I[period] - model.I[period -1] - model.X[period] == 0

def setup_constraint_per_period(model, period):
    return model.X[period] <= big_M*model.Y[period]

def add_logistics_constraints(model):
    
    #1: Production capacity limits
    model.production_limits = Constraint(model.P, rule = production_limit_constraint) 
    
    #2: Inventory for the first period
    model.inventory_first_period = Constraint(rule = sum(model.d_mt[m,1] \
    for m in model.CH) + model.I[1] - model.X[1] == 0)
    
    #3: Inventory for the periods between 2 and T - 1
    model.inventory_2_to_T_1 = Constraint(list(model.P)[1:-1], rule = inventory_between_two_and_t_1)
    
    #4: Inventory for t == T
    model.inventory_end_period = Constraint(rule = sum(model.d_mt[m,T] \
    for m in model.CH) - model.I[T-1] - model.X[T] == 0)
    
    #5: Setup constraints
    model.setup_constraints = Constraint(model.P, rule = setup_constraint_per_period)
    
    return 
"""
   # Add demand or prices bounds constraints 
"""
def demand_bound_left_mt(model,channel,period):
    f_p_mt_ub = exp(model.A[channel,period] + model.B[channel,period]*model.UB[channel,period])
    return f_p_mt_ub*(model.Mars_len[period] - sum(model.d_mt[m,period] for m in model.CH)) <= model.d_mt[channel,period]

def demand_bound_right_mt(model,channel,period):
    f_p_mt_lb = exp(model.A[channel,period] + model.B[channel,period]*model.LB[channel,period])
    return f_p_mt_lb*(model.Mars_len[period] - sum(model.d_mt[m,period] for m in model.CH)) >= model.d_mt[channel,period]

def add_demand_bounds_left_side(model):
    
    model.demand_bounds_left_side = Constraint(model.CHP, rule = demand_bound_left_mt) 

def add_demand_bounds_right_side(model):
    
    model.demand_bounds_right_side = Constraint(model.CHP, rule = demand_bound_right_mt) 
"""
   # Market share for single product model resolution
"""
def get_log_files_path(demand,periods, channels,
                       capacity,setup,instance_number):
    
    log_partial_path = f'../Results/Demand_model/{demand}/P_{periods}_CH_{channels}_cap_{capacity}_setup_{setup}/'
    log_file_name = f'Instance_{instance_number+1}_{demand}_{periods}_{channels}_cap_{capacity}_setup_{setup}_log_file'
    return f'{log_partial_path}{log_file_name}'

def solve_pricing_and_lot_sizing_multi_channel(instance,demand,capacity,setup,instance_number):
    #1: Initialize the instance data 
    global T,periods,M,channels,capacities,capacity_used \
    ,production_costs,holding_costs,setup_costs,big_M,markets_length,min_presence,A,B,LB,UB,inventory_ubs
    
    T,periods,M,channels = instance.T,instance.periods,instance.M,instance.channels
    capacities,capacity_used = instance.capacities,instance.capacity_used
    production_costs,holding_costs = instance.production_costs,instance.holding_costs
    setup_costs,big_M = instance.setup_costs,instance.big_M
    markets_length,min_presence = instance.markets_length,instance.min_presence
    A,B,LB,UB,inventory_ubs = instance.A,instance.B,instance.LB,instance.UB,instance.inventory_ubs
    
    #2: Create the model
    dm = ConcreteModel()

    #3: Add sets, parameters and variavles
    create_instance_sets(dm)
    logistic_params_initialization(dm)
    market_data_initialization(dm)
    decision_variables_creation(dm)
    compute_objective_function(dm)
    add_logistics_constraints(dm)
    add_business_constraints(dm)
    add_demand_bounds_left_side(dm)
    add_demand_bounds_right_side(dm)
    #dm.pprint()

    #4: Sovle the model
    try: 
        solver = SolverFactory('mindtpy')
        start_cpu = time.time()
        restuls = solver.solve(dm,
            strategy = 'OA',
            mip_solver='glpk', 
            nlp_solver='ipopt', 
            mip_solver_args={'timelimit': 7200},
            nlp_solver_args={'timelimit': 7200},
            tee = True,
            time_limit = 14400
            )               
        end_cpu = time.time()
        
    except:
        end_cpu = time.process_time()    
        end_exec = time.time()  
        print("Problem in resolution !")
        
    return dm, end_cpu - start_cpu

def save_model_results(model,results_path,channels,periods,cpu_time):
  
    wb = xlsxwriter.Workbook(f'{results_path}')
    worksheet = wb.add_worksheet('oa-pyomo-demand-model-results')
    worksheet.write(0,0,'Demand per channels and periods')
    start_row_channels = 2
    for channel in range(len(channels)):
        worksheet.write(start_row_channels + channel,0,channels[channel])

    start_col_periods = 1
    for period in range(len(periods)):
        worksheet.write(1,start_col_periods + period,periods[period])

    for i in range(len(channels)):
        for j in range(len(periods)):
            channel,period = channels[i],periods[j]
            worksheet.write(i+2,j+1,round(value(model.d_mt[channel,period]),2))
    
    start_row_lost_sales = start_row_channels + len(channels)+1 
    start_col_lost_sales = 1
    worksheet.write(start_row_lost_sales,0,'Lost sales per period')
    for i in range(len(periods)):
        period = periods[i]
        worksheet.write(start_row_lost_sales+1,start_col_lost_sales+i,period)
        worksheet.write(start_row_lost_sales+2,start_col_lost_sales+i,round(value(model.d_o[period]),2))
    
    start_row_production_inventory = start_row_lost_sales + 4
    start_col_production_inventory = 1
    worksheet.write(start_row_production_inventory,0,'Production and inventory per period')
    worksheet.write(start_row_production_inventory+2,0,'Production X')
    worksheet.write(start_row_production_inventory+3,0,'Inventory I')
    for i in range(len(periods)):
        period = periods[i]
        worksheet.write(start_row_production_inventory+1,start_col_production_inventory+i,period)
        worksheet.write(start_row_production_inventory+2,start_col_production_inventory+i,round(value(model.X[period])))
        if period !=len(periods):
            worksheet.write(start_row_production_inventory+3,start_col_production_inventory+i,round(value(model.I[period])))
    
    worksheet.write(start_row_production_inventory+3,len(periods),'0')
    
    total_profit_and_time_row = start_row_production_inventory + 5
    worksheet.write(total_profit_and_time_row,0,"Total profit")
    worksheet.write(total_profit_and_time_row,1,value(model.obj))
    worksheet.write(total_profit_and_time_row+1,0,"CPU time")
    worksheet.write(total_profit_and_time_row+1,1,f'{str(round(cpu_time,2))} seconds')
    wb.close()

    return 



