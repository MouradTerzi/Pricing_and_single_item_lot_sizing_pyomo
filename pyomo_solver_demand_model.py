import  sys 
import numpy as np
import pyomo
from pyomo.environ import *
import  instances_reader
from instances_reader import *
import itertools
import time
import timeit
"""
    Sets methods
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
    
    pi_mt = model.d_mt[(channel,period)]/model.B[(channel, period)]
    g_mt =  log(model.d_mt[(channel,period)]/model.d_o[period]) - model.A[(channel,period)]
    
    return pi_mt*g_mt 


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
   Market share for single product model resolution
"""

def get_log_files_path(production, demand, set_number,
                       gen_protocole, periods, channels_,
                       capacity, setup, instance_number):
    
    if set_number == '2':
        log_partial_path = f'../Results/Market_share_model/{production}_production/{demand}/set_{set_number}/'
        log_file_name = f'Instance_{instance_number}_{demand}_{len(periods)}_{len(channels)}_log_file'
        return f'{log_partial_path}{gen_protocole}_P_{len(periods)}_CH_{len(channels)}/{log_file_name}' 
    
    elif set_number == '3':
        log_partial_path = f'../Results/Market_share_model/{production}_production/{demand}/set_{set_number}/'
        log_file_name = f'Instance_{instance_number}_{demand}_{len(periods)}_{len(channels)}_cap_{capacity}_setup_{setup}_log_file'
        return f'{log_partial_path}{gen_protocole}_P_{len(periods)}_CH_{len(channels)}/cap_{capacity}_setup_{setup}/{log_file_name}' 

   
def solver_demand_single_product(T_, periods_, M_, channels_, 
                                 capacities_, capacity_used_, production_costs_, 
                                 holding_costs_, setup_costs_, big_M_, markets_length_, 
                                 min_presence_, A_, B_, LB_, UB_, inventory_ubs_
                                ):
    
    #1: Initialize the instance data 
    global ms, T, periods, M, channels, capacities, capacity_used \
    ,production_costs, holding_costs, setup_costs, big_M, markets_length, min_presence, A, B, LB, UB, inventory_ubs
    
    T, periods, M, channels = T_, periods_, M_, channels_
    capacities, capacity_used = capacities_, capacity_used_
    production_costs, holding_costs = production_costs_, holding_costs_
    setup_costs, big_M = setup_costs_, big_M_
    markets_length, min_presence = markets_length_, min_presence_
    A, B, LB, UB, inventory_ubs = A_, B_, LB_, UB_, inventory_ubs_
    
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
        
        #resolution_log = sys.stdout 
        #log_file = get_log_files_path(production, demand, set_number,
        #                            gen_protocole_, periods_, channels_,
        #                            capacity, setup, instance_number)
        
        #sys.stdout = open(f'test_2_2_instance', "w")

        solver = SolverFactory('mindtpy')
        start_exec = time.time()
        start_cpu = time.process_time()
        restuls = solver.solve(dm,
            strategy = 'OA',
            mip_solver='glpk', 
            nlp_solver='ipopt', 
            mip_solver_args={'timelimit': 7200},
            nlp_solver_args={'timelimit': 7200},
            tee = True,
            time_limit = 14400
            )
                        
        end_cpu = time.process_time()    
        end_exec = time.time()
        #sys.stdout.close()
        #sys.stdout = resolution_log
        
    except:
        
        end_cpu = time.process_time()    
        end_exec = time.time()  
        print("Problem in resolution !")
    
        
    return dm, end_cpu - start_cpu, end_exec - start_exec


"""
  # Save the model and the results
"""

def get_model_and_results_path(production, demand, set_number, 
                               gen_protocole, periods, channels, 
                               capacity, setup, instance_number):

    if set_number == '2':
        results_path = f'../Results/Market_share_model/{production}_production/{demand}/set_{set_number}/'
        return f'{results_path}{gen_protocole}_P_{periods}_CH_{channels}/Instances_{instance_number}_{demand}_{periods}_{channels}'
    
    elif set_number == '3':
        results_path = f'../Results/Market_share_model/{production}_production/{demand}/set_{set_number}/'
        return f'{results_path}{gen_protocole}_P_{periods}_CH_{channels}/cap_{capacity}_setup_{setup}/Instances_{instance_number}_{demand}_{periods}_{channels}_{capacity}_{setup}'
        

def save_demand_model_and_results(dm_model):
    """
    path =  get_model_and_results_path(production, demand, set_number, 
                                       gen_protocole, periods, channels, 
                                       capacity, setup, instance_number)
    """
    try:
        #1: Save the model
        demand_model_file = open('test_2_2_instance_model',"w")
        sys.stdout = demand_model_file
        dm_model.pprint()
        demand_model_file.close()
    
        #2: Save the resutls 
        demand_model_results_file = open(f'test_2_2_instance_results',"w")
        sys.stdout = demand_model_results_file
        dm_model.display()
        demand_model_results_file.close()

    except TypeError:
        print("Error when writing the model for the instance:") 

    return 

    

    