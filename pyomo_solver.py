import  sys 
import numpy as np
import pyomo
from pyomo.environ import *
import  instances_reader
from instances_reader import *
import itertools

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
    Logistics Params initialization
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

"""
   Markets params initialization
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


def logistic_params_initialization(model):
    
    model.capacity_used = Param(model.P, initialize = initialize_capacity_used_per_period)
    model.capacities = Param(model.P, initialize = initialize_capacity_per_period)
    model.Prod_cost = Param(model.P, initialize = initialize_production_costs)
    model.Hold_cost = Param(list(model.P)[:-1], initialize = initialize_holding_costs)
    model.Setup_cost = Param(model.P, initialize = initialize_setup_costs)
    
    return 


def market_data_initialization(model):
    
    model.min_presence = Param(channels, initialize = initialize_minimum_markets_presence)
    model.Mars_len = Param(model.P, initialize = initialize_markets_length)
    model.A = Param(model.CHP, initialize = initialize_demand_params_a)
    model.B = Param(model.CHP, initialize = initialize_demand_params_b)
    model.LB = Param(model.CHP, initialize = initialize_lbs)
    model.UB = Param(model.CHP, initialize = initialize_ubs)
    
    return 

"""
   Variables creation
"""

def initialize_theta_mt(model, channel, period):
    if channel == 'o':
        return 0.45
    else:
        return 0.3
    

def X_bounds(model, period):
    return 0, capacities[period - 1]/capacity_used[period - 1]


def decision_variables_creation(model):

    
    model.theta_mt = Var(model.CHP, within = PositiveReals, bounds = (None,1), initialize = initialize_theta_mt)
    model.theta_o = Var(model.P, within = PositiveReals, bounds = (None,1), initialize = 0.25)
    model.X = Var(model.P, within = NonNegativeReals, bounds = X_bounds, initialize = 0)
    model.I = Var(list(model.P)[:-1], within = NonNegativeReals, initialize = 0)
    model.Y = Var(model.P, within = Binary, initialize = 1)
    
    return 

"""
  Objective function
"""

def compute_profit_mt(model,channel, period):
    
    pi_mt = (markets_length[period - 1]*model.theta_mt[(channel,period)])/(model.B[(channel, period)])
    g_mt =  log(model.theta_mt[(channel,period)]/model.theta_o[period]) - model.A[(channel,period)]
    
    return pi_mt*g_mt 


def compute_objective_function(model):
    
    model.profit = Expression(expr = sum(compute_profit_mt(model,channel, period) \
    for (channel, period) in ms.CHP))
    
    model.total_costs = Expression(expr = summation(model.Prod_cost,model.X) \
    + summation(model.Hold_cost,model.I) + summation(model.Setup_cost,model.Y))
    
    model.obj = Objective(expr = model.profit - model.total_costs, sense = maximize)
    
    return 

"""
  Add Business constraints
"""

def minimum_presence_mt(model, channel, period):
    
    return ms.min_presence[channel]*sum(model.theta_mt[(m,period)] for m in ms.CH) \
    <= model.theta_mt[(channel,period)]


def sum_ms_equal_to_one(model, period):
    return sum([model.theta_mt[(m,period)] for m in ms.CH]) + ms.theta_o[period] == 1


def add_business_constraints(model):
    
    model.presence_constraints = Constraint(model.CHP, rule = minimum_presence_mt)
    model.ms_equal_to_one_constraints = Constraint(model.P, rule = sum_ms_equal_to_one)

"""
  Add Logistic constraints
"""

def production_limit_constraint(model,period):
    return model.X[period] <= model.capacities[period]/model.capacity_used[period]


def inventory_between_two_and_t_1(model,period):
    
    return model.Mars_len[period]*sum(model.theta_mt[m,period] for m in model.CH) \
    + model.I[period] - model.I[period -1] - model.X[period] == 0


def setup_constraint_per_period(model, period):
    return model.X[period] <= big_M*model.Y[period]


def add_logistics_constraints(model):
    
    #1: Production capacity limits
    model.production_limits = Constraint(model.P, rule = production_limit_constraint) 
    
    #2: Inventory at the end of the first period
    model.inventory_first_period = Constraint(rule = model.Mars_len[1]*sum(ms.theta_mt[m,1] \
    for m in model.CH) + model.I[1] - model.X[1] == 0)
    
    #3: Inventory for the periods between 2 and T - 1
    model.inventory_2_to_T_1 = Constraint(list(model.P)[1:-1], rule = inventory_between_two_and_t_1)
    
    #4: Inventory for t == T
    model.inventory_end_period = Constraint(rule = model.Mars_len[T]*sum(model.theta_mt[m,T] \
    for m in model.CH) - model.I[T-1] - ms.X[T] == 0)
    
    #5: Setup constraints
    model.setup_constraints = Constraint(model.P, rule = setup_constraint_per_period)
    
    return 

"""
   Add theta or prices bounds constraints 
"""

def theta_bound_left_mt(model,channel,period):
    
    f_p_mt_ub = exp(model.A[channel,period] + model.B[channel,period]*model.UB[channel,period])
    return f_p_mt_ub*(1 - sum(model.theta_mt[m,period] for m in model.CH)) <= model.theta_mt[channel,period]


def theta_bound_right_mt(model,channel,period):
    
    f_p_mt_lb = exp(model.A[channel,period] + model.B[channel,period]*model.LB[channel,period])
    return f_p_mt_lb*(1 - sum(model.theta_mt[m,period] for m in model.CH)) >= model.theta_mt[channel,period]


def add_theta_bounds_left_side(model):
    
    model.theta_bounds_left_side = Constraint(model.CHP, rule = theta_bound_left_mt) 


def add_theta_bounds_right_side(model):
    
    model.theta_bounds_right_side = Constraint(model.CHP, rule = theta_bound_right_mt) 

"""
   Market share for single product model resolution
"""

def solver_market_share_single_product(T_, periods_, M_, channels_, 
                                      capacities_, capacity_used_, 
                                      production_costs_, holding_costs_, 
                                      setup_costs_, big_M_, \
                                      markets_length_, min_presence_,
                                      A_, B_, LB_, UB_):
    
    #1: Initialize the instance data 
    global ms, T, periods, M, channels, capacities, capacity_used \
    ,production_costs, holding_costs, setup_costs, big_M, markets_length, min_presence, A, B, LB, UB
    
    T, periods, M, channels = T_, periods_, M_, channels_
    capacities, capacity_used = capacities_, capacity_used_
    production_costs, holding_costs = production_costs_, holding_costs_
    setup_costs, big_M = setup_costs_, big_M_
    markets_length, min_presence = markets_length_, min_presence_
    A, B, LB, UB = A_, B_, LB_, UB_

    #2: Create the model
    ms = ConcreteModel()

    #3: Add sets, parameters and variavles
    create_instance_sets(ms)
    logistic_params_initialization(ms)
    market_data_initialization(ms)
    decision_variables_creation(ms)
    compute_objective_function(ms)
    add_business_constraints(ms)
    add_logistics_constraints(ms)
    add_theta_bounds_left_side(ms)
    add_theta_bounds_right_side(ms)

    #4: Sovle the model
    SolverFactory('mindtpy').solve(ms, mip_solver='glpk', nlp_solver='ipopt', tee = True) 
    
    return ms

"""
  Save the model and the results
"""

def save_ms_model_and_results(ms_model, demand, gen_protocole,
                              set_, periods, channels, 
                              instance_number):
    
    path = f'../Results/{demand}/{gen_protocole}/{set_}/P_{periods}_CH_{channels}/Instance_{instance_number}_{demand}_{periods}_{channels}' #1_MNL_6_2
    #1: Save the model
    ms_model_file = open(path+"_model","w")
    sys.stdout = ms_model_file
    ms_model.pprint()
    ms_model_file.close()

    #2: Save the resutls 
    ms_model_results_file = open(path+"_results","w")
    sys.stdout = ms_model_results_file
    ms_model.display()
    ms_model_results_file.close()

    return 

    

    