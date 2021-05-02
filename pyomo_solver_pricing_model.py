import  sys 
import numpy as np
import pyomo
from pyomo.environ import *
import  instances_reader
from instances_reader import *
import itertools
import time

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


def logistic_params_initialization(model):
    
    model.capacity_used = Param(model.P, initialize = initialize_capacity_used_per_period, within = PositiveReals)
    model.capacities = Param(model.P, initialize = initialize_capacity_per_period, within = PositiveReals)
    model.Prod_cost = Param(model.P, initialize = initialize_production_costs, within = PositiveReals)
    model.Hold_cost = Param(list(model.P)[:-1], initialize = initialize_holding_costs, within = PositiveReals)
    model.Setup_cost = Param(model.P, initialize = initialize_setup_costs, within = PositiveReals)
    
    return 

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


def market_data_initialization(model):

    model.min_presence = Param(channels, initialize = initialize_minimum_markets_presence, within = PositiveReals)
    model.Mars_len = Param(model.P, initialize = initialize_markets_length, within = PositiveReals)
    model.A = Param(model.CHP, initialize = initialize_demand_params_a, within = PositiveReals)
    model.B = Param(model.CHP, initialize = initialize_demand_params_b, within = NegativeReals)
    model.LB = Param(model.CHP, initialize = initialize_lbs, within = PositiveReals)
    model.UB = Param(model.CHP, initialize = initialize_ubs, within = PositiveReals)
    
    return 

"""
   Variables creation
"""    
def prices_bounds(model,channel,period):

    return (model.LB[channel,period], model.UB[channel,period])


def I_upper_bounds(model, period):
    return (0,inventory_ubs[period - 1])


def decision_variables_creation(model):
    
    model.prices = Var(model.CHP, within = PositiveReals)
    model.demand = Var(model.CHP, within = PositiveReals)
    model.X = Var(model.P, within = NonNegativeReals, initialize = 0)
    model.I = Var(list(model.P)[:-1], bounds= I_upper_bounds, within = NonNegativeReals, initialize = 0)
    model.Y = Var(model.P, within = Binary)
    
    return 

"""
    Objective function
"""

def compute_objective_function(model):
    

    model.profit = Expression(expr = summation(model.prices, model.demand))
    
    model.total_costs = Expression(expr = summation(model.Prod_cost,model.X) \
    + summation(model.Hold_cost,model.I) + summation(model.Setup_cost,model.Y))
    
    model.obj = Objective(expr = model.profit.expr - model.total_costs.expr, sense = maximize)

    return 

"""
  Add Logistic constraints
"""

def production_limit_constraint(model,period):
    return model.X[period] <= model.capacities[period]/model.capacity_used[period]


def inventory_between_two_and_t_1(model,period):
    
    return model.Mars_len[period]*sum(model.demand[m,period] for m in model.CH) \
    + model.I[period] - model.I[period -1] - model.X[period] == 0


def setup_constraint_per_period(model, period):
    return model.X[period] <= big_M*model.Y[period]


def add_logistics_constraints(model):
    
    #1: Production capacity limits
    model.production_limits = Constraint(model.P, rule = production_limit_constraint) 
    
    #2: Inventory for the first period
    model.inventory_first_period = Constraint(rule = sum(model.demand[m,1] \
    for m in model.CH) + model.I[1] - model.X[1] == 0)
    
    #3: Inventory for the periods between 2 and T - 1
    model.inventory_2_to_T_1 = Constraint(list(model.P)[1:-1], rule = inventory_between_two_and_t_1)
    
    #4: Inventory for t == T
    model.inventory_end_period = Constraint(rule = sum(model.demand[m,T] \
    for m in model.CH) - model.I[T-1] - model.X[T] == 0)
    
    #5: Setup constraints
    model.setup_constraints = Constraint(model.P, rule = setup_constraint_per_period)
    
    return 

"""
  Add Business constraints
"""
def mnl_demand(model, channel, period):
    
    demand_nom = model.Mars_len[period]*exp(model.A[channel,period] + model.B[channel,period]*model.prices[channel,period])
    demand_den = 1 + sum([exp(model.A[m,period] + model.B[m,period]*model.prices[m,period]) for m in model.CH])
  
    return (demand_nom/demand_den) == model.demand[channel,period]
   

def minimum_presence_mt(model, channel, period):
    
    return model.min_presence[channel]*sum(model.demand[(m,period)] for m in model.CH) \
    <= model.demand[(channel,period)]

def add_business_constraints(model):
    
    model.mnl_demand = Constraint(model.CHP, rule = mnl_demand)
    model.presence_constraints = Constraint(model.CHP, rule = minimum_presence_mt)
    return 
    
"""
  Prices bouns
"""

def lb_prices(model,channel,period):
    return model.prices[channel,period] >= model.LB[channel,period]

def ub_prices(model,channel,period):
    return model.prices[channel,period] <= model.UB[channel,period]


def add_prices_bounds_constraints(model):
    
    model.prices_lbs = Constraint(model.CHP, rule = lb_prices)
    model.prices_ubs = Constraint(model.CHP, rule = ub_prices)

    #model.presence_constraints = Constraint(model.CHP, rule = minimum_presence_mt)
    return 

"""
    Solve the pricing model
"""
def solver_pricing_single_product(T_, periods_, M_, channels_, set_, 
                                      demand_, demand_params_, set_number,
                                      instance_number_, gen_protocole_, capacities_, 
                                      capacity_used_, production_costs_, 
                                      holding_costs_, setup_costs_, 
                                      big_M_, markets_length_, min_presence_,
                                      A_, B_, LB_, UB_, inventory_ubs_):
    
    #1: Initialize the instance data 
    global prices_model, T, periods, M, channels, capacities, capacity_used \
    ,production_costs, holding_costs, setup_costs, big_M, markets_length, min_presence, A, B, LB, UB, inventory_ubs
    
    T, periods, M, channels = T_, periods_, M_, channels_
    capacities, capacity_used = capacities_, capacity_used_
    production_costs, holding_costs = production_costs_, holding_costs_
    setup_costs, big_M = setup_costs_, big_M_
    markets_length, min_presence = markets_length_, min_presence_
    A, B, LB, UB, inventory_ubs = A_, B_, LB_, UB_, inventory_ubs_
    
    #2: Create the model
    prices_model = ConcreteModel()

    #3: Add sets, parameters and variavles
    create_instance_sets(prices_model)
    logistic_params_initialization(prices_model)
    market_data_initialization(prices_model)
    decision_variables_creation(prices_model)
    compute_objective_function(prices_model)
    add_logistics_constraints(prices_model)
    add_business_constraints(prices_model)
    add_prices_bounds_constraints(prices_model)
    #prices_model.pprint()
    
    print("here")
    #4: Sovle the model
    #try:
        
        #resolution_log = sys.stdout 
        #log_file = f'../Results/Pricing_model/{demand_}/{set_}/{gen_protocole_}_P_{len(periods_)}_CH_{len(channels_)}_set_{set_number}/{demand_params_}/' 
        #sys.stdout = open(f'{log_file}_Instance_{instance_number_}_{demand_}_{len(periods)}_{len(channels)}_log_file', "w")
        
    #start = time.process_time()
    SolverFactory('mindtpy').solve(prices_model,
                                    
                                    tee = True
                                    )
    
        #end = time.process_time()    
        #sys.stdout.close()
        #sys.stdout = resolution_log
               
    #except:
    #    end = time.process_time()   
    #    print("Instance infeasible !")
    #    return prices_model, end - start
    

    return prices_model, -1 #end - start

