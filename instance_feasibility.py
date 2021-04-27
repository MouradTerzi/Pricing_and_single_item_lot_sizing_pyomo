import  sys 
import numpy as np
import pyomo
from pyomo.environ import *

def channels_set_init(model,channel):
    
    return [channels.index(channel) + 1]


def create_instance_sets(model):
    
    """
      1. Channels set      
    """

    model.CH = Set(channels, initialize = channels_set_init)

    return

"""
   Initialize the parameters for the model
"""

def initialize_minimum_markets_presence(model, channel):
    return min_presence[list(model.CH).index(channel)]



def solve_peiord_nlp(channels_,A_t_,B_t_,LB_t_,UB_t_):

    global channels, A_t, B_t, LB_t, UB_t
    
    channels, A_t, B_t, LB_t, UB_t = channels_,A_t_,B_t_,LB_t_,UB_t_
    
    nlp_ms = ConcreteModel()
    create_instance_sets(nlp_ms)
    nlp_ms.pprint()


def check_instance_feasibility(channels, periods,
                               min_presence, A, B,
                               LB, UB):

    print("Minimum presence is:",min_presence)

    print("The demand (A) parameters are:")
    [print("A[",channel,period,"]=",A[channel, period]) for (channel,period) in A]

    print("The demand (B) parameters are:")
    [print("B[",channel,period,"]=",B[channel, period]) for (channel,period) in B]

    print("The LB values are:")
    [print("LB[",channel,period,"]=",LB[channel, period]) for (channel,period) in LB]

    print("The UB values are:")
    [print("UB[",channel,period,"]=",UB[channel, period]) for (channel,period) in UB]

    theta_mt_init = {}
    theta_o_t_init = {}

    for period in periods: 
        #1: Get the parameters of the period 
        A_t = dict(((channel,period),A[channel,period]) for channel in channels)
        B_t = dict(((channel,period),B[channel,period]) for channel in channels)
        LB_t = dict(((channel,period),LB[channel,period]) for channel in channels)
        UB_t = dict(((channel,period),UB[channel,period]) for channel in channels)

        solve_peiord_nlp(channels,A_t,B_t,LB_t,UB_t)
    
    return 

 