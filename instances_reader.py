import sys 
import os
import shutil 
import random
import numpy as np 

from instance import *

class Reader:

    """
    # Read prices' lower and upper bounds from instances with lingo format (Markets data)
    """
    def read_lower_bounds(self, instance, 
                         lbs_index, channels, periods):
        
        LB = {}
        channel_ind = 0
        
        while '!Upper' not in instance[lbs_index].split(' '):
            
            channel_lbs = instance[lbs_index].split(' ')
            channel_lbs = [float(channel_lbs[i]) for i in range(len(channel_lbs)) \
                        if channel_lbs[i] != '' and channel_lbs[i] !='~']
            
            for period in periods:
                LB[channels[channel_ind],period] = channel_lbs[period - 1]
            
            channel_ind += 1
            lbs_index += 1  
        return LB
 
    def read_upper_bounds(self, instance, 
                         ubs_index, channels, periods):
        
        UB = {}
        channel_ind = 0
        
        while '!Demand' not in instance[ubs_index].split(' '):
            channel_ubs = instance[ubs_index].split(' ')
            channel_ubs = [float(channel_ubs[i]) for i in range(len(channel_ubs)) \
                        if channel_ubs[i] != '' and channel_ubs[i] !='~']
            
            for period in periods:    
                UB[channels[channel_ind],period] = channel_ubs[period - 1]
            
            channel_ind += 1
            ubs_index += 1  
            
        return UB

    """
    # Read demand parameters from instances with lingo format (Markets data)
    """

    def read_demand_parameter_A(self, instance, 
                               a_index, channels, periods):
        
        A = {}
        channel_ind = 0
        
        while '!Demand' not in instance[a_index].split(' '):
            channel_a = instance[a_index].split(' ')
            channel_a = [float(channel_a[i]) for i in range(len(channel_a)) \
                        if channel_a[i] != '' and channel_a[i] != '~']
            
            for period in periods:    
                A[channels[channel_ind],period] = channel_a[period - 1]
            
            channel_ind += 1
            a_index += 1  
        
        return A

    def read_demand_parameter_B(self, instance, 
                                in_b_index, channels, periods):
        
        B = {}
        channel_ind = 0
        
        for b_index in range(in_b_index, in_b_index + len(channels)):
            channel_b = instance[b_index].split(' ')
            channel_b = [float(channel_b[i]) for i in range(len(channel_b)) \
                    if channel_b[i] != '' and channel_b[i] != '~']

            for period in periods:   
                B[channels[channel_ind],period] = channel_b[period - 1]

            channel_ind += 1
        
        return B

    """
    # Read logistics data from instances with lingo format
    """

    def read_logistics_data_lingo_format(self,instance):
    
        capacity_used = instance[8].split(' ')
        capacity_used = [float(capacity_used[i]) for i in \
                        range(len(capacity_used)) if capacity_used[i] !='' and capacity_used[i] != '~']
        
        capacities = instance[10].split(' ')
        capacities = [float(capacities[i]) for i in \
                        range(len(capacities)) if capacities[i] !='' and capacities[i] != '~']
        
        production_costs = instance[12].split(' ')
        production_costs = [float(production_costs[i]) for i in \
                        range(len(production_costs)) if production_costs[i] !='' and production_costs[i] != '~']
        
        holding_costs = instance[14].split(' ')
        holding_costs = [float(holding_costs[i]) for i in \
                        range(len(holding_costs)) if holding_costs[i] !=''and holding_costs[i] != '~']
        
        setup_costs  = instance[16].split(' ')
        setup_costs = [float(setup_costs[i]) for i in \
                        range(len(setup_costs)) if setup_costs[i] !='' and setup_costs[i] != '~']
        
        inventory_ubs = np.array(capacities)[:-1]/np.array(capacity_used)[:-1]
        inventory_ubs = list(np.cumsum(inventory_ubs))

        big_M = np.sum(np.array(capacities)/np.array(capacity_used))
        return capacity_used, capacities, production_costs, holding_costs, setup_costs, inventory_ubs, big_M
    
    def read_market_data_lingo_format(self, instance,
                                  channels,periods,M):
        
        markets_length = instance[22].split(' ')
        markets_length = [float(markets_length[i]) for i in \
                        range(len(markets_length)) if markets_length[i] !='' and markets_length[i] != '~']
    
        min_presence = instance[24].split(' ')
        min_presence = [float(min_presence[i]) for i in \
                        range(len(min_presence)) if min_presence[i] !='' and min_presence[i] != '~']
    
        
        LB = self.read_lower_bounds(instance, 26, channels, periods)
        UB = self.read_upper_bounds(instance, 26 + M + 1, channels, periods)
        A = self.read_demand_parameter_A(instance, 26 + 2*M + 2, channels, periods) 
        B = self.read_demand_parameter_B(instance, 26 + 3*M + 3, channels, periods)
        
        return markets_length, min_presence, LB, UB, A, B 
     
    def get_instance_path(self,demand,M,T, 
                        capacity, setup, 
                        instanceNumber):
        
            path = f'../../Instances/{demand}/P_{T}_CH_{M}_cap_{capacity}_setup_{setup}/'
            return f'{path}Instance_{str(instanceNumber+1)}_{demand}_{T}_{M}_cap_{capacity}_setup_{setup}.LDT'
            
    def read_instance_lingo_format(self,demand,M,T,capacity,
                                   setup,instanceNumber):
        
        try:
            instancePath = self.get_instance_path(demand,M,T,capacity, 
                                                setup,instanceNumber)   

            print(instancePath)                      
            f = open(instancePath,"r")
        
        except FileNotFoundError:
            print("Wrong file or file path")
            return [-1]*17, "Instance not found"
        
        instance = f.read().split('\n')
        instance = [instance[i] for i in range(len(instance)) if instance[i] != ' ']
        
        """
        # Read the general data lingo format
        """
        
        T = int(instance[2].split(' ')[0])
        periods = [t+1 for t in range(T)]
        channels = instance[4].split(' ')[:]
        channels = [channels[i] for i in range(len(channels)) if channels[i] != '' and channels[i] != '~']
        M = len(channels) 
        
        """
        # Read the logistics data lingo format
        """  
        
        capacityUsed,capacities,productionCosts, \
        holdingCosts,setupCosts,inventoryUbs,bigM = self.read_logistics_data_lingo_format(instance)
        
        """
        # Read markets data lingo format
        """

        marketsLength,minPresence,LB,UB,A,B = self.read_market_data_lingo_format(instance,channels,periods,M)

        instance = InstancePricingLotSizingMultiChannel(T,periods,M,channels,capacities,
                                                        capacityUsed,productionCosts,
                                                        holdingCosts,setupCosts,bigM,
                                                        marketsLength,minPresence,A,B,LB,
                                                        UB,inventoryUbs)
        
        return instance
    
    def show_instance_data(self,instance):

        print(f'General data:')
        print(f'Periods number: {instance.T}')
        print(f'Periods:{instance.periods}')
        
        print(f'Logistics data:')
        print(f'Production capacities per period: {instance.capacities}')
        print(f'Capacity used per period:{instance.capacity_used}')
        print(f'Production costs: {instance.production_costs}')
        print(f'Holding costs: {instance.holding_costs}')
        print(f'Setup costs: {instance.setup_costs}')
        print(f'Big M: {instance.big_M}\n\n')
        
        print(f'Markets data:')
        
        print(f'Channels number: {instance.M}')
        print(f'Channels list: {instance.channels}')
        print(f'Market lengths: {instance.markets_length}')
        print(f'Minimum presence: {instance.min_presence}')
        print(f'Demand parameters A:')
        [print("A[",channel,period,"] = ",instance.A[channel,period]) for channel in instance.channels for period in instance.periods]
        print("\nDemand parameters B:")
        [print("B[",channel,period,"] = ",instance.B[channel,period]) for channel in instance.channels for period in instance.periods]
        print("\nPrices lower bounds:")
        [print("LB[",channel,period,"] = ",instance.LB[channel,period]) for channel in instance.channels for period in instance.periods]
        print("\nPrices upper bounds")
        [print("UB[",channel,period,"] = ",instance.UB[channel,period]) for channel in instance.channels for period in instance.periods]
        
        return 
      



        
