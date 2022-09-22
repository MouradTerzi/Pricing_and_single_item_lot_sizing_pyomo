class InstancePricingLotSizingMultiChannel:

    def __init__(self,T,periods,M,channels,
                 capacities,capacity_used,
                 production_costs,holding_costs,
                 setup_costs,big_M,markets_length, 
                 min_presence,A,B,LB,UB,inventory_ubs):

        self.T = T 
        self.periods = periods
        self.M = M 
        self.channels = channels 
        self.capacities = capacities
        self.capacity_used = capacity_used
        self.production_costs = production_costs
        self.holding_costs = holding_costs
        self.setup_costs = setup_costs
        self.big_M = big_M
        self.markets_length = markets_length
        self.min_presence = min_presence
        self.A = A
        self.B = B 
        self.LB = LB
        self.UB = UB 
        self.inventory_ubs = inventory_ubs        
        
        return 

    