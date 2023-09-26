#!/usr/bin/env python3
import numpy as np

#####################################################
### Graph-based mutually exciting point processes ###
#####################################################

class gb_mep:
    
    ## Initialise the class from a DataFrame with columns 'start_id', 'end_id', 'start_time' and 'end_time'
    def __init__(self, df, id_map, distance_matrix):
        ## Define DataFrame with event times on the network
        if not set(['start_id','end_id','start_time','end_time']).issubset(df.columns):
            return ValueError("The DataFrame in input should have columns 'start_id', 'end_id', 'start_time' and 'end_time'.")
        else:
            self.df = df
        ## Define dictionary with map from ID to names and vice-versa
        self.id_map = id_map
        ## Define distance matrix
        self.distance_matrix = distance_matrix