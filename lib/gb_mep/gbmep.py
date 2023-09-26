#!/usr/bin/env python3
import numpy as np
from scipy.optimize import minimize

#####################################################
### Graph-based mutually exciting point processes ###
#####################################################

class gb_mep:
    
    ### Initialise the class from a DataFrame with columns 'start_id', 'end_id', 'start_time' and 'end_time'
    def __init__(self, df, id_map, distance_matrix):
        # Define DataFrame with event times on the network
        if not set(['start_id','end_id','start_time','end_time']).issubset(df.columns):
            return ValueError("The DataFrame in input should have columns 'start_id', 'end_id', 'start_time' and 'end_time'.")
        else:
            # Store DataFrame
            self.df = df
            # Find unique nodes (node IDs should start at 0 and match the dictionary and ID mapping dictionary)
            self.nodes = np.unique(self.df[['start_id','end_id']].values)
            # For each node, find all start times and end times
            self.start_times = {}
            self.end_times = {}
            for node in self.nodes:
                self.start_times[node] = np.array(self.df['start_time'][self.df['start_id'] == node].sort_values())
                self.end_times[node] = np.array(self.df['end_time'][self.df['end_id'] == node].sort_values())
        # Define dictionary with map from ID to names and vice-versa
        self.id_map = id_map
        # Define distance matrix
        self.distance_matrix = distance_matrix
        self.M = self.distance_matrix.shape[0]
        self.T = int(np.ceil(np.max(self.df[['start_time','end_time']].values)))

    ###
    def fit(self, start_times=True, end_times=True, distance_start=True, distance_end=False):
        res = {}
        for node in self.nodes:
            print('\r', node, '-', self.id_map[node], ' '*20, end='\r')
            # Recursion
            for secondary_node in self.nodes:
                start_breaks = np.searchsorted(a=self.start_times[secondary_node], v=self.start_times[node], side='left')
                end_breaks = np.searchsorted(a=self.end_times[secondary_node], v=self.start_times[node], side='left')
                time_diffs_A = {}; time_diffs_A_prime = {}
                for k, t in enumerate(self.start_times[node]):
                    if k != 0:
                        time_diffs_A[k, secondary_node] = t - self.start_times[secondary_node][start_breaks[k-1]:start_breaks[k]]
                        time_diffs_A_prime[k, secondary_node] = t - self.end_times[secondary_node][end_breaks[k-1]:end_breaks[k]]
                    else:
                        time_diffs_A[k, secondary_node] = t - self.start_times[secondary_node][:start_breaks[k]]
                        time_diffs_A_prime[k, secondary_node] = t - self.end_times[secondary_node][:end_breaks[k]]
            res[node] = minimize(fun=self.negative_loglikelihood_full, x0=np.array([np.log(k+1/self.T), -1.0, -2.0, -1.0, -1.0, -2.0, 0.0]),
                                    args=(node, time_diffs_A, time_diffs_A_prime)
        return res

    ### Calculate negative log-likelihood for the full model, for a specific node index
    def negative_loglikelihood_full(self, params, node_index, time_diffs_A, time_diffs_A_prime):
        # Transform parameters to original scale (lambda, alpha, beta, theta, alpha_prime, beta_prime, theta_prime)
        params = np.exp(params)
        params[2] += params[1]
        params[5] += params[4]
        # Time differences for starting times for node with corresponding index
        time_diffs = np.diff(self.start_times[node_index])
        # Pre-define arrays for the recursive terms (A and A_prime)
        A = np.zeros((len(time_diffs)+1,self.M))
        A_prime = np.zeros((len(time_diffs)+1,self.M))
        # Obtain distance between node and all other nodes
        ds = self.distance_matrix[node_index]
        # Compensator component of loglikelihood
        ll = -params[0] * self.T
        # Loop over all nodes
        for node in self.nodes:
            # Compensator components of loglikelihood
            ll -= np.exp(-params[3] * ds[node]) * params[1] / params[2] * np.sum(np.exp(-params[2] * (self.T - self.start_times[node]) - 1))
            ll -= np.exp(-params[6] * ds[node]) * params[4] / params[5] * np.sum(np.exp(-params[5] * (self.T - self.end_times[node]) - 1))
            # Loop over all events and update recursive terms A and A_prime
            for k, _ in enumerate(self.start_times[node_index]):    
                A[k,node] = ((np.exp(-params[2] * time_diffs[k-1]) * A[k-1,node]) if k > 0 else 0) + np.sum(np.exp(-params[2] * time_diffs_A[k, node])) 
                A_prime[k,node] = ((np.exp(-params[5] * time_diffs[k-1]) * A_prime[k-1,node]) if k > 0 else 0) + np.sum(np.exp(-params[5] * time_diffs_A_prime[k, node])) 
        # Calculate B and use it to update the log-likelihood
        B = np.exp(-params[3] * ds) * params[1] * A + np.exp(-params[6] * ds) * params[4] * A 
        ll += np.sum(np.log(params[0] + B.sum(axis=1)))
        # Return final value
        return -ll