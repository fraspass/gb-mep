#!/usr/bin/env python3
import numpy as np
from scipy.optimize import minimize
from collections import Counter

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
        self.N = Counter(); self.N_prime = Counter()
        for node in self.nodes:
            self.N[node] = len(self.start_times[node])
            self.N_prime[node] = len(self.end_times[node])

    ### Fit the model to a subset of nodes
    def fit(self, x0, subset_nodes=None, start_times=True, end_times=True, distance_start=False, distance_end=False, thresh=1):
        # Define the dictionary for results
        res = {}
        # If the subset of nodes is not specified, consider all nodes
        if subset_nodes is None:
            subset_nodes = self.nodes
        # Loop over all nodes in the subset
        for node in subset_nodes:
            # Print node and name
            print('\r', node, '-', self.id_map[node], ' '*20, end='\r')
            # Obtain distances if required
            if distance_start or distance_end:
                ds = self.distance_matrix[node]
            # Calculate required elements for recursion
            time_diffs_A = {}; time_diffs_A_prime = {}
            if not start_times and not end_times:
                condition_full = False
            else:
                condition_full = True
                condition = not (distance_start or distance_end)
            if condition_full:
                for secondary_node in ([node] if condition else np.where(ds < thresh)[0]):
                    if self.N[secondary_node] + self.N_prime[secondary_node] > 0:
                        start_breaks = np.searchsorted(a=self.start_times[secondary_node], v=self.start_times[node], side='left')
                        end_breaks = np.searchsorted(a=self.end_times[secondary_node], v=self.start_times[node], side='left')
                        for k, t in enumerate(self.start_times[node]):
                            if k != 0:
                                time_diffs_A[k, secondary_node] = t - self.start_times[secondary_node][start_breaks[k-1]:start_breaks[k]]
                                time_diffs_A_prime[k, secondary_node] = t - self.end_times[secondary_node][end_breaks[k-1]:end_breaks[k]]
                            else:
                                time_diffs_A[k, secondary_node] = t - self.start_times[secondary_node][:start_breaks[k]]
                                time_diffs_A_prime[k, secondary_node] = t - self.end_times[secondary_node][:end_breaks[k]]
                    else:
                        for k, t in enumerate(self.start_times[node]):
                            time_diffs_A[k, secondary_node] = np.array([])
                            time_diffs_A_prime[k, secondary_node] = np.array([])
            # Obtain the correct log-likelihood function based on fitting parameters
            if start_times and end_times and distance_start and distance_end:
                f = self.negative_loglikelihood_full
                f_args = (node, time_diffs_A, time_diffs_A_prime, np.where(ds < thresh)[0])
            elif start_times and end_times and distance_start and not distance_end:
                f = self.negative_loglikelihood_full_start
                f_args = (node, time_diffs_A, time_diffs_A_prime, np.where(ds < thresh)[0])
            elif start_times and end_times and not distance_start and not distance_end:
                f = self.negative_loglikelihood_smep
                f_args = (node, time_diffs_A, time_diffs_A_prime)
            elif start_times and not end_times:
                f = self.negative_loglikelihood_sep
                f_args = (node, time_diffs_A)
            elif not start_times and end_times:
                f = self.negative_loglikelihood_mep
                f_args = (node, time_diffs_A_prime)
            elif not start_times and not end_times:
                f = 'Poisson'
            else:
                return ValueError('The chosen combination of start_times, end_times, distance_start and distance_end is not supported by this library.')
            # Minimise negative log-likelihood, or obtain exact solution if the function is simply the Poisson process
            if f == 'Poisson':
                res[node] = len(self.start_times[node]) / self.T
            else:
                res[node] = minimize(fun=f, x0=x0, args=f_args)
        return res

    ### Calculate negative log-likelihood for the self-exciting model, for a specific node index
    def negative_loglikelihood_sep(self, p, node_index, time_diffs_A):
        # Transform parameters to original scale (lambda, alpha, beta)
        params = np.exp(p)
        params[2] += params[1]
        # Time differences for starting times for node with corresponding index
        time_diffs = np.diff(self.start_times[node_index])
        # Pre-define arrays for the recursive terms (A)
        A = np.zeros(len(time_diffs)+1)
        # Compensator component of loglikelihood
        ll = -params[0] * self.T
        # Compensator components of loglikelihood
        ll += params[1] / params[2] * np.sum(np.exp(-params[2] * (self.T - self.start_times[node_index])) - 1)
        # Loop over all events and update recursive term A
        for k, _ in enumerate(self.start_times[node_index]):    
            A[k] = ((np.exp(-params[2] * time_diffs[k-1]) * A[k-1]) if k > 0 else 0) + np.sum(np.exp(-params[2] * time_diffs_A[k, node_index]))
        # Calculate B and use it to update the log-likelihood
        B = params[1] * A
        ll += np.sum(np.log(params[0] + B.sum()))
        # Return final value
        return -ll

    ### Calculate negative log-likelihood for the mutually exciting model, for a specific node index
    def negative_loglikelihood_mep(self, p, node_index, time_diffs_A_prime):
        # Transform parameters to original scale (lambda, alpha_prime, beta_prime)
        params = np.exp(p)
        params[2] += params[1]
        # Time differences for starting times for node with corresponding index
        time_diffs = np.diff(self.start_times[node_index])
        # Pre-define arrays for the recursive terms (A_prime)
        A_prime = np.zeros(len(time_diffs)+1)
        # Compensator component of loglikelihood
        ll = -params[0] * self.T
        # Compensator components of loglikelihood
        ll += params[1] / params[2] * np.sum(np.exp(-params[2] * (self.T - self.end_times[node_index])) - 1)
        # Loop over all events and update recursive term A_prime
        for k, _ in enumerate(self.start_times[node_index]):    
            A_prime[k] = ((np.exp(-params[2] * time_diffs[k-1]) * A_prime[k-1]) if k > 0 else 0) + np.sum(np.exp(-params[2] * time_diffs_A_prime[k, node_index])) 
        # Calculate B and use it to update the log-likelihood
        B = params[1] * A_prime
        ll += np.sum(np.log(params[0] + B.sum()))
        # Return final value
        return -ll

    ### Calculate negative log-likelihood for the self-and-mutually exciting model, for a specific node index
    def negative_loglikelihood_smep(self, p, node_index, time_diffs_A, time_diffs_A_prime):
        # Transform parameters to original scale (lambda, alpha, beta, alpha_prime, beta_prime)
        params = np.exp(p)
        params[2] += params[1]
        params[4] += params[3]
        # Time differences for starting times for node with corresponding index
        time_diffs = np.diff(self.start_times[node_index])
        # Pre-define arrays for the recursive terms (A and A_prime)
        A = np.zeros(len(time_diffs)+1)
        A_prime = np.zeros(len(time_diffs)+1)
        # Compensator component of loglikelihood
        ll = -params[0] * self.T
        # Compensator components of loglikelihood
        ll += params[1] / params[2] * np.sum(np.exp(-params[2] * (self.T - self.start_times[node_index])) - 1)
        ll += params[3] / params[4] * np.sum(np.exp(-params[4] * (self.T - self.end_times[node_index])) - 1)
        # Loop over all events and update recursive terms A and A_prime
        for k, _ in enumerate(self.start_times[node_index]):    
            A[k] = ((np.exp(-params[2] * time_diffs[k-1]) * A[k-1]) if k > 0 else 0) + np.sum(np.exp(-params[2] * time_diffs_A[k, node_index])) 
            A_prime[k] = ((np.exp(-params[4] * time_diffs[k-1]) * A_prime[k-1]) if k > 0 else 0) + np.sum(np.exp(-params[4] * time_diffs_A_prime[k, node_index])) 
        # Calculate B and use it to update the log-likelihood
        B = params[1] * A + params[3] * A_prime
        ll += np.sum(np.log(params[0] + B.sum()))
        # Return final value
        return -ll

    ### Calculate negative log-likelihood for the full model without a distance function for the end times, for a specific node index
    def negative_loglikelihood_full_start(self, p, node_index, time_diffs_A, time_diffs_A_prime, subset_nodes):
        # Transform parameters to original scale (lambda, alpha, beta, theta, alpha_prime, beta_prime, theta_prime)
        params = np.exp(p)
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
        for node in subset_nodes:
            # Compensator components of loglikelihood
            ll += np.exp(-params[3] * ds[node]) * params[1] / params[2] * np.sum(np.exp(-params[2] * (self.T - self.start_times[node])) - 1)
            ll += np.exp(-params[6] * ds[node]) * params[4] / params[5] * np.sum(np.exp(-params[5] * (self.T - self.end_times[node])) - 1)
            # Loop over all events and update recursive terms A and A_prime
            for k, _ in enumerate(self.start_times[node_index]):    
                A[k, node] = ((np.exp(-params[2] * time_diffs[k-1]) * A[k-1, node]) if k > 0 else 0) + np.sum(np.exp(-params[2] * time_diffs_A[k, node]))
                if node == node_index:
                    A_prime[k, node] = ((np.exp(-params[5] * time_diffs[k-1]) * A_prime[k-1, node]) if k > 0 else 0) + np.sum(np.exp(-params[5] * time_diffs_A_prime[k, node])) 
        # Calculate B and use it to update the log-likelihood
        B = np.exp(-params[3] * ds) * params[1] * A + params[4] * A_prime
        ll += np.sum(np.log(params[0] + B.sum(axis=1)))
        # Return final value
        return -ll

    ### Calculate negative log-likelihood for the full model, for a specific node index
    def negative_loglikelihood_full(self, p, node_index, time_diffs_A, time_diffs_A_prime, subset_nodes):
        # Transform parameters to original scale (lambda, alpha, beta, theta, alpha_prime, beta_prime, theta_prime)
        params = np.exp(p)
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
        for node in subset_nodes:
            # Compensator components of loglikelihood
            ll += np.exp(-params[3] * ds[node]) * params[1] / params[2] * np.sum(np.exp(-params[2] * (self.T - self.start_times[node])) - 1)
            ll += np.exp(-params[6] * ds[node]) * params[4] / params[5] * np.sum(np.exp(-params[5] * (self.T - self.end_times[node])) - 1)
            # Loop over all events and update recursive terms A and A_prime
            for k, _ in enumerate(self.start_times[node_index]):    
                A[k, node] = ((np.exp(-params[2] * time_diffs[k-1]) * A[k-1, node]) if k > 0 else 0) + np.sum(np.exp(-params[2] * time_diffs_A[k, node])) 
                A_prime[k, node] = ((np.exp(-params[5] * time_diffs[k-1]) * A_prime[k-1, node]) if k > 0 else 0) + np.sum(np.exp(-params[5] * time_diffs_A_prime[k, node])) 
        # Calculate B and use it to update the log-likelihood
        B = np.exp(-params[3] * ds) * params[1] * A + np.exp(-params[6] * ds) * params[4] * A_prime
        ll += np.sum(np.log(params[0] + B.sum(axis=1)))
        # Return final value
        return -ll

    ## Calculate p-values for Poisson process
    def pvals_poisson_process(self, param, node_index):
        return np.exp(-param * np.insert(arr=np.diff(self.start_times[node_index]), obj=0, values=self.start_times[node_index][0]))

    ## Calculate p-values for self-exciting process
    def pvals_sep(self, params, node_index):
        # Time differences for starting times for node with corresponding index
        time_diffs = np.diff(self.start_times[node_index])
        # Pre-define arrays for the recursive terms (A)
        A = np.zeros(len(time_diffs)+1)
        # Loop over all events and update recursive term A
        for k, _ in enumerate(self.start_times[node_index]):    
            A[k] = np.exp(-params[2] * time_diffs[k-1]) * ((A[k-1] if k > 0 else 0) + 1)
        # Calculate B and use it to update the log-likelihood
        return np.exp(-params[0] * np.insert(arr=time_diffs, obj=0, values=self.start_times[node_index][0]) + params[1] / params[2] * np.insert(arr=np.diff(A)-1, obj=0, values=A[0]))