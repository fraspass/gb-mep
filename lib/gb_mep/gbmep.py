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
            # Find unique nodes (node IDs should start at 0 with maximum index M-1 and match the dictionary and ID mapping dictionary)
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

    ### Fit the Poisson process models 
    def fit_poisson(self, subset_nodes=None):
        # Define the dictionary for results
        res = {}
        # If the subset of nodes is not specified, consider all nodes
        if subset_nodes is None:
            subset_nodes = self.nodes
        # Loop over all nodes in the subset
        for node in subset_nodes:
            res[node] = self.N[node] / self.T
        # Return dictionary
        return res        

    ### Fit the model to a subset of nodes
    def fit(self, x0, subset_nodes=None, start_times=True, end_times=True, distance_start=False, distance_end=False, thresh=1, min_nodes=None, shared_parameters=True, optimiser='L-BFGS-B'):
        # Define the dictionary for results
        res = {}
        # If the subset of nodes is not specified, consider all nodes
        if subset_nodes is None:
            subset_nodes = self.nodes
        # Loop over all nodes in the subset
        for node in subset_nodes:
            # Print node and name
            print('\r', node, '-', self.id_map[node], ' '*20, end='\r')
            # Initial values for optimisation
            if isinstance(x0, dict):
                if node in x0:
                    if hasattr(x0[node],'x'):
                        starting_values = x0[node].x
                    else:
                        starting_values = x0[node]
                else:
                    return ValueError('Node' + str(node) + 'is not included in dictionary for initialisation')
            else:
                starting_values = x0
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
                if not condition:
                    neighbours = np.intersect1d(ar1=self.nodes, ar2=np.where(ds < thresh)[0])
                    if min_nodes is not None and min_nodes > len(neighbours):
                        neighbours = np.intersect1d(ar1=self.nodes, ar2=np.where(ds < np.sort(ds)[min_nodes])[0])
                for secondary_node in ([node] if condition else neighbours):
                    start_breaks = np.searchsorted(a=self.start_times[secondary_node], v=self.start_times[node], side='left')
                    end_breaks = np.searchsorted(a=self.end_times[secondary_node], v=self.start_times[node], side='left')
                    for k, t in enumerate(self.start_times[node]):
                        if k > 0:
                            time_diffs_A[k, secondary_node] = t - self.start_times[secondary_node][start_breaks[k-1]:start_breaks[k]]
                            time_diffs_A_prime[k, secondary_node] = t - self.end_times[secondary_node][end_breaks[k-1]:end_breaks[k]]
                        else:
                            time_diffs_A[k, secondary_node] = t - self.start_times[secondary_node][:start_breaks[k]]
                            time_diffs_A_prime[k, secondary_node] = t - self.end_times[secondary_node][:end_breaks[k]]
            # Obtain the correct log-likelihood function based on fitting parameters
            add_neighbours = False
            if start_times and end_times and distance_start and distance_end:
                f = self.negative_loglikelihood_gbmep_nonshared if not shared_parameters else self.negative_loglikelihood_gbmep
                f_args = (node, time_diffs_A, time_diffs_A_prime, neighbours)
                add_neighbours = True
            elif start_times and end_times and distance_start and not distance_end:
                f = self.negative_loglikelihood_gbmep_start_self_nonshared if not shared_parameters else self.negative_loglikelihood_gbmep_start_self
                f_args = (node, time_diffs_A, time_diffs_A_prime, neighbours)
                add_neighbours = True
            elif start_times and not end_times and distance_start:
                f = self.negative_loglikelihood_gbmep_start_nonshared if not shared_parameters else self.negative_loglikelihood_gbmep_start
                f_args = (node, time_diffs_A, neighbours)
                add_neighbours = True
            elif start_times and end_times and not distance_start and not distance_end:
                f = self.negative_loglikelihood_smep
                f_args = (node, time_diffs_A, time_diffs_A_prime)
            elif start_times and not end_times:
                f = self.negative_loglikelihood_sep
                f_args = (node)
            elif not start_times and end_times:
                f = self.negative_loglikelihood_mep
                f_args = (node, time_diffs_A_prime)
            elif not start_times and not end_times:
                f = 'Poisson'
            else:
                return ValueError('The chosen combination of start_times, end_times, distance_start and distance_end is not supported by this library.')
            # Minimise negative log-likelihood, or obtain exact solution if the function is simply the Poisson process
            if f == 'Poisson':
                res[node] = np.log(self.N[node] / self.T)
            else:
                res[node] = minimize(fun=f, x0=starting_values, args=f_args, method=optimiser)
                if add_neighbours:
                    res[node].subset_nodes = neighbours
        return res

    ### Calculate negative log-likelihood for the self-exciting model, for a specific node index
    def negative_loglikelihood_sep(self, p, node_index):
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
            A[k] = np.exp(-params[2] * time_diffs[k-1]) * ((A[k-1] + 1) if k > 0 else 0)
        # Calculate B and use it to update the log-likelihood
        B = params[1] * A
        ll += np.sum(np.log(params[0] + B))
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
        ll += np.sum(np.log(params[0] + B))
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
        ll += np.sum(np.log(params[0] + B))
        # Return final value
        return -ll

    ### Calculate negative log-likelihood for the full model without a distance function for the end times, for a specific node index
    def negative_loglikelihood_gbmep_start(self, p, node_index, time_diffs_A, subset_nodes):
        # Transform parameters to original scale (lambda, alpha, beta, theta)
        params = np.exp(p)
        params[2] += params[1]
        # Time differences for starting times for node with corresponding index
        time_diffs = np.diff(self.start_times[node_index])
        # Obtain distance between node and all other nodes
        ds = self.distance_matrix[node_index]
        # Pre-define arrays for the recursive terms (A)
        A = np.zeros((len(time_diffs)+1, len(subset_nodes)))
        # Compensator component of loglikelihood
        ll = -params[0] * self.T
        # Loop over all nodes
        for nn, node in enumerate(subset_nodes):
            # Compensator components of loglikelihood
            ll += np.exp(-params[3] * ds[node]) * params[1] / params[2] * np.sum(np.exp(-params[2] * (self.T - self.start_times[node])) - 1)
            # Loop over all events and update recursive terms A and A_prime
            for k, _ in enumerate(self.start_times[node_index]):    
                A[k, nn] = ((np.exp(-params[2] * time_diffs[k-1]) * A[k-1, nn]) if k > 0 else 0) + np.sum(np.exp(-params[2] * time_diffs_A[k, node]))
        # Calculate B and use it to update the log-likelihood
        B = np.exp(-params[3] * ds[subset_nodes]) * params[1] * A
        ll += np.sum(np.log(params[0] + B.sum(axis=1)))
        # Return final value
        return -ll
    
    ### Calculate negative log-likelihood for the full model without a distance function for the end times, for a specific node index
    def negative_loglikelihood_gbmep_start_self(self, p, node_index, time_diffs_A, time_diffs_A_prime, subset_nodes):
        # Transform parameters to original scale (lambda, alpha, beta, theta, alpha_prime, beta_prime)
        params = np.exp(p)
        params[2] += params[1]
        params[5] += params[4]
        # Time differences for starting times for node with corresponding index
        time_diffs = np.diff(self.start_times[node_index])
        # Obtain distance between node and all other nodes
        ds = self.distance_matrix[node_index]
        # Pre-define arrays for the recursive terms (A and A_prime)
        A = np.zeros((len(time_diffs)+1, len(subset_nodes)))
        A_prime = np.zeros((len(time_diffs)+1, len(subset_nodes)))
        # Compensator component of loglikelihood
        ll = -params[0] * self.T
        # Loop over all nodes
        for nn, node in enumerate(subset_nodes):
            # Compensator components of loglikelihood
            ll += np.exp(-params[3] * ds[node]) * params[1] / params[2] * np.sum(np.exp(-params[2] * (self.T - self.start_times[node])) - 1)
            if node == node_index:
                ll += params[4] / params[5] * np.sum(np.exp(-params[5] * (self.T - self.end_times[node])) - 1)
            # Loop over all events and update recursive terms A and A_prime
            for k, _ in enumerate(self.start_times[node_index]):    
                A[k, nn] = ((np.exp(-params[2] * time_diffs[k-1]) * A[k-1, nn]) if k > 0 else 0) + np.sum(np.exp(-params[2] * time_diffs_A[k, node]))
                if node == node_index:
                    A_prime[k, nn] = ((np.exp(-params[5] * time_diffs[k-1]) * A_prime[k-1, nn]) if k > 0 else 0) + np.sum(np.exp(-params[5] * time_diffs_A_prime[k, node])) 
        # Calculate B and use it to update the log-likelihood
        B = np.exp(-params[3] * ds[subset_nodes]) * params[1] * A + params[4] * A_prime
        ll += np.sum(np.log(params[0] + B.sum(axis=1)))
        # Return final value
        return -ll

    ### Calculate negative log-likelihood for the full model, for a specific node index
    def negative_loglikelihood_gbmep(self, p, node_index, time_diffs_A, time_diffs_A_prime, subset_nodes):
        # Transform parameters to original scale (lambda, alpha, beta, theta, alpha_prime, beta_prime, theta_prime)
        params = np.exp(p)
        params[2] += params[1]
        params[5] += params[4]
        # Time differences for starting times for node with corresponding index
        time_diffs = np.diff(self.start_times[node_index])
        # Obtain distance between node and all other nodes
        ds = self.distance_matrix[node_index]
        # Pre-define arrays for the recursive terms (A and A_prime)
        A = np.zeros((len(time_diffs)+1, len(subset_nodes)))
        A_prime = np.zeros((len(time_diffs)+1, len(subset_nodes)))
        # Compensator component of loglikelihood
        ll = -params[0] * self.T
        # Loop over all nodes
        for nn, node in enumerate(subset_nodes):
            # Compensator components of loglikelihood
            ll += np.exp(-params[3] * ds[node]) * params[1] / params[2] * np.sum(np.exp(-params[2] * (self.T - self.start_times[node])) - 1)
            ll += np.exp(-params[6] * ds[node]) * params[4] / params[5] * np.sum(np.exp(-params[5] * (self.T - self.end_times[node])) - 1)
            # Loop over all events and update recursive terms A and A_prime
            for k, _ in enumerate(self.start_times[node_index]):    
                A[k, nn] = ((np.exp(-params[2] * time_diffs[k-1]) * A[k-1, nn]) if k > 0 else 0) + np.sum(np.exp(-params[2] * time_diffs_A[k, node])) 
                A_prime[k, nn] = ((np.exp(-params[5] * time_diffs[k-1]) * A_prime[k-1, nn]) if k > 0 else 0) + np.sum(np.exp(-params[5] * time_diffs_A_prime[k, node])) 
        # Calculate B and use it to update the log-likelihood
        B = np.exp(-params[3] * ds[subset_nodes]) * params[1] * A + np.exp(-params[6] * ds[subset_nodes]) * params[4] * A_prime
        ll += np.sum(np.log(params[0] + B.sum(axis=1)))
        # Return final value
        return -ll
    
    ### Calculate negative log-likelihood for the full model without a distance function for the end times, for a specific node index
    def negative_loglikelihood_gbmep_start_nonshared(self, p, node_index, time_diffs_A, subset_nodes):
        # Transform parameters to original scale (lambda, alpha, beta, theta)
        params = np.exp(p)
        params[2] += params[1]
        params[5] += params[4]
        # Time differences for starting times for node with corresponding index
        time_diffs = np.diff(self.start_times[node_index])
        # Obtain distance between node and all other nodes
        ds = self.distance_matrix[node_index]
        # Pre-define arrays for the recursive terms (A)
        A = np.zeros((len(time_diffs)+1, len(subset_nodes)))
        # Compensator component of loglikelihood
        ll = -params[0] * self.T
        # Loop over all nodes
        for nn, node in enumerate(subset_nodes):
            # Compensator components of loglikelihood
            if node == node_index:
                ll += params[1] / params[2] * np.sum(np.exp(-params[2] * (self.T - self.start_times[node])) - 1)
            else:
                ll += np.exp(-params[3] * ds[node]) * params[4] / params[5] * np.sum(np.exp(-params[5] * (self.T - self.start_times[node])) - 1)
            # Loop over all events and update recursive terms A and A_prime
            for k, _ in enumerate(self.start_times[node_index]):
                if node == node_index:
                    A[k, nn] += params[1] * (((np.exp(-params[2] * time_diffs[k-1]) * A[k-1, nn]) if k > 0 else 0) + np.sum(np.exp(-params[2] * time_diffs_A[k, node])))
                else:
                    A[k, nn] += params[4] * (((np.exp(-params[5] * time_diffs[k-1]) * A[k-1, nn]) if k > 0 else 0) + np.sum(np.exp(-params[5] * time_diffs_A[k, node])))
        # Calculate B and use it to update the log-likelihood
        B = np.exp(-params[3] * ds[subset_nodes]) * A
        ll += np.sum(np.log(params[0] + B.sum(axis=1)))
        # Return final value
        return -ll
    
    ### Calculate negative log-likelihood for the full model without a distance function for the end times, for a specific node index
    def negative_loglikelihood_gbmep_start_self_nonshared(self, p, node_index, time_diffs_A, time_diffs_A_prime, subset_nodes):
        # Transform parameters to original scale (lambda, alpha, beta, theta, alpha_prime, beta_prime)
        params = np.exp(p)
        params[2] += params[1]
        params[5] += params[4]
        params[7] += params[6]
        # Time differences for starting times for node with corresponding index
        time_diffs = np.diff(self.start_times[node_index])
        # Obtain distance between node and all other nodes
        ds = self.distance_matrix[node_index]
        # Pre-define arrays for the recursive terms (A and A_prime)
        A = np.zeros((len(time_diffs)+1, len(subset_nodes)))
        A_prime = np.zeros((len(time_diffs)+1, len(subset_nodes)))
        # Compensator component of loglikelihood
        ll = -params[0] * self.T
        # Loop over all nodes
        for nn, node in enumerate(subset_nodes):
            # Compensator components of loglikelihood
            if node == node_index:
                ll += params[1] / params[2] * np.sum(np.exp(-params[2] * (self.T - self.start_times[node])) - 1)
                ll += params[6] / params[7] * np.sum(np.exp(-params[7] * (self.T - self.end_times[node])) - 1)
            else:
                ll += np.exp(-params[3] * ds[node]) * params[4] / params[5] * np.sum(np.exp(-params[5] * (self.T - self.start_times[node])) - 1)
            # Loop over all events and update recursive terms A and A_prime
            for k, _ in enumerate(self.start_times[node_index]):    
                if node == node_index:
                    A[k, nn] = params[1] * (((np.exp(-params[2] * time_diffs[k-1]) * A[k-1, nn]) if k > 0 else 0) + np.sum(np.exp(-params[2] * time_diffs_A[k, node])))
                    A_prime[k, nn] = params[6] * (((np.exp(-params[7] * time_diffs[k-1]) * A_prime[k-1, nn]) if k > 0 else 0) + np.sum(np.exp(-params[7] * time_diffs_A_prime[k, node])))
                else:
                    A[k, nn] = params[4] * (((np.exp(-params[5] * time_diffs[k-1]) * A[k-1, nn]) if k > 0 else 0) + np.sum(np.exp(-params[5] * time_diffs_A[k, node])))
        # Calculate B and use it to update the log-likelihood
        B = np.exp(-params[3] * ds[subset_nodes]) * A + A_prime
        ll += np.sum(np.log(params[0] + B.sum(axis=1)))
        # Return final value
        return -ll

    ### Calculate negative log-likelihood for the full model, for a specific node index
    def negative_loglikelihood_gbmep_nonshared(self, p, node_index, time_diffs_A, time_diffs_A_prime, subset_nodes):
        # Transform parameters to original scale (lambda, alpha, beta, theta, alpha_prime, beta_prime, theta_prime)
        params = np.exp(p)
        params[2] += params[1]
        params[5] += params[4]
        params[7] += params[6]
        params[10] += params[9]
        # Time differences for starting times for node with corresponding index
        time_diffs = np.diff(self.start_times[node_index])
        # Obtain distance between node and all other nodes
        ds = self.distance_matrix[node_index]
        # Pre-define arrays for the recursive terms (A and A_prime)
        A = np.zeros((len(time_diffs)+1, len(subset_nodes)))
        A_prime = np.zeros((len(time_diffs)+1, len(subset_nodes)))
        # Compensator component of loglikelihood
        ll = -params[0] * self.T
        # Loop over all nodes
        for nn, node in enumerate(subset_nodes):
            # Compensator components of loglikelihood
            if node == node_index:
                ll += params[1] / params[2] * np.sum(np.exp(-params[2] * (self.T - self.start_times[node])) - 1)
                ll += params[6] / params[7] * np.sum(np.exp(-params[7] * (self.T - self.end_times[node])) - 1)
            else:
                ll += np.exp(-params[3] * ds[node]) * params[4] / params[5] * np.sum(np.exp(-params[5] * (self.T - self.start_times[node])) - 1)
                ll += np.exp(-params[8] * ds[node]) * params[9] / params[10] * np.sum(np.exp(-params[10] * (self.T - self.end_times[node])) - 1)
            # Loop over all events and update recursive terms A and A_prime
            for k, _ in enumerate(self.start_times[node_index]):
                if node == node_index:
                    A[k, nn] = params[1] * (((np.exp(-params[2] * time_diffs[k-1]) * A[k-1, nn]) if k > 0 else 0) + np.sum(np.exp(-params[2] * time_diffs_A[k, node])))
                    A_prime[k, nn] = params[6] * (((np.exp(-params[7] * time_diffs[k-1]) * A_prime[k-1, nn]) if k > 0 else 0) + np.sum(np.exp(-params[7] * time_diffs_A_prime[k, node])))
                else:
                    A[k, nn] = params[4] * (((np.exp(-params[5] * time_diffs[k-1]) * A[k-1, nn]) if k > 0 else 0) + np.sum(np.exp(-params[5] * time_diffs_A[k, node])))
                    A_prime[k, nn] = params[9] * (((np.exp(-params[10] * time_diffs[k-1]) * A_prime[k-1, nn]) if k > 0 else 0) + np.sum(np.exp(-params[10] * time_diffs_A_prime[k, node]))) 
        # Calculate B and use it to update the log-likelihood
        B = np.exp(-params[3] * ds[subset_nodes]) * A + np.exp(-params[8] * ds[subset_nodes]) * A_prime
        ll += np.sum(np.log(params[0] + B.sum(axis=1)))
        # Return final value
        return -ll

    ## Add test set event times to the training set event times
    def augment_start_times(self, test_set):
        # For each node, find all start times and end times in the test set and add them to the existing set
        start_times = {}
        end_times = {}
        for node in self.nodes:
            start_times[node] = np.sort(np.concatenate((self.start_times[node], test_set['start_time'][test_set['start_id'] == node])))
            end_times[node] = np.sort(np.concatenate((self.end_times[node], test_set['end_time'][test_set['start_id'] == node])))
        return start_times, end_times

    ## Calculate p-values for Poisson process
    def pvals_poisson_process(self, param, node_index, start_times=None, test_split=False):
        if start_times is None:
            start_times = self.start_times
        # Calculate p-values
        pvs = np.exp(-param * np.insert(arr=np.diff(start_times[node_index]), obj=0, values=start_times[node_index][0]))
        if not test_split:
            return pvs
        else:
            return pvs[:self.N[node_index]], pvs[self.N[node_index]:]

    ## Calculate p-values for self-exciting process
    def pvals_sep(self, params, node_index, start_times=None, test_split=False):
        if start_times is None:
            start_times = self.start_times
        # Time differences for starting times for node with corresponding index
        time_diffs = np.diff(start_times[node_index])
        # Pre-define arrays for the recursive terms (A)
        A = np.zeros(len(time_diffs)+1)
        # Loop over all events and update recursive term A
        for k, _ in enumerate(start_times[node_index]):    
            A[k] = np.exp(-params[2] * time_diffs[k-1]) * ((A[k-1] if k > 0 else 0) + 1)
        # Calculate p-values
        pvs = np.exp(-params[0] * np.insert(arr=time_diffs, obj=0, values=start_times[node_index][0]) + params[1] / params[2] * np.insert(arr=np.diff(A)-1, obj=0, values=A[0]))
        if not test_split:
            return pvs
        else:
            return pvs[:self.N[node_index]], pvs[self.N[node_index]:]
    
    ## Calculate p-values for mutually exciting process
    def pvals_mep(self, params, node_index, start_times=None, end_times=None, test_split=False):
        if start_times is None:
            start_times = self.start_times
        if end_times is None:
            end_times = self.end_times
        # Time differences for starting times for node with corresponding index
        time_diffs = np.diff(start_times[node_index])
        # Pre-define arrays for the recursive terms (A_prime)
        A_prime = np.zeros(len(time_diffs)+1)
        # Counting process N_i^prime evaluated at all start times
        end_breaks = np.searchsorted(a=end_times[node_index], v=start_times[node_index], side='left')
        end_breaks_diff = np.insert(arr=np.diff(end_breaks), obj=0, values=end_breaks[0])
        # Loop over all events and update recursive terms A and A_prime
        for k, t in enumerate(start_times[node_index]):
            if k > 0:
                t_primes = t - end_times[node_index][end_breaks[k-1]:end_breaks[k]]
            else:
                t_primes = t - end_times[node_index][:end_breaks[k]]
            A_prime[k] = ((np.exp(-params[2] * time_diffs[k-1]) * A_prime[k-1]) if k > 0 else 0) + np.sum(np.exp(-params[2] * t_primes)) 
        # Calculate p-values
        baseline_terms = -params[0] * np.insert(arr=time_diffs, obj=0, values=start_times[node_index][0])
        pvs = np.exp(baseline_terms + params[1] / params[2] * (np.insert(arr=np.diff(A_prime), obj=0, values=A_prime[0]) - end_breaks_diff))
        if not test_split:
            return pvs
        else:
            return pvs[:self.N[node_index]], pvs[self.N[node_index]:]
    
    ## Calculate p-values for self-and-mutually exciting process
    def pvals_smep(self, params, node_index, start_times=None, end_times=None, test_split=False):
        if start_times is None:
            start_times = self.start_times
        if end_times is None:
            end_times = self.end_times
        # Time differences for starting times for node with corresponding index
        time_diffs = np.diff(start_times[node_index])
        # Pre-define arrays for the recursive terms (A and A_prime)
        A = np.zeros(len(time_diffs)+1)
        A_prime = np.zeros(len(A))
        ## Counting process N_i^prime evaluated at all start times
        end_breaks = np.searchsorted(a=end_times[node_index], v=start_times[node_index], side='left')
        end_breaks_diff = np.insert(arr=np.diff(end_breaks), obj=0, values=end_breaks[0])
        # Loop over all events and update recursive term A and A_prime
        for k, t in enumerate(start_times[node_index]):
            if k > 0:
                t_primes = t - end_times[node_index][end_breaks[k-1]:end_breaks[k]] 
            else:
                t_primes = t - end_times[node_index][:end_breaks[k]]    
            A[k] = np.exp(-params[2] * time_diffs[k-1]) * ((A[k-1] if k > 0 else 0) + 1)
            A_prime[k] = ((np.exp(-params[4] * time_diffs[k-1]) * A_prime[k-1]) if k > 0 else 0) + np.sum(np.exp(-params[4] * t_primes)) 
        # Calculate p-values
        baseline_terms = -params[0] * np.insert(arr=time_diffs, obj=0, values=start_times[node_index][0])
        A_terms = params[1] / params[2] * np.insert(arr=np.diff(A)-1, obj=0, values=A[0])
        A_prime_terms = params[3] / params[4] * (np.insert(arr=np.diff(A_prime), obj=0, values=A_prime[0]) - end_breaks_diff)
        pvs = np.exp(baseline_terms + A_terms + A_prime_terms)
        if not test_split:
            return pvs
        else:
            return pvs[:self.N[node_index]], pvs[self.N[node_index]:]
    
    ## Calculate p-values for GB-MEP with distance function
    def pvals_gbmep_start(self, params, node_index, subset_nodes, start_times=None, end_times=None, test_split=False):
        if start_times is None:
            start_times = self.start_times
        if end_times is None:
            end_times = self.end_times
        # Time differences for starting times for node with corresponding index
        time_diffs = np.diff(start_times[node_index])
        # Pre-define arrays for the recursive terms (A)
        A = np.zeros((len(time_diffs)+1))
        # Obtain distance between node and all other nodes
        ds = self.distance_matrix[node_index]
        # Calculate baseline terms for p-values
        baseline_terms = -params[0] * np.insert(arr=time_diffs, obj=0, values=start_times[node_index][0])
        A_terms = np.zeros(len(A))
        # Calculate required elements for recursion
        for secondary_node in subset_nodes:
            start_breaks = np.searchsorted(a=start_times[secondary_node], v=start_times[node_index], side='left')
            start_breaks_diff = np.insert(arr=np.diff(start_breaks), obj=0, values=start_breaks[0])
            for k, t in enumerate(start_times[node_index]):
                if k > 0:
                    time_diffs_A = t - start_times[secondary_node][start_breaks[k-1]:start_breaks[k]]
                else:
                    time_diffs_A = t - start_times[secondary_node][:start_breaks[k]]
                ## Update A and A_prime
                A[k] = ((np.exp(-params[2] * time_diffs[k-1]) * A[k-1]) if k > 0 else 0) + np.sum(np.exp(-params[2] * time_diffs_A)) 
            ## Update A and A_prime terms for calculation of differences between conpensators
            A_terms += np.exp(-params[3] * ds[secondary_node]) * params[1] / params[2] * (np.insert(arr=np.diff(A), obj=0, values=A[0]) - start_breaks_diff)
        ## Return p-values
        pvs = np.exp(baseline_terms + A_terms)
        if not test_split:
            return pvs
        else:
            return pvs[:self.N[node_index]], pvs[self.N[node_index]:]

    ## Calculate p-values for GB-MEP with distance function
    def pvals_gbmep_start_self(self, params, node_index, subset_nodes, start_times=None, end_times=None, test_split=False):
        if start_times is None:
            start_times = self.start_times
        if end_times is None:
            end_times = self.end_times
        # Time differences for starting times for node with corresponding index
        time_diffs = np.diff(start_times[node_index])
        # Pre-define arrays for the recursive terms (A and A_prime)
        A = np.zeros((len(time_diffs)+1))
        A_prime = np.zeros((len(time_diffs)+1))
        # Obtain distance between node and all other nodes
        ds = self.distance_matrix[node_index]
        # Calculate baseline terms for p-values
        baseline_terms = -params[0] * np.insert(arr=time_diffs, obj=0, values=start_times[node_index][0])
        A_terms = np.zeros(len(A)); A_prime_terms = np.zeros(len(A_prime))
        # Calculate required elements for recursion
        for secondary_node in subset_nodes:
            start_breaks = np.searchsorted(a=start_times[secondary_node], v=start_times[node_index], side='left')
            start_breaks_diff = np.insert(arr=np.diff(start_breaks), obj=0, values=start_breaks[0])
            if secondary_node == node_index:
                end_breaks = np.searchsorted(a=end_times[secondary_node], v=start_times[node_index], side='left')
                end_breaks_diff = np.insert(arr=np.diff(end_breaks), obj=0, values=end_breaks[0])
            for k, t in enumerate(start_times[node_index]):
                if k > 0:
                    time_diffs_A = t - start_times[secondary_node][start_breaks[k-1]:start_breaks[k]]
                    if secondary_node == node_index:
                        time_diffs_A_prime = t - end_times[secondary_node][end_breaks[k-1]:end_breaks[k]]
                else:
                    time_diffs_A = t - start_times[secondary_node][:start_breaks[k]]
                    if secondary_node == node_index:
                        time_diffs_A_prime = t - end_times[secondary_node][:end_breaks[k]]
                ## Update A and A_prime
                A[k] = ((np.exp(-params[2] * time_diffs[k-1]) * A[k-1]) if k > 0 else 0) + np.sum(np.exp(-params[2] * time_diffs_A)) 
                if secondary_node == node_index:
                    A_prime[k] = ((np.exp(-params[4] * time_diffs[k-1]) * A_prime[k-1]) if k > 0 else 0) + np.sum(np.exp(-params[4] * time_diffs_A_prime)) 
            ## Update A and A_prime terms for calculation of differences between conpensators
            A_terms += np.exp(-params[3] * ds[secondary_node]) * params[1] / params[2] * (np.insert(arr=np.diff(A), obj=0, values=A[0]) - start_breaks_diff)
            if secondary_node == node_index:
                A_prime_terms += params[4] / params[5] * (np.insert(arr=np.diff(A_prime), obj=0, values=A_prime[0]) - end_breaks_diff)
        ## Return p-values
        pvs = np.exp(baseline_terms + A_terms + A_prime_terms)
        if not test_split:
            return pvs
        else:
            return pvs[:self.N[node_index]], pvs[self.N[node_index]:]

    ## Calculate p-values for GB-MEP with distance function
    def pvals_gbmep(self, params, node_index, subset_nodes, start_times=None, end_times=None, test_split=False):
        if start_times is None:
            start_times = self.start_times
        if end_times is None:
            end_times = self.end_times
        # Time differences for starting times for node with corresponding index
        time_diffs = np.diff(start_times[node_index])
        # Pre-define arrays for the recursive terms (A and A_prime)
        A = np.zeros((len(time_diffs)+1))
        A_prime = np.zeros((len(time_diffs)+1))
        # Obtain distance between node and all other nodes
        ds = self.distance_matrix[node_index]
        # Calculate baseline terms for p-values
        baseline_terms = -params[0] * np.insert(arr=time_diffs, obj=0, values=start_times[node_index][0])
        A_terms = np.zeros(len(A)); A_prime_terms = np.zeros(len(A_prime))
        # Calculate required elements for recursion
        for secondary_node in subset_nodes:
            start_breaks = np.searchsorted(a=start_times[secondary_node], v=start_times[node_index], side='left')
            start_breaks_diff = np.insert(arr=np.diff(start_breaks), obj=0, values=start_breaks[0])
            end_breaks = np.searchsorted(a=end_times[secondary_node], v=start_times[node_index], side='left')
            end_breaks_diff = np.insert(arr=np.diff(end_breaks), obj=0, values=end_breaks[0])
            for k, t in enumerate(start_times[node_index]):
                if k > 0:
                    time_diffs_A = t - start_times[secondary_node][start_breaks[k-1]:start_breaks[k]]
                    time_diffs_A_prime = t - end_times[secondary_node][end_breaks[k-1]:end_breaks[k]]
                else:
                    time_diffs_A = t - start_times[secondary_node][:start_breaks[k]]
                    time_diffs_A_prime = t - end_times[secondary_node][:end_breaks[k]]
                ## Update A and A_prime
                A[k] = ((np.exp(-params[2] * time_diffs[k-1]) * A[k-1]) if k > 0 else 0) + np.sum(np.exp(-params[2] * time_diffs_A)) 
                A_prime[k] = ((np.exp(-params[4] * time_diffs[k-1]) * A_prime[k-1]) if k > 0 else 0) + np.sum(np.exp(-params[4] * time_diffs_A_prime)) 
            ## Update A and A_prime terms for calculation of differences between conpensators
            A_terms += np.exp(-params[3] * ds[secondary_node]) * params[1] / params[2] * (np.insert(arr=np.diff(A), obj=0, values=A[0]) - start_breaks_diff)
            A_prime_terms += np.exp(-params[6] * ds[secondary_node]) * params[4] / params[5] * (np.insert(arr=np.diff(A_prime), obj=0, values=A_prime[0]) - end_breaks_diff)
        ## Return p-values
        pvs = np.exp(baseline_terms + A_terms + A_prime_terms)
        if not test_split:
            return pvs
        else:
            return pvs[:self.N[node_index]], pvs[self.N[node_index]:]
        
    ## Calculate p-values for GB-MEP with distance function
    def pvals_gbmep_start_nonshared(self, params, node_index, subset_nodes, start_times=None, end_times=None, test_split=False):
        if start_times is None:
            start_times = self.start_times
        if end_times is None:
            end_times = self.end_times
        # Time differences for starting times for node with corresponding index
        time_diffs = np.diff(start_times[node_index])
        # Pre-define arrays for the recursive terms (A)
        A = np.zeros((len(time_diffs)+1))
        # Obtain distance between node and all other nodes
        ds = self.distance_matrix[node_index]
        # Calculate baseline terms for p-values
        baseline_terms = -params[0] * np.insert(arr=time_diffs, obj=0, values=start_times[node_index][0])
        A_terms = np.zeros(len(A))
        # Calculate required elements for recursion
        for secondary_node in subset_nodes:
            start_breaks = np.searchsorted(a=start_times[secondary_node], v=start_times[node_index], side='left')
            start_breaks_diff = np.insert(arr=np.diff(start_breaks), obj=0, values=start_breaks[0])
            for k, t in enumerate(start_times[node_index]):
                if k > 0:
                    time_diffs_A = t - start_times[secondary_node][start_breaks[k-1]:start_breaks[k]]
                else:
                    time_diffs_A = t - start_times[secondary_node][:start_breaks[k]]
                ## Update A and A_prime
                if secondary_node == node_index:
                    A[k] = ((np.exp(-params[2] * time_diffs[k-1]) * A[k-1]) if k > 0 else 0) + np.sum(np.exp(-params[2] * time_diffs_A))
                else:
                    A[k] = ((np.exp(-params[4] * time_diffs[k-1]) * A[k-1]) if k > 0 else 0) + np.sum(np.exp(-params[4] * time_diffs_A))
            ## Update A and A_prime terms for calculation of differences between conpensators
            if secondary_node == node_index:
                A_terms += params[1] / params[2] * (np.insert(arr=np.diff(A), obj=0, values=A[0]) - start_breaks_diff)
            else:
                A_terms += np.exp(-params[3] * ds[secondary_node]) * params[4] / params[5] * (np.insert(arr=np.diff(A), obj=0, values=A[0]) - start_breaks_diff)
        ## Return p-values
        pvs = np.exp(baseline_terms + A_terms)
        if not test_split:
            return pvs
        else:
            return pvs[:self.N[node_index]], pvs[self.N[node_index]:]

    ## Calculate p-values for GB-MEP with distance function
    def pvals_gbmep_start_self_nonshared(self, params, node_index, subset_nodes, start_times=None, end_times=None, test_split=False):
        if start_times is None:
            start_times = self.start_times
        if end_times is None:
            end_times = self.end_times
        # Time differences for starting times for node with corresponding index
        time_diffs = np.diff(start_times[node_index])
        # Pre-define arrays for the recursive terms (A and A_prime)
        A = np.zeros((len(time_diffs)+1))
        A_prime = np.zeros((len(time_diffs)+1))
        # Obtain distance between node and all other nodes
        ds = self.distance_matrix[node_index]
        # Calculate baseline terms for p-values
        baseline_terms = -params[0] * np.insert(arr=time_diffs, obj=0, values=start_times[node_index][0])
        A_terms = np.zeros(len(A)); A_prime_terms = np.zeros(len(A_prime))
        # Calculate required elements for recursion
        for secondary_node in subset_nodes:
            start_breaks = np.searchsorted(a=start_times[secondary_node], v=start_times[node_index], side='left')
            start_breaks_diff = np.insert(arr=np.diff(start_breaks), obj=0, values=start_breaks[0])
            if secondary_node == node_index:
                end_breaks = np.searchsorted(a=end_times[secondary_node], v=start_times[node_index], side='left')
                end_breaks_diff = np.insert(arr=np.diff(end_breaks), obj=0, values=end_breaks[0])
            for k, t in enumerate(start_times[node_index]):
                if k > 0:
                    time_diffs_A = t - start_times[secondary_node][start_breaks[k-1]:start_breaks[k]]
                    if secondary_node == node_index:
                        time_diffs_A_prime = t - end_times[secondary_node][end_breaks[k-1]:end_breaks[k]]
                else:
                    time_diffs_A = t - start_times[secondary_node][:start_breaks[k]]
                    if secondary_node == node_index:
                        time_diffs_A_prime = t - end_times[secondary_node][:end_breaks[k]]
                ## Update A and A_prime
                if secondary_node == node_index:
                    A[k] = ((np.exp(-params[2] * time_diffs[k-1]) * A[k-1]) if k > 0 else 0) + np.sum(np.exp(-params[2] * time_diffs_A)) 
                    A_prime[k] = ((np.exp(-params[7] * time_diffs[k-1]) * A_prime[k-1]) if k > 0 else 0) + np.sum(np.exp(-params[7] * time_diffs_A_prime)) 
                else:
                    A[k] = ((np.exp(-params[5] * time_diffs[k-1]) * A[k-1]) if k > 0 else 0) + np.sum(np.exp(-params[5] * time_diffs_A)) 
            ## Update A and A_prime terms for calculation of differences between conpensators
            if secondary_node == node_index:
                A_terms += params[1] / params[2] * (np.insert(arr=np.diff(A), obj=0, values=A[0]) - start_breaks_diff)
                A_prime_terms += params[6] / params[7] * (np.insert(arr=np.diff(A_prime), obj=0, values=A_prime[0]) - end_breaks_diff)
            else:
                A_terms += np.exp(-params[3] * ds[secondary_node]) * params[4] / params[5] * (np.insert(arr=np.diff(A), obj=0, values=A[0]) - start_breaks_diff)
        ## Return p-values
        pvs = np.exp(baseline_terms + A_terms + A_prime_terms)
        if not test_split:
            return pvs
        else:
            return pvs[:self.N[node_index]], pvs[self.N[node_index]:]

    ## Calculate p-values for GB-MEP with distance function
    def pvals_gbmep_nonshared(self, params, node_index, subset_nodes, start_times=None, end_times=None, test_split=False):
        if start_times is None:
            start_times = self.start_times
        if end_times is None:
            end_times = self.end_times
        # Time differences for starting times for node with corresponding index
        time_diffs = np.diff(start_times[node_index])
        # Pre-define arrays for the recursive terms (A and A_prime)
        A = np.zeros((len(time_diffs)+1))
        A_prime = np.zeros((len(time_diffs)+1))
        # Obtain distance between node and all other nodes
        ds = self.distance_matrix[node_index]
        # Calculate baseline terms for p-values
        baseline_terms = -params[0] * np.insert(arr=time_diffs, obj=0, values=start_times[node_index][0])
        A_terms = np.zeros(len(A)); A_prime_terms = np.zeros(len(A_prime))
        # Calculate required elements for recursion
        for secondary_node in subset_nodes:
            start_breaks = np.searchsorted(a=start_times[secondary_node], v=start_times[node_index], side='left')
            start_breaks_diff = np.insert(arr=np.diff(start_breaks), obj=0, values=start_breaks[0])
            end_breaks = np.searchsorted(a=end_times[secondary_node], v=start_times[node_index], side='left')
            end_breaks_diff = np.insert(arr=np.diff(end_breaks), obj=0, values=end_breaks[0])
            for k, t in enumerate(start_times[node_index]):
                if k > 0:
                    time_diffs_A = t - start_times[secondary_node][start_breaks[k-1]:start_breaks[k]]
                    time_diffs_A_prime = t - end_times[secondary_node][end_breaks[k-1]:end_breaks[k]]
                else:
                    time_diffs_A = t - start_times[secondary_node][:start_breaks[k]]
                    time_diffs_A_prime = t - end_times[secondary_node][:end_breaks[k]]
                ## Update A and A_prime
                if secondary_node == node_index:
                    A[k] = ((np.exp(-params[2] * time_diffs[k-1]) * A[k-1]) if k > 0 else 0) + np.sum(np.exp(-params[2] * time_diffs_A)) 
                    A_prime[k] = ((np.exp(-params[7] * time_diffs[k-1]) * A_prime[k-1]) if k > 0 else 0) + np.sum(np.exp(-params[7] * time_diffs_A_prime)) 
                else:
                    A[k] = ((np.exp(-params[5] * time_diffs[k-1]) * A[k-1]) if k > 0 else 0) + np.sum(np.exp(-params[5] * time_diffs_A)) 
                    A_prime[k] = ((np.exp(-params[10] * time_diffs[k-1]) * A_prime[k-1]) if k > 0 else 0) + np.sum(np.exp(-params[10] * time_diffs_A_prime)) 
            ## Update A and A_prime terms for calculation of differences between conpensators
            if secondary_node == node_index:
                A_terms += params[1] / params[2] * (np.insert(arr=np.diff(A), obj=0, values=A[0]) - start_breaks_diff)
                A_prime_terms += params[6] / params[7] * (np.insert(arr=np.diff(A_prime), obj=0, values=A_prime[0]) - end_breaks_diff)
            else:
                A_terms += np.exp(-params[3] * ds[secondary_node]) * params[4] / params[5] * (np.insert(arr=np.diff(A), obj=0, values=A[0]) - start_breaks_diff)
                A_prime_terms += np.exp(-params[8] * ds[secondary_node]) * params[9] / params[10] * (np.insert(arr=np.diff(A_prime), obj=0, values=A_prime[0]) - end_breaks_diff)
        ## Return p-values
        pvs = np.exp(baseline_terms + A_terms + A_prime_terms)
        if not test_split:
            return pvs
        else:
            return pvs[:self.N[node_index]], pvs[self.N[node_index]:]