#!/usr/bin/env python3
import numpy as np
import pickle
import os

## Check if directory exists
if not os.path.exists('results'):
   os.makedirs('results')

## Import library gb_mep
import gb_mep

## Load all required files after importing relevant libraries
import pickle
import numpy as np
import pandas as pd
santander_train = pd.read_csv('data/santander_train.csv') 
santander_train.columns = ['end_id', 'start_id', 'end_time', 'start_time']
santander_distances = np.load('data/santander_distances.npy')
with open('data/santander_dictionary.pkl', 'rb') as f:
    santander_dictionary = pickle.load(f)

## Obtain gb_mep object and expand DataFrame with the test set
G = gb_mep.gb_mep(df=santander_train, id_map=santander_dictionary, distance_matrix=santander_distances)

## Poisson process fit
res_pp = G.fit_poisson()
## Mutually exciting process fit
res_mep = G.fit(x0=-4*np.ones(3), subset_nodes=G.nodes, start_times=False, end_times=True, distance_start=False, distance_end=False)
## Self-exciting process fit
res_sep = G.fit(x0=-4*np.ones(3), subset_nodes=G.nodes, start_times=True, end_times=False, distance_start=False, distance_end=False)
## Self-and-mutually exciting process fit
res_smep = G.fit(x0=-4*np.ones(5), subset_nodes=G.nodes, start_times=True, end_times=True, distance_start=False, distance_end=False)

## Save the results - Poisson process
with open('results/res_pp_end.pkl', 'wb') as f:
    pickle.dump(res_pp, f)
## Save the results - Mutually exciting
with open('results/res_mep_end.pkl', 'wb') as f:
    pickle.dump(res_mep, f)
## Save the results - Self-exciting process
with open('results/res_sep_end.pkl', 'wb') as f:
    pickle.dump(res_sep, f)
## Save the results - Self-and-mutually exciting process
with open('results/res_smep_end.pkl', 'wb') as f:
    pickle.dump(res_smep, f)