#!/usr/bin/env python3
import numpy as np
import pickle
import os

## Check if directory exists
if not os.path.exists('results_2020'):
   os.makedirs('results_2020')

## Import library gb_mep
import gb_mep

## Load all required files after importing relevant libraries
import pickle
import numpy as np
import pandas as pd
santander = pd.read_csv('data/santander_summaries/santander_summaries_postprocessed/santander_2020.csv')
santander_train = santander[santander.start_time < (12 * 7 * 60 * 24)]
santander_test = santander[santander.start_time >= (12 * 7 * 60 * 24)]
santander_distances = np.load('data/santander_summaries/santander_summaries_postprocessed/santander_distances.npy')
santander_dictionary = {}
for i in np.unique(santander_train['start_id']):
    santander_dictionary[i] = i

## Obtain gb_mep object and expand DataFrame
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
with open('results_2020/res_pp.pkl', 'wb') as f:
    pickle.dump(res_pp, f)
## Save the results - Mutually exciting
with open('results_2020/res_mep.pkl', 'wb') as f:
    pickle.dump(res_mep, f)
## Save the results - Self-exciting process
with open('results_2020/res_sep.pkl', 'wb') as f:
    pickle.dump(res_sep, f)
## Save the results - Self-and-mutually exciting process
with open('results_2020/res_smep.pkl', 'wb') as f:
    pickle.dump(res_smep, f)