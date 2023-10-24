#!/usr/bin/env python3
import numpy as np
import argparse
import pickle
import os

## Check if directory exists
if not os.path.exists('results_2020/res_gbmep_start'):
   os.makedirs('results_2020/res_gbmep_start')

## Parser to give parameter values 
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--suffix', type=int, dest='suffix', default=0, const=True, nargs='?', help='Suffix for name of results and node subset.')

## Parse arguments
args = parser.parse_args()
suffix = args.suffix

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

## Obtain gb_mep object and expand DataFrame with the test set
G = gb_mep.gb_mep(df=santander_train, id_map=santander_dictionary, distance_matrix=santander_distances)

## Obtain node subset from the name of the suffix
lower = int(50*suffix)
upper = int(50*(suffix+1))
nodes_subset = G.nodes[lower:int(np.min([upper,789]))]

## Import benchmark results for initialisation
with open('results_2020/res_sep.pkl', 'rb') as f:
    res_sep = pickle.load(f)

## Obtain results via model fitting
start_vals = gb_mep.append_to_dictionary(d=res_sep, val=-1)
res_gbmep = G.fit(x0=start_vals, subset_nodes=nodes_subset, start_times=True, end_times=False, distance_start=True, distance_end=False, thresh=.5, min_nodes=3, shared_parameters=True)

## Save the results
with open('results_2020/res_gbmep_start/res_gbmep_start_' + str(suffix) + '.pkl', 'wb') as f:
    pickle.dump(res_gbmep, f)