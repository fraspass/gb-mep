#!/usr/bin/env python3
import numpy as np
import argparse
import pickle
import os

## Check if directory exists
if not os.path.exists('results/res_gbmep_start'):
   os.makedirs('results/res_gbmep_start')

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
santander_train = pd.read_csv('data/santander_train.csv')
santander_distances = np.load('data/santander_distances.npy')
with open('data/santander_dictionary.pkl', 'rb') as f:
    santander_dictionary = pickle.load(f)

## Obtain gb_mep object and expand DataFrame with the test set
G = gb_mep.gb_mep(df=santander_train, id_map=santander_dictionary, distance_matrix=santander_distances)

## Obtain node subset from the name of the suffix
lower = int(50*suffix)
upper = int(50*(suffix+1))
nodes_subset = G.nodes[lower:int(np.min([upper,798]))]

## Import benchmark results for initialisation
with open('results/res_sep.pkl', 'rb') as f:
    res_sep = pickle.load(f)

## Obtain results via model fitting
start_vals = gb_mep.append_to_dictionary(d=res_sep, val=1)
res_gbmep = G.fit(x0=start_vals, subset_nodes=nodes_subset, start_times=True, end_times=False, distance_start=True, distance_end=False, thresh=.5, min_nodes=3, shared_parameters=True)

## Save the results
with open('results/res_gbmep_start/res_gbmep_start_' + str(suffix) + '.pkl', 'wb') as f:
    pickle.dump(res_gbmep, f)