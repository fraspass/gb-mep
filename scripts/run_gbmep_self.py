#!/usr/bin/env python3
import numpy as np
import argparse
import pickle

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
santander_test = pd.read_csv('data/santander_test.csv') 
santander_distances = np.load('data/santander_distances.npy')
with open('data/santander_dictionary.pkl', 'rb') as f:
    santander_dictionary = pickle.load(f)

## Obtain gb_mep object and expand DataFrame with the test set
G = gb_mep.gb_mep(df=santander_train, id_map=santander_dictionary, distance_matrix=santander_distances)
start_times, end_times = G.augment_start_times(santander_test)

## Obtain node subset from the name of the suffix
lower = int(50*suffix)
upper = int(50*(suffix+1))
nodes_subset = G.nodes[lower:int(np.min([upper,798]))]

## Obtain results via model fitting
res_gbmep = G.fit(x0=-np.ones(4), subset_nodes=nodes_subset, start_times=True, end_times=False, distance_start=True, distance_end=False, thresh=1)

## Save the results
with open('results/res_gbmep_self/res_gbmep_self_' + str(suffix) + '.pkl', 'wb') as f:
    pickle.dump(res_gbmep, f)