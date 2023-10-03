#!/usr/bin/env python3
import numpy as np
import argparse

## PARSER to give parameter values 
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--lower' type=int, dest='lower', default=0, const=True, nargs='?', help='Lower bound on the node ID')
parser.add_argument('-u', '--upper' type=int, dest='upper', default=810, const=True, nargs='?', help='Upper bound on the node ID')
parser.add_argument('-s', '--suffix' type=int, dest='suffix', default=0, const=True, nargs='?', help='Suffix for nome of results')

## Parse arguments
args = parser.parse_args()
lower = args.lower
upper = args.upper
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

## 
res_gbmep_self = G.fit(x0=-np.ones(4), subset_nodes=G.nodes[lower:upper], start_times=True, end_times=False, distance_start=True, distance_end=False, thresh=1)