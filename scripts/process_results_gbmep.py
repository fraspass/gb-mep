#!/usr/bin/env python3
import numpy as np
import pickle
import glob, os
import scipy.stats as stats

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

## Load results for the Poisson process model
with open('results/res_pp.pkl', 'rb') as f:
    res_pp = pickle.load(f)

## Load results for the MEP model
with open('results/res_mep.pkl', 'rb') as f:
    res_mep = pickle.load(f)

## Load results for the SEP model
with open('results/res_sep.pkl', 'rb') as f:
    res_sep = pickle.load(f)

## Load results for the SMEP model
with open('results/res_smep.pkl', 'rb') as f:
    res_smep = pickle.load(f)

## Load results for the GB-MEP model without start times only
res_gbmep_start = {}
for file in glob.glob('results/res_gbmep_start/*.pkl'):
    with open(file, 'rb') as f:
        res_gbmep_start.update(pickle.load(f))

## Load results for the GB-MEP model
res_gbmep = {}
for file in glob.glob('results/res_gbmep/*.pkl'):
    with open(file, 'rb') as f:
        res_gbmep.update(pickle.load(f))

## Obtain gb_mep object and expand DataFrame with the test set
G = gb_mep.gb_mep(df=santander_train, id_map=santander_dictionary, distance_matrix=santander_distances)
start_times, end_times = G.augment_start_times(santander_test)

## Initialise dictionaries for observed quantiles
y_poisson_train = {}; y_poisson_test = {}; y_mep_train = {}; y_mep_test = {}; y_sep_train = {}; y_sep_test = {}
y_smep_train = {}; y_smep_test = {}; y_gbmep_start_train = {}; y_gbmep_start_test = {}; y_gbmep_train = {}; y_gbmep_test = {}

## Intialise dictionaries for KS scores
k_poisson_train = {}; k_mep_train = {}; k_sep_train = {}; k_smep_train = {}; k_gbmep_start_train = {}; k_gbmep_train = {}
k_poisson_test = {}; k_mep_test = {}; k_sep_test = {}; k_smep_test = {}; k_gbmep_start_test = {}; k_gbmep_test = {}

## Define theoretical quantiles
x = np.linspace(start=0, stop=1, num=501, endpoint=False)[1:]

## Loop over all nodes
for index in G.nodes:
    # Print index and station name
    print('\r', index, '-', G.id_map[index], ' '*20, end='\r')
    # p-values for Poisson process
    p_poisson_train, p_poisson_test = G.pvals_poisson_process(param=res_pp[index], node_index=index, start_times=start_times, test_split=True)
    # p-values for MEP
    pp = np.exp(res_mep[index].x)
    pp[2] += pp[1]
    p_mep_train, p_mep_test = G.pvals_mep(params=pp, node_index=index, start_times=start_times, end_times=end_times, test_split=True)
    # p-values for SEP
    pp = np.exp(res_sep[index].x)
    pp[2] += pp[1]
    p_sep_train, p_sep_test = G.pvals_sep(params=pp, node_index=index, start_times=start_times, test_split=True)
    # p-values for SMEP
    pp = np.exp(res_smep[index].x)
    pp[2] += pp[1]; pp[4] += pp[3]
    p_smep_train, p_smep_test = G.pvals_smep(params=pp, node_index=index, start_times=start_times, end_times=end_times, test_split=True)
    # p-values for GB-MEP with start times only
    pp = np.exp(res_gbmep_start[index].x)
    pp[2] += pp[1]
    p_gbmep_start_train, p_gbmep_start_test = G.pvals_gbmep_start(params=pp, node_index=index, thresh=1, start_times=start_times, test_split=True)
    # p-values for GB-MEP
    pp = np.exp(res_gbmep[index].x)
    pp[2] += pp[1]; pp[5] += pp[4]
    p_gbmep_train, p_gbmep_test = G.pvals_gbmep_start_self(params=pp, node_index=index, thresh=1, start_times=start_times, end_times=end_times, test_split=True)
    # Caclulate observed percentiles for training set
    y_poisson_train[index] = np.percentile(a=p_poisson_train, q=x*100)
    y_mep_train[index] = np.percentile(a=p_mep_train, q=x*100)
    y_sep_train[index] = np.percentile(a=p_sep_train, q=x*100)
    y_smep_train[index] = np.percentile(a=p_smep_train, q=x*100)
    y_gbmep_start_train[index] = np.percentile(a=p_gbmep_start_train, q=x*100)
    y_gbmep_train[index] = np.percentile(a=p_gbmep_train, q=x*100)
    # Caclulate observed percentiles for test set
    if len(p_poisson_test) > 0:
        y_poisson_test[index] = np.percentile(a=p_poisson_test, q=x*100)
        y_mep_test[index] = np.percentile(a=p_mep_test, q=x*100)
        y_sep_test[index] = np.percentile(a=p_sep_test, q=x*100)
        y_smep_test[index] = np.percentile(a=p_smep_test, q=x*100)
        y_gbmep_start_test[index] = np.percentile(a=p_gbmep_start_test, q=x*100)
        y_gbmep_test[index] = np.percentile(a=p_gbmep_test, q=x*100)
    # Calculate KS statistic for training set
    k_poisson_train[index] = stats.kstest(p_poisson_train, stats.uniform.cdf)
    k_mep_train[index] = stats.kstest(p_mep_train, stats.uniform.cdf)
    k_sep_train[index] = stats.kstest(p_sep_train, stats.uniform.cdf)
    k_smep_train[index] = stats.kstest(p_smep_train, stats.uniform.cdf)
    k_gbmep_start_train[index] = stats.kstest(p_gbmep_start_train, stats.uniform.cdf)
    k_gbmep_train[index] = stats.kstest(p_gbmep_train, stats.uniform.cdf)
    # Calculate KS statistic for test set
    if len(p_poisson_test) > 0:
        k_poisson_test[index] = stats.kstest(p_poisson_test, stats.uniform.cdf)
        k_mep_test[index] = stats.kstest(p_mep_test, stats.uniform.cdf)
        k_sep_test[index] = stats.kstest(p_sep_test, stats.uniform.cdf)
        k_smep_test[index] = stats.kstest(p_smep_test, stats.uniform.cdf)
        k_gbmep_start_test[index] = stats.kstest(p_gbmep_start_test, stats.uniform.cdf)
        k_gbmep_test[index] = stats.kstest(p_gbmep_test, stats.uniform.cdf)

## Save the results
y_train = {}; k_train = {}
y_test = {}; k_test = {}
y_train['poisson'] = y_poisson_train; y_train['mep'] = y_mep_train; y_train['sep'] = y_sep_train
y_train['smep'] = y_smep_train; y_train['gbmep_start'] = y_gbmep_start_train; y_train['gbmep'] = y_gbmep_train
k_train['poisson'] = k_poisson_train; k_train['mep'] = k_mep_train; k_train['sep'] = k_sep_train
k_train['smep'] = k_smep_train; k_train['gbmep_start'] = k_gbmep_start_train; k_train['gbmep'] = k_gbmep_train
y_test['poisson'] = y_poisson_test; y_test['mep'] = y_mep_test; y_test['sep'] = y_sep_test
y_test['smep'] = y_smep_test; y_test['gbmep_start'] = y_gbmep_start_test; y_test['gbmep'] = y_gbmep_test
k_test['poisson'] = k_poisson_test; k_test['mep'] = k_mep_test; k_test['sep'] = k_sep_test
k_test['smep'] = k_smep_test; k_test['gbmep_start'] = k_gbmep_start_test; k_test['gbmep'] = k_gbmep_test

## Check if directory exists
if not os.path.exists('results/res_qq_start'):
   os.makedirs('results/res_qq_start')

## Save as .pkl files
with open('results/res_qq_start/y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)
with open('results/res_qq_start/k_train.pkl', 'wb') as f:
    pickle.dump(k_train, f)
with open('results/res_qq_start/y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)
with open('results/res_qq_start/k_test.pkl', 'wb') as f:
    pickle.dump(k_test, f)