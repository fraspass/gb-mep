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
ks_poisson_train = {}; ks_mep_train = {}; ks_sep_train = {}; ks_smep_train = {}; ks_gbmep_start_train = {}; ks_gbmep_train = {}
ks_poisson_test = {}; ks_mep_test = {}; ks_sep_test = {}; ks_smep_test = {}; ks_gbmep_start_test = {}; ks_gbmep_test = {}

## Intialise dictionaries for Cramér-von Mises scores
cvm_poisson_train = {}; cvm_mep_train = {}; cvm_sep_train = {}; cvm_smep_train = {}; cvm_gbmep_start_train = {}; cvm_gbmep_train = {}
cvm_poisson_test = {}; cvm_mep_test = {}; cvm_sep_test = {}; cvm_smep_test = {}; cvm_gbmep_start_test = {}; cvm_gbmep_test = {}

## Define theoretical quantiles
x = np.linspace(start=0, stop=1, num=501, endpoint=False)[1:]

## Loop over all nodes
for index in G.nodes:
    # Print index and station name
    print('\r', index, '-', G.id_map[index], ' '*20, end='\r')
    # p-values for Poisson process
    p_poisson_train, p_poisson_test = G.pvals_poisson_process(param=res_pp[index], node_index=index, start_times=start_times, test_split=True)
    # p-values for MEP
    pp = gb_mep.transform_parameters(res_mep[index].x)
    p_mep_train, p_mep_test = G.pvals_mep(params=pp, node_index=index, start_times=start_times, end_times=end_times, test_split=True)
    # p-values for SEP
    pp = gb_mep.transform_parameters(res_sep[index].x)
    p_sep_train, p_sep_test = G.pvals_sep(params=pp, node_index=index, start_times=start_times, test_split=True)
    # p-values for SMEP
    pp = gb_mep.transform_parameters(res_smep[index].x)
    p_smep_train, p_smep_test = G.pvals_smep(params=pp, node_index=index, start_times=start_times, end_times=end_times, test_split=True)
    # p-values for GB-MEP with start times only
    pp  = gb_mep.transform_parameters(res_gbmep_start[index].x)
    p_gbmep_start_train, p_gbmep_start_test = G.pvals_gbmep_start(params=pp, node_index=index, subset_nodes=res_gbmep_start[index].subset_nodes, start_times=start_times, test_split=True)
    # p-values for GB-MEP
    pp = gb_mep.transform_parameters(res_gbmep[index].x)
    p_gbmep_train, p_gbmep_test = G.pvals_gbmep_start_self(params=pp, node_index=index, subset_nodes=res_gbmep[index].subset_nodes, start_times=start_times, end_times=end_times, test_split=True) 
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
    ks_poisson_train[index] = stats.kstest(p_poisson_train, stats.uniform.cdf)
    ks_mep_train[index] = stats.kstest(p_mep_train, stats.uniform.cdf)
    ks_sep_train[index] = stats.kstest(p_sep_train, stats.uniform.cdf)
    ks_smep_train[index] = stats.kstest(p_smep_train, stats.uniform.cdf)
    ks_gbmep_start_train[index] = stats.kstest(p_gbmep_start_train, stats.uniform.cdf)
    ks_gbmep_train[index] = stats.kstest(p_gbmep_train, stats.uniform.cdf)
    # Calculate KS statistic for test set
    if len(p_poisson_test) > 0:
        ks_poisson_test[index] = stats.kstest(p_poisson_test, stats.uniform.cdf)
        ks_mep_test[index] = stats.kstest(p_mep_test, stats.uniform.cdf)
        ks_sep_test[index] = stats.kstest(p_sep_test, stats.uniform.cdf)
        ks_smep_test[index] = stats.kstest(p_smep_test, stats.uniform.cdf)
        ks_gbmep_start_test[index] = stats.kstest(p_gbmep_start_test, stats.uniform.cdf)
        ks_gbmep_test[index] = stats.kstest(p_gbmep_test, stats.uniform.cdf)
    # Calculate Cramér-von Mises statistic for training set
    cvm_poisson_train[index] = np.sqrt(stats.cramervonmises(rvs=p_poisson_train, cdf=stats.uniform.cdf).statistic / G.N[index])
    cvm_mep_train[index] = np.sqrt(stats.cramervonmises(rvs=p_mep_train, cdf=stats.uniform.cdf).statistic / G.N[index])
    cvm_sep_train[index] = np.sqrt(stats.cramervonmises(rvs=p_sep_train, cdf=stats.uniform.cdf).statistic / G.N[index])
    cvm_smep_train[index] = np.sqrt(stats.cramervonmises(rvs=p_smep_train, cdf=stats.uniform.cdf).statistic / G.N[index])
    cvm_gbmep_start_train[index] = np.sqrt(stats.cramervonmises(rvs=p_gbmep_start_train, cdf=stats.uniform.cdf).statistic / G.N[index])
    cvm_gbmep_train[index] = np.sqrt(stats.cramervonmises(rvs=p_gbmep_train, cdf=stats.uniform.cdf).statistic / G.N[index])
    # Calculate Cramér-von Mises for test set
    if len(p_poisson_test) > 0:
        cvm_poisson_test[index] = np.sqrt(stats.cramervonmises(rvs=p_poisson_test, cdf=stats.uniform.cdf).statistic / G.N[index])
        cvm_mep_test[index] = np.sqrt(stats.cramervonmises(rvs=p_mep_test, cdf=stats.uniform.cdf).statistic / G.N[index])
        cvm_sep_test[index] = np.sqrt(stats.cramervonmises(rvs=p_sep_test, cdf=stats.uniform.cdf).statistic / G.N[index])
        cvm_smep_test[index] = np.sqrt(stats.cramervonmises(rvs=p_smep_test, cdf=stats.uniform.cdf).statistic / G.N[index])
        cvm_gbmep_start_test[index] = np.sqrt(stats.cramervonmises(rvs=p_gbmep_start_test, cdf=stats.uniform.cdf).statistic / G.N[index])
        cvm_gbmep_test[index] = np.sqrt(stats.cramervonmises(rvs=p_gbmep_test, cdf=stats.uniform.cdf).statistic / G.N[index])

## Save the results
y_train = {}; ks_train = {}; cvm_train = {}
y_test = {}; ks_test = {}; cvm_test = {}
y_train['poisson'] = y_poisson_train; y_train['mep'] = y_mep_train; y_train['sep'] = y_sep_train
y_train['smep'] = y_smep_train; y_train['gbmep_start'] = y_gbmep_start_train; y_train['gbmep'] = y_gbmep_train
ks_train['poisson'] = ks_poisson_train; ks_train['mep'] = ks_mep_train; ks_train['sep'] = ks_sep_train
ks_train['smep'] = ks_smep_train; ks_train['gbmep_start'] = ks_gbmep_start_train; ks_train['gbmep'] = ks_gbmep_train
cvm_train['poisson'] = cvm_poisson_train; cvm_train['mep'] = cvm_mep_train; cvm_train['sep'] = cvm_sep_train
cvm_train['smep'] = cvm_smep_train; cvm_train['gbmep_start'] = cvm_gbmep_start_train; cvm_train['gbmep'] = cvm_gbmep_train
y_test['poisson'] = y_poisson_test; y_test['mep'] = y_mep_test; y_test['sep'] = y_sep_test
y_test['smep'] = y_smep_test; y_test['gbmep_start'] = y_gbmep_start_test; y_test['gbmep'] = y_gbmep_test
ks_test['poisson'] = ks_poisson_test; ks_test['mep'] = ks_mep_test; ks_test['sep'] = ks_sep_test
ks_test['smep'] = ks_smep_test; ks_test['gbmep_start'] = ks_gbmep_start_test; ks_test['gbmep'] = ks_gbmep_test
cvm_test['poisson'] = cvm_poisson_test; cvm_test['mep'] = cvm_mep_test; cvm_test['sep'] = cvm_sep_test
cvm_test['smep'] = cvm_smep_test; cvm_test['gbmep_start'] = cvm_gbmep_start_test; cvm_test['gbmep'] = cvm_gbmep_test

## Check if directory exists
if not os.path.exists('results/res_qq_start'):
   os.makedirs('results/res_qq_start')

## Save as .pkl files
with open('results/res_qq_start/y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)
with open('results/res_qq_start/ks_train.pkl', 'wb') as f:
    pickle.dump(ks_train, f)
with open('results/res_qq_start/cvm_train.pkl', 'wb') as f:
    pickle.dump(cvm_train, f)
with open('results/res_qq_start/y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)
with open('results/res_qq_start/ks_test.pkl', 'wb') as f:
    pickle.dump(ks_test, f)
with open('results/res_qq_start/cvm_test.pkl', 'wb') as f:
    pickle.dump(cvm_test, f)