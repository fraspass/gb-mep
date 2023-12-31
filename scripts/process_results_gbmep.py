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

## Load results for the full GB-MEP model
res_gbmep_full = {}
for file in glob.glob('results/res_gbmep_full/*.pkl'):
    with open(file, 'rb') as f:
        res_gbmep_full.update(pickle.load(f))

## Obtain gb_mep object and expand DataFrame with the test set
G = gb_mep.gb_mep(df=santander_train, id_map=santander_dictionary, distance_matrix=santander_distances)
start_times, end_times = G.augment_start_times(test_set=santander_test)

## Initialise dictionaries for observed quantiles
y_poisson_train = {}; y_mep_train = {}; y_sep_train = {}; y_smep_train = {}; y_gbmep_start_train = {}; y_gbmep_train = {}; y_gbmep_full_train = {}
y_poisson_test = {}; y_mep_test = {}; y_sep_test = {}; y_smep_test = {}; y_gbmep_start_test = {}; y_gbmep_test = {}; y_gbmep_full_test = {}

## Intialise dictionaries for KS scores
ks_poisson_train = {}; ks_mep_train = {}; ks_sep_train = {}; ks_smep_train = {}; ks_gbmep_start_train = {}; ks_gbmep_train = {}; ks_gbmep_full_train = {}
ks_poisson_test = {}; ks_mep_test = {}; ks_sep_test = {}; ks_smep_test = {}; ks_gbmep_start_test = {}; ks_gbmep_test = {}; ks_gbmep_full_test = {}

## Intialise dictionaries for Cramér-von Mises scores
cvm_poisson_train = {}; cvm_mep_train = {}; cvm_sep_train = {}; cvm_smep_train = {}; cvm_gbmep_start_train = {}; cvm_gbmep_train = {}; cvm_gbmep_full_train = {}
cvm_poisson_test = {}; cvm_mep_test = {}; cvm_sep_test = {}; cvm_smep_test = {}; cvm_gbmep_start_test = {}; cvm_gbmep_test = {}; cvm_gbmep_full_test = {}

## Define theoretical quantiles
x = np.linspace(start=0, stop=1, num=501, endpoint=False)[1:]

## Loop over all nodes
p_poisson_train = {}; p_mep_train = {}; p_sep_train = {}; p_smep_train = {}; p_gbmep_start_train = {}; p_gbmep_train = {}; p_gbmep_full_train = {}
p_poisson_test = {}; p_mep_test = {}; p_sep_test = {}; p_smep_test = {}; p_gbmep_start_test = {}; p_gbmep_test = {}; p_gbmep_full_test = {}
for index in G.nodes:
    # Print index and station name
    print('\r', index, '-', G.id_map[index], ' '*20, end='\r')
    # p-values for Poisson process
    p1, p2 = G.pvals_poisson_process(param=res_pp[index], node_index=index, start_times=start_times, test_split=True, validation_split=False)
    p_poisson_train[index], p_poisson_test[index] = p1, p2
    # p-values for MEP
    pp = gb_mep.transform_parameters(res_mep[index].x)
    p1, p2 = G.pvals_mep(params=pp, node_index=index, start_times=start_times, end_times=end_times, test_split=True, validation_split=False)
    p_mep_train[index], p_mep_test[index] = p1, p2
    # p-values for SEP
    pp = gb_mep.transform_parameters(res_sep[index].x)
    p1, p2 = G.pvals_sep(params=pp, node_index=index, start_times=start_times, test_split=True, validation_split=False)
    p_sep_train[index], p_sep_test[index] = p1, p2
    # p-values for SMEP
    pp = gb_mep.transform_parameters(res_smep[index].x)
    p1, p2 = G.pvals_smep(params=pp, node_index=index, start_times=start_times, end_times=end_times, test_split=True, validation_split=False)
    p_smep_train[index], p_smep_test[index] = p1, p2
    # p-values for GB-MEP with start times only
    pp  = gb_mep.transform_parameters(res_gbmep_start[index].x)
    p1, p2 = G.pvals_gbmep_start(params=pp, node_index=index, subset_nodes=res_gbmep_start[index].subset_nodes, start_times=start_times, test_split=True, validation_split=False)
    p_gbmep_start_train[index], p_gbmep_start_test[index] = p1, p2
    # p-values for GB-MEP
    pp = gb_mep.transform_parameters(res_gbmep[index].x)
    p1, p2 = G.pvals_gbmep_start_self(params=pp, node_index=index, subset_nodes=res_gbmep[index].subset_nodes, start_times=start_times, end_times=end_times, test_split=True, validation_split=False) 
    p_gbmep_train[index], p_gbmep_test[index] = p1, p2
    # p-values for GB-MEP (full)
    pp = gb_mep.transform_parameters(res_gbmep_full[index].x)
    p1, p2 = G.pvals_gbmep(params=pp, node_index=index, subset_nodes=res_gbmep_full[index].subset_nodes, start_times=start_times, end_times=end_times, test_split=True, validation_split=False)
    p_gbmep_full_train[index], p_gbmep_full_test[index] = p1, p2
    # Calculate observed percentiles for training set
    y_poisson_train[index] = np.percentile(a=p_poisson_train[index], q=x*100)
    y_mep_train[index] = np.percentile(a=p_mep_train[index], q=x*100)
    y_sep_train[index] = np.percentile(a=p_sep_train[index], q=x*100)
    y_smep_train[index] = np.percentile(a=p_smep_train[index], q=x*100)
    y_gbmep_start_train[index] = np.percentile(a=p_gbmep_start_train[index], q=x*100)
    y_gbmep_train[index] = np.percentile(a=p_gbmep_train[index], q=x*100)
    y_gbmep_full_train[index] = np.percentile(a=p_gbmep_full_train[index], q=x*100)
    # Calculate observed percentiles for test set
    if len(p_poisson_test[index]) > 0:
        y_poisson_test[index] = np.percentile(a=p_poisson_test[index], q=x*100)
        y_mep_test[index] = np.percentile(a=p_mep_test[index], q=x*100)
        y_sep_test[index] = np.percentile(a=p_sep_test[index], q=x*100)
        y_smep_test[index] = np.percentile(a=p_smep_test[index], q=x*100)
        y_gbmep_start_test[index] = np.percentile(a=p_gbmep_start_test[index], q=x*100)
        y_gbmep_test[index] = np.percentile(a=p_gbmep_test[index], q=x*100)
        y_gbmep_full_test[index] = np.percentile(a=p_gbmep_full_test[index], q=x*100)
    # Calculate KS statistic for training set
    ks_poisson_train[index] = stats.kstest(p_poisson_train[index], stats.uniform.cdf)
    ks_mep_train[index] = stats.kstest(p_mep_train[index], stats.uniform.cdf)
    ks_sep_train[index] = stats.kstest(p_sep_train[index], stats.uniform.cdf)
    ks_smep_train[index] = stats.kstest(p_smep_train[index], stats.uniform.cdf)
    ks_gbmep_start_train[index] = stats.kstest(p_gbmep_start_train[index], stats.uniform.cdf)
    ks_gbmep_train[index] = stats.kstest(p_gbmep_train[index], stats.uniform.cdf)
    ks_gbmep_full_train[index] = stats.kstest(p_gbmep_full_train[index], stats.uniform.cdf)
    # Calculate KS statistic for test set
    if len(p_poisson_test[index]) > 0:
        ks_poisson_test[index] = stats.kstest(p_poisson_test[index], stats.uniform.cdf)
        ks_mep_test[index] = stats.kstest(p_mep_test[index], stats.uniform.cdf)
        ks_sep_test[index] = stats.kstest(p_sep_test[index], stats.uniform.cdf)
        ks_smep_test[index] = stats.kstest(p_smep_test[index], stats.uniform.cdf)
        ks_gbmep_start_test[index] = stats.kstest(p_gbmep_start_test[index], stats.uniform.cdf)
        ks_gbmep_test[index] = stats.kstest(p_gbmep_test[index], stats.uniform.cdf)
        ks_gbmep_full_test[index] = stats.kstest(p_gbmep_full_test[index], stats.uniform.cdf)
    # Calculate Cramér-von Mises statistic for training set
    cvm_poisson_train[index] = np.sqrt(stats.cramervonmises(rvs=p_poisson_train[index], cdf=stats.uniform.cdf).statistic / G.N[index])
    cvm_mep_train[index] = np.sqrt(stats.cramervonmises(rvs=p_mep_train[index], cdf=stats.uniform.cdf).statistic / G.N[index])
    cvm_sep_train[index] = np.sqrt(stats.cramervonmises(rvs=p_sep_train[index], cdf=stats.uniform.cdf).statistic / G.N[index])
    cvm_smep_train[index] = np.sqrt(stats.cramervonmises(rvs=p_smep_train[index], cdf=stats.uniform.cdf).statistic / G.N[index])
    cvm_gbmep_start_train[index] = np.sqrt(stats.cramervonmises(rvs=p_gbmep_start_train[index], cdf=stats.uniform.cdf).statistic / G.N[index])
    cvm_gbmep_train[index] = np.sqrt(stats.cramervonmises(rvs=p_gbmep_train[index], cdf=stats.uniform.cdf).statistic / G.N[index])
    cvm_gbmep_full_train[index] = np.sqrt(stats.cramervonmises(rvs=p_gbmep_full_train[index], cdf=stats.uniform.cdf).statistic / G.N[index])
    # Calculate Cramér-von Mises for test set
    if len(p_poisson_test[index]) > 0:
        cvm_poisson_test[index] = np.sqrt(stats.cramervonmises(rvs=p_poisson_test[index], cdf=stats.uniform.cdf).statistic / (len(start_times[index])-G.N[index]))
        cvm_mep_test[index] = np.sqrt(stats.cramervonmises(rvs=p_mep_test[index], cdf=stats.uniform.cdf).statistic / (len(start_times[index])-G.N[index]))
        cvm_sep_test[index] = np.sqrt(stats.cramervonmises(rvs=p_sep_test[index], cdf=stats.uniform.cdf).statistic / (len(start_times[index])-G.N[index]))
        cvm_smep_test[index] = np.sqrt(stats.cramervonmises(rvs=p_smep_test[index], cdf=stats.uniform.cdf).statistic / (len(start_times[index])-G.N[index]))
        cvm_gbmep_start_test[index] = np.sqrt(stats.cramervonmises(rvs=p_gbmep_start_test[index], cdf=stats.uniform.cdf).statistic / (len(start_times[index])-G.N[index]))
        cvm_gbmep_test[index] = np.sqrt(stats.cramervonmises(rvs=p_gbmep_test[index], cdf=stats.uniform.cdf).statistic / (len(start_times[index])-G.N[index]))
        cvm_gbmep_full_test[index] = np.sqrt(stats.cramervonmises(rvs=p_gbmep_full_test[index], cdf=stats.uniform.cdf).statistic / (len(start_times[index])-G.N[index]))

## Save the results
p_train = {}; y_train = {}; ks_train = {}; cvm_train = {}
p_test = {}; y_test = {}; ks_test = {}; cvm_test = {}
# Training
p_train['poisson'] = p_poisson_train; p_train['mep'] = p_mep_train; p_train['sep'] = p_sep_train; p_train['smep'] = p_smep_train
p_train['gbmep_start'] = p_gbmep_start_train; p_train['gbmep'] = p_gbmep_train; p_train['gbmep_full'] = p_gbmep_full_train
y_train['poisson'] = y_poisson_train; y_train['mep'] = y_mep_train; y_train['sep'] = y_sep_train; y_train['smep'] = y_smep_train
y_train['gbmep_start'] = y_gbmep_start_train; y_train['gbmep'] = y_gbmep_train; y_train['gbmep_full'] = y_gbmep_full_train
ks_train['poisson'] = ks_poisson_train; ks_train['mep'] = ks_mep_train; ks_train['sep'] = ks_sep_train; ks_train['smep'] = ks_smep_train
ks_train['gbmep_start'] = ks_gbmep_start_train; ks_train['gbmep'] = ks_gbmep_train; ks_train['gbmep_full'] = ks_gbmep_full_train
cvm_train['poisson'] = cvm_poisson_train; cvm_train['mep'] = cvm_mep_train; cvm_train['sep'] = cvm_sep_train; cvm_train['smep'] = cvm_smep_train
cvm_train['gbmep_start'] = cvm_gbmep_start_train; cvm_train['gbmep'] = cvm_gbmep_train; cvm_train['gbmep_full'] = cvm_gbmep_full_train
# Test
p_test['poisson'] = p_poisson_test; p_test['mep'] = p_mep_test; p_test['sep'] = p_sep_test; p_test['smep'] = p_smep_test
p_test['gbmep_start'] = p_gbmep_start_test; p_test['gbmep'] = p_gbmep_test; p_test['gbmep_full'] = p_gbmep_full_test
y_test['poisson'] = y_poisson_test; y_test['mep'] = y_mep_test; y_test['sep'] = y_sep_test; y_test['smep'] = y_smep_test
y_test['gbmep_start'] = y_gbmep_start_test; y_test['gbmep'] = y_gbmep_test; y_test['gbmep_full'] = y_gbmep_full_test
ks_test['poisson'] = ks_poisson_test; ks_test['mep'] = ks_mep_test; ks_test['sep'] = ks_sep_test; ks_test['smep'] = ks_smep_test
ks_test['gbmep_start'] = ks_gbmep_start_test; ks_test['gbmep'] = ks_gbmep_test; ks_test['gbmep_full'] = ks_gbmep_full_test
cvm_test['poisson'] = cvm_poisson_test; cvm_test['mep'] = cvm_mep_test; cvm_test['sep'] = cvm_sep_test; cvm_test['smep'] = cvm_smep_test
cvm_test['gbmep_start'] = cvm_gbmep_start_test; cvm_test['gbmep'] = cvm_gbmep_test; cvm_test['gbmep_full'] = cvm_gbmep_full_test

## Check if directory exists
if not os.path.exists('results/res_qq_start'):
   os.makedirs('results/res_qq_start')

## Save as .pkl files
with open('results/res_qq_start/pv_train.pkl', 'wb') as f:
    pickle.dump(p_train, f)
with open('results/res_qq_start/y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)
with open('results/res_qq_start/ks_train.pkl', 'wb') as f:
    pickle.dump(ks_train, f)
with open('results/res_qq_start/cvm_train.pkl', 'wb') as f:
    pickle.dump(cvm_train, f)
with open('results/res_qq_start/pv_test.pkl', 'wb') as f:
    pickle.dump(p_test, f)
with open('results/res_qq_start/y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)
with open('results/res_qq_start/ks_test.pkl', 'wb') as f:
    pickle.dump(ks_test, f)
with open('results/res_qq_start/cvm_test.pkl', 'wb') as f:
    pickle.dump(cvm_test, f)