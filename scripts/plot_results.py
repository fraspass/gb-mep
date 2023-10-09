#!/usr/bin/env python3
import numpy as np
import pickle

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

## Load .pkl files
with open('results/res_qq_start/y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)

with open('results/res_qq_start/ks_train.pkl', 'rb') as f:
    ks_train = pickle.load(f)

with open('results/res_qq_start/cvm_train.pkl', 'rb') as f:
    cvm_train = pickle.load(f)

with open('results/res_qq_start/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

with open('results/res_qq_start/ks_test.pkl', 'rb') as f:
    ks_test = pickle.load(f)

with open('results/res_qq_start/cvm_test.pkl', 'rb') as f:
    cvm_test = pickle.load(f)

# Import plotting libraries
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Lato']})
rc('text', usetex=True)

freq = []; ks1 = []; ks2 = []
for node in ks_train['smep']:
    freq += [np.sum(santander_distances < .5, axis=1)[node]]
    ks1 += [cvm_test['smep'][node]]
    ks2 += [cvm_test['gbmep'][node]]

# Define theoretical quantiles
x = np.linspace(start=0, stop=1, num=501, endpoint=False)[1:]

# Define empty lists
y_train_tot = {}; y_test_tot = {}
ks_train_tot = {}; ks_test_tot = {}
cvm_train_tot = {}; cvm_test_tot = {}
for model in y_train:
    y_train_tot[model] = []; ks_train_tot[model] = []; cvm_train_tot[model] = []
    y_test_tot[model] = []; ks_test_tot[model] = []; cvm_test_tot[model] = []

# Initialise the subplot function using number of rows and columns 
fig, ax = plt.subplots(2,3, figsize=(12,8))

for u in [0,1]:
    for v in [1,2]:
        ax[u,v].set_yticks([])

for v in range(3):
    ax[0,v].set_xticks([])

for node in y_train['poisson']:
    for model in y_train:
        y_train_tot[model] += list(y_train[model][node])
        ks_train_tot[model] += [ks_train[model][node].statistic]
        cvm_train_tot[model] += [cvm_train[model][node]]
    ax[0,0].plot(x, y_train['poisson'][node], c='lightgray', linewidth=.5) 
    ax[0,1].plot(x, y_train['mep'][node], c='lightgray', linewidth=.5)
    ax[0,2].plot(x, y_train['sep'][node], c='lightgray', linewidth=.5) 
    ax[1,0].plot(x, y_train['smep'][node], c='lightgray', linewidth=.5)
    ax[1,1].plot(x, y_train['gbmep_start'][node], c='lightgray', linewidth=.5)
    ax[1,2].plot(x, y_train['gbmep'][node], c='lightgray', linewidth=.5)

for u in range(2):
    for v in range(3):
        ax[u,v].plot(x,x, c='black', ls='dotted', linewidth=1)

ax[0,0].plot(x, np.percentile(a=y_train_tot['poisson'], q=x*100), linewidth=1, c='black')
ax[0,1].plot(x, np.percentile(a=y_train_tot['mep'], q=x*100), linewidth=1, c='black')
ax[0,2].plot(x, np.percentile(a=y_train_tot['sep'], q=x*100), linewidth=1, c='black')
ax[1,0].plot(x, np.percentile(a=y_train_tot['smep'], q=x*100), linewidth=1, c='black')
ax[1,1].plot(x, np.percentile(a=y_train_tot['gbmep_start'], q=x*100), linewidth=1, c='black')
ax[1,2].plot(x, np.percentile(a=y_train_tot['gbmep'], q=x*100), linewidth=1, c='black')

for u in [0,1]:
    ax[u,0].set_ylabel('Observed quantiles')

for v in [0,1,2]:
    ax[1,v].set_xlabel('Theoretical quantiles')

ax[0,0].set_title('Poisson')
ax[0,1].set_title('MEP')
ax[0,2].set_title('SEP')
ax[1,0].set_title('SMEP')
ax[1,1].set_title('GB-MEP (start times only)')
ax[1,2].set_title('GB-MEP')

# Combine all the operations and display 
plt.show() 

# Combine the data into a list for plotting
ks_train = [ks_train_tot['poisson'], ks_train_tot['mep'], ks_train_tot['sep'], ks_train_tot['gbmep_start'], ks_train_tot['smep'], ks_train_tot['gbmep']]
# Create labels for the boxplots
labels = ['Poisson', 'MEP', 'SEP', 'GB-MEP (start times only)', 'SMEP', 'GB-MEP']
# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
# Create side-by-side boxplots
ax.boxplot(ks_train, labels=labels)
# Set axis labels and a title
ax.set_xlabel('Groups')
ax.set_ylabel('Values')
ax.set_title('Side-by-Side Boxplots')
# Show the plot
plt.show()

# Combine the data into a list for plotting
cvm_train = [cvm_train_tot['poisson'], cvm_train_tot['mep'], cvm_train_tot['sep'], cvm_train_tot['gbmep_start'], cvm_train_tot['smep'], cvm_train_tot['gbmep']]
# Create labels for the boxplots
labels = ['Poisson', 'MEP', 'SEP', 'GB-MEP (start times only)', 'SMEP', 'GB-MEP']
# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
# Create side-by-side boxplots
ax.boxplot(cvm_train, labels=labels)
# Set axis labels and a title
ax.set_xlabel('Groups')
ax.set_ylabel('Values')
ax.set_title('Side-by-Side Boxplots')
# Show the plot
plt.show()






# Initialise the subplot function using number of rows and columns 
fig, ax = plt.subplots(2,3, figsize=(12,8))

for u in [0,1]:
    for v in [1,2]:
        ax[u,v].set_yticks([])

for v in range(3):
    ax[0,v].set_xticks([])

for node in y_test['poisson']:
    for model in y_test:
        y_test_tot[model] += list(y_test[model][node])
        ks_test_tot[model] += [ks_test[model][node].statistic]
        cvm_test_tot[model] += [cvm_test[model][node]]
    ax[0,0].plot(x, y_test['poisson'][node], c='lightgray', linewidth=.5) 
    ax[0,1].plot(x, y_test['mep'][node], c='lightgray', linewidth=.5)
    ax[0,2].plot(x, y_test['sep'][node], c='lightgray', linewidth=.5) 
    ax[1,0].plot(x, y_test['smep'][node], c='lightgray', linewidth=.5)
    ax[1,1].plot(x, y_test['gbmep_start'][node], c='lightgray', linewidth=.5)
    ax[1,2].plot(x, y_test['gbmep'][node], c='lightgray', linewidth=.5)

for u in range(2):
    for v in range(3):
        ax[u,v].plot(x,x, c='black', ls='dotted', linewidth=1)

ax[0,0].plot(x, np.percentile(a=y_test_tot['poisson'], q=x*100), linewidth=1, c='black')
ax[0,1].plot(x, np.percentile(a=y_test_tot['mep'], q=x*100), linewidth=1, c='black')
ax[0,2].plot(x, np.percentile(a=y_test_tot['sep'], q=x*100), linewidth=1, c='black')
ax[1,0].plot(x, np.percentile(a=y_test_tot['smep'], q=x*100), linewidth=1, c='black')
ax[1,1].plot(x, np.percentile(a=y_test_tot['gbmep_start'], q=x*100), linewidth=1, c='black')
ax[1,2].plot(x, np.percentile(a=y_test_tot['gbmep'], q=x*100), linewidth=1, c='black')

for u in [0,1]:
    ax[u,0].set_ylabel('Observed quantiles')

for v in [0,1,2]:
    ax[1,v].set_xlabel('Theoretical quantiles')

ax[0,0].set_title('Poisson')
ax[0,1].set_title('MEP')
ax[0,2].set_title('SEP')
ax[1,0].set_title('SMEP')
ax[1,1].set_title('GB-MEP (start times only)')
ax[1,2].set_title('GB-MEP')

# Combine all the operations and display 
plt.show() 

# Combine the data into a list for plotting
ks_test = [ks_test_tot['poisson'], ks_test_tot['mep'], ks_test_tot['sep'], ks_test_tot['gbmep_start'], ks_test_tot['smep'], ks_test_tot['gbmep']]
# Create labels for the boxplots
labels = ['Poisson', 'MEP', 'SEP', 'GB-MEP (start times only)', 'SMEP', 'GB-MEP']
# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
# Create side-by-side boxplots
ax.boxplot(ks_test, labels=labels)
# Set axis labels and a title
ax.set_xlabel('Groups')
ax.set_ylabel('Values')
ax.set_title('Side-by-Side Boxplots')
# Show the plot
plt.show()

# Combine the data into a list for plotting
cvm_test = [cvm_test_tot['poisson'], cvm_test_tot['mep'], cvm_test_tot['sep'], cvm_test_tot['gbmep_start'], cvm_test_tot['smep'], cvm_test_tot['gbmep']]
# Create labels for the boxplots
labels = ['Poisson', 'MEP', 'SEP', 'GB-MEP (start times only)', 'SMEP', 'GB-MEP']
# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
# Create side-by-side boxplots
ax.boxplot(cvm_test, labels=labels)
# Set axis labels and a title
ax.set_xlabel('Groups')
ax.set_ylabel('Values')
ax.set_title('Side-by-Side Boxplots')
# Show the plot
plt.show()