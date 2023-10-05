#!/usr/bin/env python3
import numpy as np
import pickle

## Load .pkl files
with open('results/res_qq_start/y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)

with open('results/res_qq_start/k_train.pkl', 'rb') as f:
    k_train = pickle.load(f)

with open('results/res_qq_start/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

with open('results/res_qq_start/k_test.pkl', 'rb') as f:
    k_test = pickle.load(f)

# Import plotting libraries
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Lato']})
rc('text', usetex=True)

# Define theoretical quantiles
x = np.linspace(start=0, stop=1, num=501, endpoint=False)[1:]

# Define empty lists
y_train_tot = {}; y_test_tot = {}
k_train_tot = {}; k_test_tot = {}
for model in y_train:
    y_train_tot[model] = []; k_train_tot[model] = []
    y_test_tot[model] = []; k_test_tot[model] = []

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
        k_train_tot[model] += [k_train[model][node].statistic]
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
ks_train = [k_train_tot['poisson'], k_train_tot['mep'], k_train_tot['sep'], k_train_tot['gbmep_start'], k_train_tot['smep'], k_train_tot['gbmep']]
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
        k_test_tot[model] += [k_test[model][node].statistic]
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
ks_test = [k_test_tot['poisson'], k_test_tot['mep'], k_test_tot['sep'], k_test_tot['gbmep_start'], k_test_tot['smep'], k_test_tot['gbmep']]
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