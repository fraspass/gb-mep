# GB-MEP: Graph-based mutually exciting point processes

This repository contains the Python library `gb_mep`, which supports the paper *"Graph-based mutually exciting point processes for modelling event times in docked bike-sharing systems"* by Francesco Sanna Passino, Yining Che, and Carlos Cardoso Correia Perello. The repository contains the following directories:

* `lib` contains the _Python_ library, with code to implement and fit GB-MEPs to graphs of point processes.
* `notebooks` contains Jupyter notebooks with examples on how to use the `gb_mep` library.
* `data` contains scripts to download the Santander Cycles data used in the paper. Raw data files are *not* managed under version control, so you should follow the instructions below to obtain the preprocessed version of the data.
* `scripts` contains scripts to reproduce the parameter estimates obtained in the paper.
* `results` contains `.pkl` files containing the parameter estimates obtained from the data used in the paper. 

The _Python_ library `gb_mep` can be installed in edit mode as follows:
```
pip install -e lib/
```
Alternatively, `pip3` can also be used. After installation, the library can then be imported in any _Python_ session as follows:
```python3
import gb_mep
```
A demo on how to use the library with the Santander Cycles data can be found in `notebooks/Santander_Cycles.ipynb` and `notebooks/Santander_Cycles_2020.ipynb`.

## Santander Cycles data: downloading and preprocessing

The Santander Cycles data can be downloaded from the [TfL Cycling Data Repository](https://cycling.data.tfl.gov.uk/) (see the [terms of service](https://tfl.gov.uk/corporate/terms-and-conditions/transport-data-service)) by navigating to the directory `data` in this repository, and running the script `get_data.sh` as detailed below. The files will be stored in `.csv` files in two directories: `data/training` and `data/test`. 
```
cd data
bash get_data.sh
```
After the data has been downloaded, it can be preprocessed via the script `data_processing.py` in the directory `data`, as follows:
```
cd data
python3 data_processing.py
```
If the library `gb_mep` has been installed as described above, all libraries required for the preprocessing should already be installed. The output of the script consists in four files, stored in `data`:
- `santander_training.csv`, containing a preprocessed DataFrame with the training set data;
- `santander_test.csv`, containing a preprocessed DataFrame with the test set data;
- `santander_dictionary.pkl`, containing a dictionary with the mapping of integers to station names and viceversa (useful to match numbers in the previous files to station names);
- `santander_locations.npy`, containing a Numpy array containing distances between station, with the same encoding of rows and columns found in the `santander_dictionary.pkl` file. 

A previous version of the data (not used in the paper) can be obtained and preprocessed using the scripts labelled with `2020` in the directory `data`:
```
cd data
bash get_data.sh
python3 data_processing.py
```
The output of the script consists in files stored in a directory `santander_summaries` and its subdirectory `santander_summaries_preprocessed`. 

## Reproducing the results in the paper

To reproduce the results and plots in the paper, the *Bash* and *Python* scripts in `scripts` could be run. Note that these operations are computationally expensive, so it is recommended to run those on a remote server, not on a personal laptop. First, it is necessary to run the scripts `run_benchmarks.py`, which calculates parameter estimates for the Poisson, SEP, MEP and SMEP models, which are subsequently used to initialise SpMEP and GB-MEP to aid convergence:
```
python3 scripts/run_benchmarks.py &
``` 
Next, the *Bash* scripts `scripts/run_gbmep_start.sh`, `scripts/run_gbmep.sh` and `scripts/run_gbmep_full.sh` should be run to obtain the results of SpMEP, SpSMEP (a variation of SpMEP which includes the end times of events at the same station - this is not reported in the paper since it does not improve the results of SMEP, and it has worse performance than GB-MEP), and GB-MEP:
```
bash scripts/run_gbmep_start.sh &
bash scripts/run_gbmep.sh &
bash scripts/run_gbmep_full.sh &
``` 
Note that the above scripts launch 16 processes each (where each process fits the corresponding model on approximately 50 stations). Therefore, it is highly recommended to run these scripts one-by-one on a remote server.

After all scripts have been run, the results can be postprocessed with the following command:
```
python3 scripts/process_results_gbmep.py &
``` 
This script creates combined results files in `results/res_qq_start`, which can then be used for producing the plots reported in the paper.

Most plots and results can be obtained via the following command, after the files in `results/res_qq_start` are available: 
```
python3 scripts/plot_results.py &
``` 
Additional plots can be obtained from the Notebook `plots/Santander_Cycles_Plots.ipynb`.