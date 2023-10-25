# GB-MEP: Graph-based mutually exciting point processes

This repository contains the Python library `gb_mep`, which supports the paper *"Graph-based mutually exciting point processes for modelling event times in docked bike-sharing systems"* by Francesco Sanna Passino, Yining Che, and Carlos Cardoso Correia Perello. The repository contains the following directories:

* `lib` contains the _Python_ library, with code to implement and fit GB-MEPs to graphs of point processes.
* `notebooks` contains Jupyter notebooks with examples on how to use the `gb_mep` library.
* `data` contains scripts to download the Santander Cycles data used in the paper. Row data files are *not* managed under version control, so you should follow the instructions below to obtain the preprocessed version of the data.
* `scripts` contains scripts to reproduce the parameter estimates obtained in the paper.
* `results` contains `.pkl` files containing the parameter estimates obtained from the data used in the paper. 

The _Python_ library can be installed in edit mode as follows:
```
pip install -e lib/
```
Alternatively, `pip3` can also be used. After installation, the library can then be imported in any _Python_ session as follows:
```python3
import gb_mep
```
A demo on how to use the library with the Santander Cycles data can be found in `notebooks/Santander_Cycles.ipynb`.

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
- `santander_validation.csv`, containing a preprocessed DataFrame with the validation set data;
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