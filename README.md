# GB-MEP: Graph-based mutually exciting point processes

Template for a Python library

This repository contains a template for a Python library.

* `Notebooks` contains Jupyter notebooks with exporatory analyses and experiments,
* `lib` contains a _Python_ library with reusable bits of code (e.g. helper functions),
* `data` can be used to store data. It is *not* managed under version control. 

The _Python_ library can be installed in edit mode as follows:
```
pip install -e lib/
```
## Santander Cycles data: downloading and preprocessing
The Santander Cycles data could be downloaded from the [https://cycling.data.tfl.gov.uk/](TfL Cycling repository) (see the [https://tfl.gov.uk/corporate/terms-and-conditions/transport-data-service](terms of service)) by navigating to the directory `data`, and running the script `get_data.sh`. The files will be stored in `.csv` files in two directories: `data/training` and `data/test`. 
```
cd data
bash get_data.sh
```
After the data has been downloaded, it can be preprocessed via the script `data_process.py` in the directory `data`, as follows:
```
cd data
python3 data_process.py
```
If the library `gb_mep` has been installed as described above, all libraries required for the preprocessing should already be installed. The output of the script consists in four files, stored in `data`:
- `santander_training.csv`, containing a preprocessed DataFrame with the training set data;
- `santander_test.csv`, containing a preprocessed DataFrame with the test set data;
- `santander_dictionary.pkl`, containing a dictionary with the mapping of integers to station names and viceversa (useful to match numbers in the previous files to station names);
- `santander_locations.npy`, containing a Numpy array containing distances between station, with the same encoding of rows and columns found in the `santander_dictionary.pkl` file. 
