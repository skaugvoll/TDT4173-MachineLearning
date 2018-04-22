# Assignment 5
## TDT 4173 - Machine learning and case based reasoning
Authors: Eirik Baug, Martin Stigen, Sigve Skaugvoll

## Requirements
- Python 3.x
    -  numpy
    - scipy
    - pillow
- chars74k-lite (data set)

### How to install
Pleas use a virtual environment, and install the packages into the virtual environment.
To install either run the script `setup.py` in `src` or use `pip install [package]`.

## Overview
To make sure you have the necessary packages installed, pleas run
`python setup.py` located in the `src` folder.

The assignment consists of two machine learning methods; `knn` and `....`, which is used to predict the a handwritten character
in the modified data set `chars74k-lite`.

### data_generator.py
The file `data_generator.py` uses `pillow` to read in the grayscale images and store them in
a numpy array. The data is `flat`.
There are methods for shuffling the data, getting cases and there labels in seperated lists
or as `[case-features, label]`.

###  k_nearest_neighbor.py
_**This script is self-implemented.**_

One of the methods tried for classifying is knn. There are two ways this method can be envoked,
one is one-thread and the other is multi-threaded.
This script is reliant on the `k_nn_thread.py` file and the `data_generator.py` script.


## Feature selection
The feature selection implemented are `feature scaling` which ensures that the features are
mapped from [0,255] to [0,1].