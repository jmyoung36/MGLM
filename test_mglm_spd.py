#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:29:58 2019

@author: jonyoung
"""

import numpy as np
from sklearn.datasets import make_spd_matrix
from riem_mglm import mglm_spd
import pandas as pd

# set data directory
# just use project directory as this is small test data only
data_dir = '/home/jonyoung/IoP_data/Projects/riem_mglm/'
#data_dir = '/home/k1511004/Projects/riem_mglm/'

# create SPD matrix data: 20 8x8 spd matrices
N = 20
mat_size = 8
dimX = 2
#Y = np.zeros((mat_size, mat_size, N))
#for i in range(N) :
#    
#    Y[:, :, i] = make_spd_matrix(mat_size)
#    
## create column vectors, dimX by n_m
## fill with random data
#X = np.random.rand(dimX, N) * 10

# squash the matrices to vectors and save both X & Y as .csv so they can be used by matlab
#Y = np.reshape(Y, (mat_size * mat_size, N))
#np.savetxt(data_dir + 'Y_test.csv', Y, delimiter=',')
#np.savetxt(data_dir + 'X_test.csv', X, delimiter=',')

# read in the X and Y
Y = np.genfromtxt(data_dir + 'Y_test.csv', delimiter=',')
Y = np.reshape(Y, (mat_size, mat_size, N))
X = np.genfromtxt(data_dir + 'X_test.csv', delimiter=',')

# try to run MGLM...
#p, V, E, Y_hat, gnorm = mglm_spd(X, Y, 20)

# real data
# directories for real data
covariance_data_dir = '/home/jonyoung/IoP_data/Data/PSYSCAN/Legacy_data/Dublin/'
covariance_metadata_dir = '/home/jonyoung/IoP_data/Data/PSYSCAN/Legacy_data/3pt3_definitive/'
#covariance_data_dir = '/home/k1511004/Data/PSYSCAN/Legacy_data/Dublin/'
#covariance_metadata_dir = '/home/k1511004/Data/PSYSCAN/Legacy_data/3pt3_definitive/'


# try using real data
# set data file containing predictors
data_file = 'connectivity_data_scaled.csv'

# do we want to normalise demographics?
normalise_demographics = True;

# read in some data
connectivity_data = np.genfromtxt(covariance_data_dir + data_file, delimiter=',')

# split into data and labels
labels = connectivity_data[:, 0]
connectivity_data = connectivity_data[:, 1:]

# take get characteristics
n_connections = np.shape(connectivity_data)[1]
n_regions = int(np.sqrt(n_connections))
tril_inds = np.tril_indices(n_regions, k=-1)

# read in subjects we have connectivity for
connectivity_subjects_data = pd.read_csv(covariance_metadata_dir + 'Dublin_subjects_111.txt', header=None)

# join labels and data to subjects
connectivity_subjects_data.columns = ['subject']
connectivity_subjects_data['labels'] = pd.Series(labels)
connectivity_subjects_data = pd.concat([connectivity_subjects_data, pd.DataFrame(connectivity_data)], axis=1)

# read in definitive list of subjects
definitive_subject_metadata = pd.read_excel(covariance_metadata_dir + 'Dublin_subjects_include_schizoaffective.xlsx', header=None)

# join metadata and data
connectivity_metadata_data = pd.merge(definitive_subject_metadata, connectivity_subjects_data, left_on=0, right_on='subject')

# get number of subjects
n_subjects = len(connectivity_metadata_data)

# pull out connectivity data
connectivity_data = connectivity_metadata_data.iloc[:, 8:].as_matrix()

# get demographics
demographics = connectivity_metadata_data.iloc[:, 2:4].as_matrix().astype(float)
demographics = np.flip(demographics, 1)

# normalise predictors - optional
if normalise_demographics :
    
    
    demographics = demographics - np.min(demographics, axis=0)
    demographics = demographics / np.max(demographics, axis=0)
    
# transpose and reshape for mglm
demographics = np.transpose(demographics)
connectivity_data = np.transpose(connectivity_data)
connectivity_data = np.reshape(connectivity_data, (290, 290, 108))

# try to run MGLM...
p, V, E, Y_hat, gnorm = mglm_spd(demographics, connectivity_data, 10)