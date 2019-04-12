#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Config
dataset = "training_setA"
path = "../" + dataset +"/"
kfold_split = 10
nan_to_neg = False
mm = True
std = False
numpy_save = True

# Script name struct for report
script_name = 'npysave'
dl_ = '_'
name_struct_meta = "_N_scale_mem"

## Imports
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

## Random seed
np.random.seed(seed=0)

## Folder and files
fnames = os.listdir(path)  
fnames.sort()
if 'README.md' in fnames:
    fnames.remove('README.md')
print('last file: ', fnames[-1])

n = len(fnames)
print(n, ' files present')

## Read data
def read_challenge_data(input_file, return_header = False):
    with open(input_file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')

    # ignore SepsisLabel column if present
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        data = data[:, :-1]
    return (data)

def read_challenge_data_label(input_file, return_header = False):
    with open(input_file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')

    # ignore SepsisLabel column if present
    if column_names[-1] == 'SepsisLabel':
        sep_lab = data[:,-1] 
        column_names = column_names[:-1]
        data = data[:, :-1]
    if return_header:
        return (data, sep_lab, column_names)

    else:
        return (data, sep_lab)

## Create the feature matrix
features = []
patient = []
sepsis_label = []

## read data
for i in range(n):
    input_file = os.path.join(path, fnames[i])
    if i ==0:
        data, sep_lab, columns = read_challenge_data_label(input_file, return_header=True)
    else: 
        data, sep_lab = read_challenge_data_label(input_file)
    features.append(data)
    sepsis_label.append(sep_lab)
    pat = i * np.ones((sep_lab.shape), dtype=np.int)
    patient.append(pat)

feature_matrix = np.concatenate(features)
del(features)
sepsis_label = np.concatenate(sepsis_label)
patient = np.concatenate(patient)

## Separate pointers
feature_phys = feature_matrix[:,:-6]    ## Physiology
feature_demog = feature_matrix[:,-6:]   ## Demographics

## Get sepsis patients
patient_sep = np.zeros(len(sepsis_label),dtype=np.int)
for i in range(n):
    i_pat = np.where(patient==i)[0]
    patient_sep[i_pat] = int(np.sum(sepsis_label[i_pat])>0)*np.ones(len(i_pat), dtype=np.int)
    
patient_sep_idx = patient[np.where(patient_sep!=0)]
patient_healthy_idx = patient[np.where(patient_sep==0)]

## Normalize mm(all) or std (sepsis, phys) vals, feature-based
if mm and not std:
    scaler = MinMaxScaler()
    for i in range(n):
        i_pat = np.where(patient==i)[0]
        scaler.fit(feature_matrix[i_pat,:])
        feature_matrix[i_pat,:] = scaler.transform(feature_matrix[i_pat,:])

elif std and not mm:
    scaler = StandardScaler()
    scaler.fit(feature_phys[patient_healthy_idx,:])
    feature_phys[:,:] = scaler.transform(feature_phys[:,:])

## nan to negative
if nan_to_neg:
    feature_matrix[np.isnan(feature_matrix)]=-1
    print("Changed nan to -1")
    
if numpy_save:
    npyfilename = '../npy/' + dataset + '_mm.npy'
    np.save(npyfilename, feature_matrix)
    print(npyfilename, 'saved')
    np.save('../npy/' + dataset + '_patient.npy', patient)
    print('../npy/' + dataset + '_patient.npy', 'saved')
    np.save('../npy/' + dataset + '_Y.npy', sepsis_label)
    print('../npy/' + dataset + '_Y.npy', 'saved')

