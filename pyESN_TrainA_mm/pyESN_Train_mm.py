#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Config
dataset = "training_setA"
path = "../" + dataset +"/"
nan_to_neg = True
biased_regress = True
normal_equations = True
mm = True
std = False
numpy_load = True

## ESN parameters
N_def = 200         # Neurons
scale_def = 1.00      # scaling
mem_def = 0.25      # memory
exponent_def = 0.1    # sigmoid exponent

# Script name struct for report
script_name = 'ESNtrain_mm'
dl_ = '_'
name_struct_meta = "_N_scale_mem"
name_struct = '_{:03d}_{:1.3f}_{:1.3f}'.format(N_def, scale_def, mem_def)

## Imports
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import ESNtools

## Random seed
np.random.seed(seed=0)

## Create the feature matrix
features = []
patient = []
sepsis_label = []
dataloaded = False

## Read data 
if not numpy_load:
    ## Folder and files
    fnames = os.listdir(path)  
    fnames.sort()
    if 'README.md' in fnames:
        fnames.remove('README.md')

    print('last file: ', fnames[-1])
    
    n = len(fnames)
    print(n, ' files present')
    
    ## read data
    for i in range(n):
        input_file = os.path.join(path, fnames[i])
        if i ==0:
            data, sep_lab, columns = ESNtools.read_challenge_data_label(input_file, return_header=True)
        else: 
            data, sep_lab = ESNtools.read_challenge_data_label(input_file)
        features.append(data)
        sepsis_label.append(sep_lab)
        pat = i * np.ones((sep_lab.shape), dtype=np.int)
        patient.append(pat)

    feature_matrix = np.concatenate(features)
    del(features)
    sepsis_label = np.concatenate(sepsis_label)
    patient = np.concatenate(patient)
    dataloaded = True
    
else:
    if mm:
        npyfilename = "../npy/" + dataset + "_mm.npy"
        mm = False
        print(npyfilename, '(mm) to be loaded')

    else:
        npyfilename = "../npy/" + dataset + ".npy"
        print(npyfilename, '(mm) to be loaded')

    
    feature_matrix = np.load(npyfilename)
    npyfilename = "../npy/" + dataset + "_patient.npy"
    patient = np.load(npyfilename)
    npyfilename = "../npy/" + dataset + "_Y.npy"
    sepsis_label = np.load(npyfilename)

    n = len(np.unique(patient))
    print(n, ' files present')
    
    dataloaded = True

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
if mm:
    scaler = MinMaxScaler()
    for i in range(n):
        i_pat = np.where(patient==i)[0]
        scaler.fit(feature_matrix[i_pat,:])
        feature_matrix[i_pat,:] = scaler.transform(feature_matrix[i_pat,:])

elif std:
    scaler = StandardScaler()
    scaler.fit(feature_phys[patient_healthy_idx,:])
    feature_phys[:,:] = scaler.transform(feature_phys[:,:])

## nan to negative
if nan_to_neg:
    feature_matrix[np.isnan(feature_matrix)]=-1
    print("Changed nan to -1")
    
## ESN Generation
N = N_def           # Neurons
mem = mem_def       # memory
scale = scale_def   # scaling factor

## Nonlinear mapping function
sigmoid_exponent = exponent_def
func = ESNtools.sigmoid

## Mask parameters
M = 2*np.random.rand(np.shape(feature_matrix)[1],N)-1
Mb = 2*np.random.rand(1,N)-1

## Perform ESN feed
ESN = ESNtools.feedESN(feature_matrix, N, M, Mb, scale, mem, func, sigmoid_exponent)
del feature_matrix

## Target Y 
Y = sepsis_label

## Obtain weights
w = ESNtools.get_weights_lu_biasedNE(ESN, Y)

import platform
import time

user = platform.uname()[1] + '@' + platform.platform() 
dir_path = os.path.dirname(os.path.realpath(__file__))

## Write results
wfile = 'w_' + script_name + name_struct + '.txt'
with open(wfile, 'w') as f:
    f.write('Weights N=%d %s' % (N, dataset))
    if dataloaded:
        for s in w:
            f.write('\n%g' % s)

## write to report file
output_file = 'report_' + script_name + name_struct + '.txt'
with open(output_file, 'w') as f:
    f.write(user + '\n')
    f.write(dir_path + '\n')
    f.write(__file__ + '\n')
    f.write(time.strftime("%Y-%m-%d %H:%M") + '\n')
    f.write('Dataset: ' + path + '\n')
    f.write('{:03d} \t N \n'.format(N))
    f.write('%d \t exp\n' % sigmoid_exponent)
    f.write('{:1.3f} \t scale \n'.format(scale))
    f.write('{:1.3f} \t mem \n'.format(mem))

    
print(user)
print(dir_path)
print(__file__)
print(time.strftime("%Y-%m-%d %H:%M"))
print('Dataset: ' + path)
print('N: {:03d}'.format(N))
print('exp: {:1.3f}'.format(sigmoid_exponent))
print('scale: {:1.3f}'.format(scale))
print('mem: {:1.3f}'.format(mem))

