#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Config
dataset = "training_setB"
path = "../" + dataset +"/"
nan_to_neg = True
biased_regress = True
normal_equations = True
mm = True
std = False
numpy_load = True
multi_files = True
th_max = 0.0802 
th_min = -0.1545
th_scale = 0.4675

## ESN parameters
N_def = 200         # Neurons
scale_def = 0.50    # scaling
mem_def = 0.50      # memory
exponent_def = 1.0  # sigmoid exponent

# Script name struct for report
script_name = 'eval_ESNtest_mm'
name_struct_meta = "_N_scale_mem"
name_struct = '_{:03d}_{:1.3f}_{:1.3f}'.format(N_def, scale_def, mem_def)

## Imports
import sys
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import ESNtools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
#import warnings


## Random seed
np.random.seed(seed=0)

## Create the feature matrix
features = []
patient = []
sepsis_label = []
dataloaded = False

## Get sepsis patients
def get_sepsis_patients(sepsis_label, patient):
    patient_sep = np.zeros(len(sepsis_label),dtype=np.int)
    for i in range(n):
        i_pat = np.where(patient==i)[0]
        patient_sep[i_pat] = int(np.sum(sepsis_label[i_pat])>0)*np.ones(len(i_pat), dtype=np.int)
        
    patient_sep_idx = patient[np.where(patient_sep!=0)]
    patient_healthy_idx = patient[np.where(patient_sep==0)]
    return patient_sep, patient_sep_idx, patient_healthy_idx

## Read ESN weights
def read_ESN_weights(input_wfile, N):
    with open(input_wfile) as file:
        w = (np.loadtxt(file, skiprows=1))
    return w

## Read data 
if (not numpy_load) and (multi_files):
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
    
elif numpy_load and multi_files:
    if mm:
        npyfilename = "../npy/" + dataset + "_mm.npy"
        mm = False
        print(npyfilename, '(mm) to be loaded')

    else:
        npyfilename = "../npy/" + dataset + ".npy"
        print(npyfilename, '(not mm) to be loaded')

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

## Normalize mm(all) or std (sepsis, phys) vals, feature-based
if mm:
    scaler = MinMaxScaler()
    for i in range(n):
        i_pat = np.where(patient==i)[0]
        scaler.fit(feature_matrix[i_pat,:])
        feature_matrix[i_pat,:] = scaler.transform(feature_matrix[i_pat,:])
#        with np.warnings.catch_warnings():
#            np.warnings.filterwarnings('ignore', 'All-NaN slice encountered')
#            warnings.filterwarnings('ignore', category=RuntimeWarning)
#            feature_matrix[i_pat,:] = scaler.transform(feature_matrix[i_pat,:])

elif std:
    ## (Get sepsis patients)
    patient_sep, patient_sep_idx, patient_healthy_idx = get_sepsis_patients(sepsis_label, patient)
    scaler = StandardScaler()
    scaler.fit(feature_phys[patient_healthy_idx,:])
    feature_phys[:,:] = scaler.transform(feature_phys[:,:])

## nan to negative
if nan_to_neg:
    feature_matrix[np.isnan(feature_matrix)]=-1
    print("Changed nan to -1")
    
## ESN Generation parameters
N = N_def           # Neurons
mem = mem_def       # memory
scale = scale_def   # scaling factor

## Load ESN weights
print('w' + name_struct_meta, './w' + name_struct + '.txt', "wfile to be loaded")
input_wfile = './w' + name_struct + '.txt'
w = read_ESN_weights(input_wfile, N)
print('w: ', np.shape(w))

## Nonlinear mapping function
sigmoid_exponent = exponent_def
func = ESNtools.sigmoid

## Mask parameters
M = 2*np.random.rand(np.shape(feature_matrix)[1],N)-1
Mb = 2*np.random.rand(1,N)-1

## Perform ESN feed
print('ESN: ')
ESN = ESNtools.feedESN(feature_matrix, N, M, Mb, scale, mem, func, sigmoid_exponent)
del feature_matrix


## Compute class prediction
Y_pred = (np.matmul(ESN,w))

scores = (Y_pred - th_min) / th_scale
labels = np.asarray(Y_pred > th_max, dtype = np.int)
scores[np.where(scores > 1.0)[0]]=1.0
scores[np.where(scores < 0.0)[0]]=0.0
 
## Target Y 
#Y = sepsis_label

## Obtain weights
#w = ESNtools.get_weights_lu_biasedNE(ESN, Y)


## Eval ESN Sepsis Labels
#print('sl' + name_struct_meta, './w' + name_struct + '.txt', "labels to be loaded")
#npy_labels = './w' + name_struct + '.txt'
#sl = np.load(npyfilename)
#print('w: ', np.shape(w))
## Metrics
target = sepsis_label
Pr = precision_score(target, labels)
Re = recall_score(target, labels)
ACC = accuracy_score(target, labels)
#auc = roc_auc_score(target, Y_pred)
auc = roc_auc_score(target, scores)
f1 = f1_score(target, labels)




## REPORT AND RESULTS
import platform
import time
user = platform.uname()[1] + '@' + platform.platform() 
dir_path = os.path.dirname(os.path.realpath(__file__))

## Write class results
# write predictions to output file
if not multi_files:
    record_name = sys.argv[1]
    if record_name.endswith('.psv'):
        record_name = record_name[:-4]

else:
    record_name = 'y_' + script_name + name_struct

output_file = record_name + '.out'
with open(output_file, 'w') as f:
    f.write('PredictedProbability|PredictedLabel\n')
    if patient.size != 0:
        for (s, l) in zip(scores, labels):
            f.write('%1.3f|%d\n' % (s, l))


## write to report file
output_file = 'report_' + script_name + name_struct + '.txt'
with open(output_file, 'w') as f:
    f.write(user + '\n')
    f.write(dir_path + '\n')
    f.write(__file__ + '\n')
    f.write(time.strftime("%Y-%m-%d %H:%M") + '\n')
    f.write('Test Dataset: ' + path + '\n')
    f.write('{:03d} \t N \n'.format(N))
    f.write('{:1.3f} \t scale \n'.format(scale))
    f.write('{:1.3f} \t mem \n'.format(mem))
    f.write('{:1.3f} \t exp\n' .format(sigmoid_exponent))
    f.write('%2.4f \t th\n' % th_max)
    f.write('%2.4f \t Pr\n' % Pr)
    f.write('%2.4f \t Re\n' % Re)
    f.write('%2.4f \t F1\n' % f1)
    f.write('%2.4f \t ACC\n' % ACC)
    f.write('%2.4f \t AUC\n' % auc)
    
    
print(user)
print(dir_path)
print(__file__)
print(time.strftime("%Y-%m-%d %H:%M"))
print('Dataset: ' + path)
print('N: %d' % N)
print('scale: %2.4f' % scale)
print('mem: %2.4f' % mem)
print('exp: %2.4f' % sigmoid_exponent)
print('th: %2.4f' % th_max)
print('Pr: %2.4f' % Pr)
print('Re: %2.4f' % Re)
print('F1: %2.4f' % f1)
print('ACC: %2.4f' % ACC)
print('AUC: %2.4f' % auc)

