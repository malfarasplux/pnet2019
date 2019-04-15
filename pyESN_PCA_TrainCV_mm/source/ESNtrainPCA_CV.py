#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Config
dataset = "training"
path = "../../" + dataset +"/"
kfold_split = 10
nan_to_neg = True
biased_regress = True
normal_equations = True
mm = True
std = False
numpy_load = True

## ESN parameters
N_def = 100         # Neurons
scale_def = 0.500   # scaling
mem_def = 0.500      # memory
exponent_def = 1    # sigmoid exponent

# Script name struct for report
script_name = 'ESNtrainPCA_CV'
name_struct_meta = "_N_scale_mem"
name_struct = '_{:03d}_{:1.3f}_{:1.3f}'.format(N_def, scale_def, mem_def)

## Imports
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import ESNtools

## Read data functions
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

## Get sepsis patients
def get_sepsis_patients(sepsis_label, patient):
    patient_sep = np.zeros(len(sepsis_label),dtype=np.int)
    for i in range(n):
        i_pat = np.where(patient==i)[0]
        patient_sep[i_pat] = int(np.sum(sepsis_label[i_pat])>0)*np.ones(len(i_pat), dtype=np.int)
        
    patient_sep_idx = patient[np.where(patient_sep!=0)]
    patient_healthy_idx = patient[np.where(patient_sep==0)]
    return patient_sep, patient_sep_idx, patient_healthy_idx


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
    dataloaded = True
    
else:
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

## Divide in sets
X = ESN
y = sepsis_label
groups = patient

skf = StratifiedKFold(n_splits=kfold_split)
skf.get_n_splits(X)

## KFold
results = []
target = []
kk = 0
for train_index, test_index in skf.split(X,y):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    pca = PCA(.8, whiten=True)
    pca = pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    
    if biased_regress:
        if normal_equations:
            w = ESNtools.get_weights_lu_biasedNE(X_train, y_train)

        Y_pred = (np.matmul(X_test,w))

    else:
        ESNinv = np.linalg.pinv(X_train)
        w = np.matmul(ESNinv, y_train)
        Y_pred = (np.matmul(X_test,w))

    print(kk, ' realisation ')
    print("auc: ", roc_auc_score(y_test, Y_pred))
    kk +=1
    target.append(y_test)
    results.append(Y_pred)

## Evaluate results
results = np.concatenate(results)
target = np.concatenate(target)
auc = roc_auc_score(target,results)
print('auc: ', auc)

## Threshold study
th_i = np.min(results)
th_f = np.max(results)
th_steps = 1000
th_step = (th_f-th_i)/th_steps
thsum = 0
th = np.zeros((1000, 1), dtype = np.double)
f1 =np.zeros((1000, 1), dtype = np.double)

print("Threshold: Loop between ",  th_i, th_i+th_step*th_steps)
for i, j in enumerate(np.arange(th_i, th_f, th_step)):
    if j < th_steps:
        th[i] = j
        f1[i] = f1_score(target, results > th[i])
        thsum = thsum + th[i]
        if i%100 == 0:
            print(i, th[i], f1[i])

        if f1[i] < 0.001 and np.abs(thsum) > 0:
            th = th[:i]
            f1 = f1[:i]
            break

## Max Threshold
th_max = th[np.argmax(f1)]

## Metrics
Pr = precision_score(target, results > th_max)
Re = recall_score(target, results > th_max)
ACC = accuracy_score(target, results > th_max)
auc = roc_auc_score(target, results)
f1 = f1_score(target, results > th_max)

patient_sep = np.zeros(len(sepsis_label),dtype=np.int)
for i in range(n):
    i_pat = np.where(patient==i)[0]
    patient_sep[i_pat] = int(np.sum(sepsis_label[i_pat])>0)*np.ones(len(i_pat), dtype=np.int)
    
np.sum(patient_sep)

np.sum(sepsis_label)

import platform
import time

user = platform.uname()[1] + '@' + platform.platform() 
dir_path = os.path.dirname(os.path.realpath(__file__))

# write to report file
output_file = 'report_' + script_name + name_struct + '.txt'
with open(output_file, 'w') as f:
    f.write(user + '\n')
    f.write(dir_path + '\n')
    f.write(__file__ + '\n')
    f.write(time.strftime("%Y-%m-%d %H:%M") + '\n')
    f.write('Dataset: ' + path + '\n')
    f.write('{:03d} \t N \n'.format(N))
    f.write('{:1.3f} \t scale \n'.format(scale))
    f.write('{:1.3f} \t mem \n'.format(mem))
    f.write('%d \t exp\n' % sigmoid_exponent)
    f.write('(%2.4f, %2.4f, %2.4f) \t th_i, th_f, *th_sc\n' % (th_i, th_f, th_f-th_i))
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
print('N: {:03d}'.format(N))
print('scale: {:1.3f}'.format(scale))
print('mem: {:1.3f}'.format(mem))
print('exp: %d' % sigmoid_exponent)
print('th_i, th_f, *th_sc: (%2.4f, %2.4f, %2.4f)' % (th_i, th_f, th_f-th_i))
print('th: %2.4f' % th_max)
print('Pr: %2.4f' % Pr)
print('Re: %2.4f' % Re)
print('F1: %2.4f' % f1)
print('ACC: %2.4f' % ACC)
print('AUC: %2.4f' % auc)

## Write results
# write to report file
#res = 'res_' + script_name + name_struct + '.out'
#with open(res, 'w') as f:
#    f.write('Target|Predicted|PredictedLabel %g' % th_max)
#    if dataloaded:
#        for (s, l) in zip(target, results):
#            f.write('\n%d|%g|%d' % (s, l, int(l>th_max)))
