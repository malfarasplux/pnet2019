#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Config
dataset = "training_AB"
path = "../" + dataset +"/"
kfold_split = 10
nan_to_neg = False
nan_to_zero = True
biased_regress = True
normal_equations = True
mm = False
std = False
numpy_load = True
nanfill = False


## ESN parameters
N_def = 100           # Neurons
scale_def = 0.001     # scaling
mem_def = 0.1         # memory
exponent_def = 1.0    # sigmoid exponent

# Script name struct for report
script_name = 'ESNtrainCV'
name_struct_meta = "_N_scale_mem"
name_struct = '_{:03d}_{:1.3f}_{:1.3f}'.format(N_def, scale_def, mem_def)

## Imports
import numpy as np
import os
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
#import matplotlib.pyplot as plt
import ESNtools
import GSK

# Fix boundary nans (replicate head/tail vals)
def nan_bounds(feats):
    nanidx = np.where(np.isnan(feats))[0]
    pointer_left = 0
    pointer_right = len(feats)-1
    fix_left = pointer_left in nanidx
    fix_right = pointer_right in nanidx
    while fix_left:
        if pointer_left in nanidx:
            pointer_left += 1
            # print("pointer_left:", pointer_left)
        else:
            val_left = feats[pointer_left]
            feats[:pointer_left] = val_left*np.ones((1,pointer_left),dtype=np.float)
            fix_left = False

    while fix_right:
        if pointer_right in nanidx:
            pointer_right -= 1
            # print("pointer_right:", pointer_right)
        else:
            val_right = feats[pointer_right]
            feats[pointer_right+1:] = val_right*np.ones((1,len(feats)-pointer_right-1),dtype=np.float)
            fix_right = False 
        
# nan interpolation
def nan_interpolate(feats):
    nanidx = np.where(np.isnan(feats))[0]
    nan_remain = len(nanidx)
    nanid = 0
    while nan_remain > 0:
        nanpos = nanidx[nanid] 
        nanval = feats[nanpos-1]
        nan_remain -= 1

        nandim = 1
        initpos = nanpos

        # Check whether it extends
        while nanpos+1 in nanidx:
            nanpos += 1
            nanid += 1
            nan_remain -= 1
            nandim += 1
            # Average sides
            if np.isfinite(feats[nanpos+1]):
                nanval = 0.5 * (nanval + feats[nanpos+1])

        # Single value average    
        if nandim == 1:
            nanval = 0.5 * (nanval + feats[nanpos+1])
        feats[initpos:initpos+nandim] = nanval*np.ones((1,nandim),dtype=np.double)
        nanpos += 1
        nanid += 1    

## Get sepsis patients
def get_sepsis_patients(sepsis_label, patient):
    patient_sep = np.zeros(len(sepsis_label),dtype=np.int)
    for i in range(n):
        i_pat = np.where(patient==i)[0]
        patient_sep[i_pat] = int(np.sum(sepsis_label[i_pat])>0)*np.ones(len(i_pat), dtype=np.int)
        
    patient_sep_idx = np.where(patient_sep!=0)[0]
    patient_healthy_idx = np.where(patient_sep==0)[0]
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

    
    npyfilename = "../npy/" + dataset + "_patient.npy"
    patient = np.load(npyfilename)
    print(npyfilename, " loaded")
    npyfilename = "../npy/" + dataset + "_Y.npy"
    sepsis_label = np.load(npyfilename)
    print(npyfilename, " loaded")

#ADD nanfill tag
    if nanfill:
        dataset = dataset + "_nanfill"    
    
    if mm:
        npyfilename = "../npy/" + dataset + "_mm.npy"
        mm = False
        print(npyfilename, '(mm) to be loaded')

    else:
        npyfilename = "../npy/" + dataset + ".npy"
        print(npyfilename, '(not mm) to be loaded')

    n = len(np.unique(patient))
    print(n, ' files present')
    
    dataloaded = True
    feature_matrix = np.load(npyfilename)
    feature_matrix_test = np.copy(feature_matrix)


##Flatten patient
patient = patient.flatten()

## Separate pointers
feature_phys = feature_matrix[:,:-6]    ## Physiology
feature_demog = feature_matrix[:,-6:]   ## Demographics

##Split dataset
patients_id_samples = []
for idx in np.unique(patient):
    patients_id_samples.append(np.where(patient == idx)[0])

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

## nan to zero
if nan_to_zero:
    feature_matrix[np.isnan(feature_matrix)]=0
    print("Changed nan to 0")

## Septic groups stratify
patient_sep, patient_sep_idx, patient_healthy_idx = get_sepsis_patients(sepsis_label, patient)
#healthy_patient_list =  np.unique(patient[patient_healthy_idx])
#sep_patient_list =  np.unique(patient[patient_sep_idx])


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
pat_shift = np.append(np.where(np.diff(patient)!=0)[0] + 1, [len(patient)])
pat_ipos = 0
print("pat_shift: ",len(pat_shift))

allocateESN = True
print('ESN: ')
if allocateESN: 
    ESN = np.ones((len(feature_matrix),N+1), dtype = np.float)    
    for i in range(len(pat_shift)):
        print("Feeding ESN patient:", i)
        ESN[pat_ipos:pat_shift[i],:] = ESNtools.feedESN(feature_matrix[pat_ipos:pat_shift[i]], N, M, Mb, scale, mem, func, sigmoid_exponent)
        pat_ipos = pat_shift[i]
            
else:
    for i in range(len(pat_shift)):
        if i == 0:
            ESN = ESNtools.feedESN(feature_matrix[pat_ipos:pat_shift[i]], N, M, Mb, scale, mem, func, sigmoid_exponent)
        else:
            ESN = np.vstack((ESN, ESNtools.feedESN(feature_matrix[pat_ipos:pat_shift[i]], N, M, Mb, scale, mem, func, sigmoid_exponent)))
        pat_ipos = pat_shift[i]
    
del feature_matrix

## Divide in sets
X = ESN
y = sepsis_label
groups = patient

#SFK
#skf = StratifiedKFold(n_splits=kfold_split)
#skf.get_n_splits(X)

#GSKF
train_index, test_index = GSK.GroupStratifiedKFold(np.hstack([X, y.reshape(-1,1), patient_sep.reshape(-1,1), groups.reshape(-1,1)]), 10)

## KFold
results = []
target = []
kk = 0
#for train_index, test_index in skf.split(X,y): #Stratified KFold
for j in range(len(train_index)):                                  #GSKF
    X_train, X_test = X[train_index[j]], X[test_index[j]]          #GSKF
    y_train, y_test = y[train_index[j]], y[test_index[j]]          #GSKF
    patients_id_train, patients_id_test = patient[train_index[j]], patient[test_index[j]]
    
    if biased_regress:
        if normal_equations:
            w = ESNtools.get_weights_lu_biasedNE(X_train, y_train)
        
            print("Start testing...", flush=True)
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
    f.write('%1.3f \t exp\n' % sigmoid_exponent)
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
print('exp: %1.3f' % sigmoid_exponent)
print('th_i, th_f, *th_sc: (%2.4f, %2.4f, %2.4f)' % (th_i, th_f, th_f-th_i))
print('th: %2.4f' % th_max)
print('Pr: %2.4f' % Pr)
print('Re: %2.4f' % Re)
print('F1: %2.4f' % f1)
print('ACC: %2.4f' % ACC)
print('AUC: %2.4f' % auc)