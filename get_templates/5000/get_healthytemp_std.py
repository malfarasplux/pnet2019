#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Config
dataset = "training_1"
path = "../" + dataset + "/"
numpy_load = True
nanfill = False
mm = False
std = False
nan_to_neg = False

# Script name struct for report
script_name = 'get_healthytemp'

## Imports
import numpy as np
import os


# Fix boundary nans (replicate head/tail vals)
def nan_bounds(feats):
    nanidx = np.where(np.isnan(feats))[0]
    pointer_left = 0
    pointer_right = len(feats) - 1
    fix_left = pointer_left in nanidx
    fix_right = pointer_right in nanidx
    while fix_left:
        if pointer_left in nanidx:
            pointer_left += 1
            # print("pointer_left:", pointer_left)
        else:
            val_left = feats[pointer_left]
            feats[:pointer_left] = val_left * np.ones((1, pointer_left), dtype=np.float)
            fix_left = False

    while fix_right:
        if pointer_right in nanidx:
            pointer_right -= 1
            # print("pointer_right:", pointer_right)
        else:
            val_right = feats[pointer_right]
            feats[pointer_right + 1:] = val_right * np.ones((1, len(feats) - pointer_right - 1), dtype=np.float)
            fix_right = False

        # nan interpolation


def nan_interpolate(feats):
    nanidx = np.where(np.isnan(feats))[0]
    nan_remain = len(nanidx)
    nanid = 0
    while nan_remain > 0:
        nanpos = nanidx[nanid]
        nanval = feats[nanpos - 1]
        nan_remain -= 1

        nandim = 1
        initpos = nanpos

        # Check whether it extends
        while nanpos + 1 in nanidx:
            nanpos += 1
            nanid += 1
            nan_remain -= 1
            nandim += 1
            # Average sides
            if np.isfinite(feats[nanpos + 1]):
                nanval = 0.5 * (nanval + feats[nanpos + 1])

        # Single value average
        if nandim == 1:
            nanval = 0.5 * (nanval + feats[nanpos + 1])
        feats[initpos:initpos + nandim] = nanval * np.ones((1, nandim), dtype=np.double)
        nanpos += 1
        nanid += 1

    ## Get sepsis patients


def get_sepsis_patients(sepsis_label, patient):
    patient_sep = np.zeros(len(sepsis_label), dtype=np.int)
    for i in range(n):
        i_pat = np.where(patient == i)[0]
        patient_sep[i_pat] = int(np.sum(sepsis_label[i_pat]) > 0) * np.ones(len(i_pat), dtype=np.int)

    patient_sep_idx = np.where(patient_sep != 0)[0]
    patient_healthy_idx = np.where(patient_sep == 0)[0]
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
        if i == 0:
            data, sep_lab, columns = ESNtools.read_challenge_data_label(input_file, return_header=True)
        else:
            data, sep_lab = ESNtools.read_challenge_data_label(input_file)
        features.append(data)
        sepsis_label.append(sep_lab)
        pat = i * np.ones((sep_lab.shape), dtype=np.int)
        patient.append(pat)

    feature_matrix = np.concatenate(features)
    del (features)
    sepsis_label = np.concatenate(sepsis_label)
    patient = np.concatenate(patient)
    dataloaded = True

else:
    npyfilename = "../npy/" + dataset + "_patient.npy"
    patient = np.load(npyfilename)
    npyfilename = "../npy/" + dataset + "_Y.npy"
    sepsis_label = np.load(npyfilename)

    # ADD nanfill tag
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

##Flatten patient
patient = patient.flatten()

## Separate pointers
feature_phys = feature_matrix[:, :-6]  ## Physiology
feature_demog = feature_matrix[:, -6:]  ## Demographics

## Normalize mm(all) or std (sepsis, phys) vals, feature-based
if mm:
    scaler = MinMaxScaler()
    for i in range(n):
        i_pat = np.where(patient == i)[0]
        scaler.fit(feature_matrix[i_pat, :])
        feature_matrix[i_pat, :] = scaler.transform(feature_matrix[i_pat, :])

elif std:
    ## (Get sepsis patients)
    patient_sep, patient_sep_idx, patient_healthy_idx = get_sepsis_patients(sepsis_label, patient)
    scaler = StandardScaler()
    scaler.fit(feature_phys[patient_healthy_idx, :])
    feature_phys[:, :] = scaler.transform(feature_phys[:, :])

## nan to negative
if nan_to_neg:
    feature_matrix[np.isnan(feature_matrix)] = -1
    print("Changed nan to -1")

# Compute and save the templates
patient_sep, patient_sep_idx, patient_healthy_idx = get_sepsis_patients(sepsis_label, patient)
healthy_patient_list = np.unique(patient[patient_healthy_idx])
sep_patient_list = np.unique(patient[patient_sep_idx])
htemplist = []
for i in range(len(healthy_patient_list)):
    i_pat = np.where(patient == i)[0]
    htemplist.append(np.hstack((np.nanstd(feature_phys[i_pat, :], axis=0), feature_demog[i_pat[-1], :])))
htemp = np.nanstd(htemplist, axis=0).reshape(1, -1)
htemp_phys = htemp[:, :-6]
htemp_demog = htemp[:, -6:]
np.savetxt(dataset + '_healthytemp_std.txt', htemp, delimiter=', ', fmt='%1.17f')

stemplist = []
for i in range(len(sep_patient_list)):
    i_pat = np.where(patient == i)[0]
    stemplist.append(np.hstack((np.nanstd(feature_phys[i_pat, :], axis=0), feature_demog[i_pat[-1], :])))
stemp = np.nanstd(stemplist, axis=0).reshape(1, -1)
stemp_phys = stemp[:, :-6]
stemp_demog = stemp[:, -6:]
np.savetxt(dataset + '_septemp_std.txt', stemp, delimiter=', ', fmt='%1.17f')

# patient_sep = np.zeros(len(sepsis_label),dtype=np.int)
# for i in range(n):
#     i_pat = np.where(patient==i)[0]
#     patient_sep[i_pat] = int(np.sum(sepsis_label[i_pat])>0)*np.ones(len(i_pat), dtype=np.int)
# np.sum(patient_sep)
# np.sum(sepsis_label)

import platform
import time

user = platform.uname()[1] + '@' + platform.platform()
dir_path = os.path.dirname(os.path.realpath(__file__))

# write to report file
output_file = 'report_' + script_name + '.txt'
with open(output_file, 'w') as f:
    f.write(user + '\n')
    f.write(dir_path + '\n')
    f.write(__file__ + '\n')
    f.write(time.strftime("%Y-%m-%d %H:%M") + '\n')
    f.write('Dataset: ' + path + '\n')

print(user)
print(dir_path)
print(__file__)
print(time.strftime("%Y-%m-%d %H:%M"))
print('Dataset: ' + path)
