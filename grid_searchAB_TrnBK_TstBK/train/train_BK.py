#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Config
# biased_regress = True
# normal_equations = True
dataset = "training_AB"
path = "../" + dataset + "/"
kfold_split = 10
mm = False
std = False
numpy_load = True
nanfill = False

## ESN parameters
N = 100             # Neurons
scale = 0.0001      # scaling
mem = 1.0           # memory
exponent = 1.0      # sigmoid exponent


## Imports
import numpy as np
import os
import ESNtools


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


npyfilename = "../../Rui/Datasets/" + dataset + "_patient.npy"
patient = np.load(npyfilename)
print(npyfilename, " loaded")
npyfilename = "../../Rui/Datasets/" + dataset + "_Y.npy"
sepsis_label = np.load(npyfilename)
print(npyfilename, " loaded")

n = len(np.unique(patient))
print(n, ' files present')

dataloaded = True
feature_matrix = np.load(npyfilename)

##Flatten patient
patient = patient.flatten()

## Separate pointers
feature_phys = feature_matrix[:, :-6]  ## Physiology
feature_demog = feature_matrix[:, -6:]  ## Demographics

## Nonlinear mapping function
sigmoid_exponent = exponent
func = ESNtools.sigmoid

script_name = 'ESNtrainCV'
name_struct_meta = "_N_scale_mem"
name_struct = '_{:03d}_{:1.4f}_{:1.4f}'.format(N, scale, mem)

## ESN Generation parameters

## Perform ESN feed
pat_shift = np.append(np.where(np.diff(patient) != 0)[0] + 1, [len(patient)])

print('ESN: ')
allocateESN = True
#####BACKWARD INTERP FOR THE ESNs
pat_ipos = 0
allocateESN = True

ESN = np.ones((len(feature_matrix), N + 1), dtype=np.float)
M = 2 * np.random.rand(np.shape(feature_matrix)[1], N) - 1
Mb = 2 * np.random.rand(1, N) - 1

for i in range(len(pat_shift)):
    patients_features = feature_matrix[pat_ipos:pat_shift[i]]
    print("Feeding patient ", i)
    for h, hour in enumerate(patients_features):
        features = patients_features[:h + 1]
        for f in range(features.shape[1]):
            if np.sum(np.isnan(features[:, f])) < len(features[:, f]):
                nan_bounds(features[:, f])
                nan_interpolate(features[:, f])
            else:
                features[:, f] = np.nan_to_num(features[:, f], 0)
        ESN[pat_ipos, :] = ESNtools.feedESN(features, N, M, Mb, scale, mem, func, sigmoid_exponent)[-1]
        pat_ipos = pat_ipos + 1


## Divide in sets
X = ESN
y = sepsis_label

w = ESNtools.get_weights_biasedNE(X, y)

import platform

user = platform.uname()[1] + '@' + platform.platform()
dir_path = os.path.dirname(os.path.realpath(__file__))

## Write results
wfile = 'w_' + script_name + name_struct + '.txt'
with open(wfile, 'w') as f:
    f.write('Weights N=%d %s' % (N, dataset))
    if dataloaded:
        for s in w:
            f.write('\n%g' % s)