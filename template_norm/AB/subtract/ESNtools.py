#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESN tools module
"""

## Imports
import numpy as np
import scipy.linalg as linalg

### Read data ################################################################
def read_challenge_data(input_file, return_header = False):
    """Reads challenge data ignoring target.

    Parameters
    ----------
    input_file : filename or path to open
    
    return_header : bool to pick column names
    """

    with open(input_file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')

    ## ignore SepsisLabel column if present
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        data = data[:, :-1]
    if return_header:
        return (data, column_names)

    else:
        return data
    
def read_challenge_data_label(input_file, return_header = False):
    """Reads challenge data and target.

    Parameters
    ----------
    input_file : filename or path to open
    
    return_header : bool to pick column names
    """

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

### Map data ################################################################
def sigmoid(x, exponent):
    """Apply a [-0.5, 0.5] sigmoid function."""
    
    return 1/(1+np.exp(-exponent*x))-0.5

def rectifier(x, slope=1):
    """Apply a rectifier function."""
    z = np.zeros(np.shape(x))
    z[~np.less(x,0)] = slope*x[~np.less(x,0)]
    return z
    
### Feed data into Echo State Network #######################################
def feedESN(features, neurons, mask, mask_bias, scale, mem, func, f_arg):
    """Feeds data into a ring Echo State Network. Returns ESN state.
    Adds extra (1) neuron for Ax + b = Y linear system.

    Parameters
    ----------
    features : (np.array) feature matrix original data (samples,features)
    
    neurons : (int) number of neurons to use

    mask : (np.array) input weights mask matrix (usually randomly generated)
    
    mask_bias : (np.array) initialisation bias per neuron
    
    scale : (float) input scaling factor

    mem : (float) memory feedback factor

    func : (function) nonlinear mapping function

    f_arg : (float) function parameter. sigmoid exponent or slope in rect
    """
    
    ESN = np.hstack((np.matmul(features, mask), np.ones((np.shape(features)[0],1), dtype=np.double)))
    #print(np.shape(ESN))
    #print(np.min(ESN), np.max(ESN))
    p = np.zeros((1,neurons),dtype=np.double)

    for i in range(np.shape(features)[0]):
        in_val = scale * (ESN[i,:-1] + mask_bias) + p * mem
        
        ## Apply transform
        ESN[i,:-1] = func(in_val, f_arg)
        
        ## Connect preceding neighbour 
        p = np.copy(np.roll(ESN[i,:-1],1))
    return ESN

### Get ESN training weights (NE: normal eq.) ############################
def get_weights_biasedNE(ESN, target):
    """Computes ESN training weights solving (pinv) the NE linear system w/ bias.
    Parameters
    ----------
    ESN : (np.array) Echo State Network state
    
    target : (np.array) target labels to train with
    
    """
    Y_aux = np.matmul(ESN.T,target)
    ESNinv = np.linalg.pinv(np.matmul(ESN.T,ESN))
    w = np.matmul(ESNinv, Y_aux)
    return w

def get_weights_qr_biasedNE(ESN, target):
    """Computes ESN training weights solving (qr) the NE linear system w/ bias.
    Parameters
    ----------
    ESN : (np.array) Echo State Network state
    
    target : (np.array) target labels to train with
    
    """
    Q, R = linalg.qr((np.matmul(ESN.T,ESN)))            # QR decomposition with qr function (RtR)w = RtY
    Y_aux = np.dot(Q.T, np.matmul(ESN.T, target))       # Let y=Q'.ESNt using matrix multiplication
    w = linalg.solve(R, Y_aux)                          # Solve Rx=y
    return w

def get_weights_lu_biasedNE(ESN, target):
    """Computes ESN training weights solving (lu) the NE linear system w/ bias.
    Parameters
    ----------
    ESN : (np.array) Echo State Network state
    
    target : (np.array) target labels to train with
    
    """
    LU = linalg.lu_factor((np.matmul(ESN.T,ESN)))      # LU decomposition with (RtR)w = RtY
    Y_aux = (np.matmul(ESN.T, target))                 # Let Y=ESNt.y using matrix multiplication
    w = linalg.lu_solve(LU, Y_aux)                     # Solve Rx=y
    return w
