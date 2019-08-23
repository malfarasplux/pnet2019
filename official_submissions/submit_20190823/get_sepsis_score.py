#!/usr/bin/env python

import numpy as np


def get_sepsis_score(data, model):
    feature_matrix = data
    feature_matrix[np.isnan(feature_matrix)]=-1

    # Use model parameters
    ESNtools = model['f']

    ## ESN Generation parameters
    N = model['N_def']                        # Neurons
    mem = model['mem_def']                    # memory
    scale = model['scale_def']                # scaling factor

    ## Nonlinear mapping function
    sigmoid_exponent = model['exponent_def']  # sig exponent
    func = ESNtools.sigmoid
    
    ## Mask parameters
    # M = 2*np.random.rand(np.shape(feature_matrix)[1],N)-1
    # Mb = 2*np.random.rand(1,N)-1
    M = model['M']
    Mb = model['Mb']    


    ##Weights and thresholds
    w = model['w']
    th_max = model['th_max'] 
    th_min = model['th_min']
    th_scale = model['th_scale']

    ## Perform ESN feed
    # Apply backwards interpolation
    for f in range(feature_matrix.shape[1]):
        if np.sum(np.isnan(feature_matrix[:, f])) < len(feature_matrix[:, f]):
            ESNtools.nan_bounds(feature_matrix[:, f])
            ESNtools.nan_interpolate(feature_matrix[:, f])
        else:
            feature_matrix[:, f] = np.nan_to_num(feature_matrix[:, f], -1)
    ESN = ESNtools.feedESN(feature_matrix, N, M, Mb, scale, mem, func, sigmoid_exponent)

    del feature_matrix
    
    ## Compute class prediction
    single_sample = True
    if single_sample:
        Y_pred = (np.matmul(ESN[-1, :], w))
        scores = (Y_pred - th_min) / th_scale
        labels = np.asarray(Y_pred > th_max, dtype = np.int)
        scores[np.where(scores > 1.0)[0]]=1.0
        scores[np.where(scores < 0.0)[0]]=0.0

    else:
        Y_pred = (np.matmul(ESN,w))
        scores = (Y_pred - th_min) / th_scale
        labels = np.asarray(Y_pred > th_max, dtype = np.int)
        scores[np.where(scores > 1.0)[0]]=1.0
        scores[np.where(scores < 0.0)[0]]=0.0
        print(scores)
        print(np.shape(scores))
    return scores, labels

def load_sepsis_model():
    import scipy.linalg as linalg
    
    # Random seed
    np.random.seed(seed=0)
    class ESNT:
        """
        ESN tools module
        """
        
        ### Map data ################################################################
        @staticmethod
        def sigmoid(x, exponent):
            """Apply a [-0.5, 0.5] sigmoid function."""
            
            return 1/(1+np.exp(-exponent*x))-0.5
        
        ### Feed data into Echo State Network #######################################
        @staticmethod
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
            p = np.zeros((1,neurons),dtype=np.double)
        
            for i in range(np.shape(features)[0]):
                in_val = scale * (ESN[i,:-1] + mask_bias) + p * mem
                
                ## Apply transform
                ESN[i,:-1] = func(in_val, f_arg)
                
                ## Connect preceding neighbour 
                p = np.copy(np.roll(ESN[i,:-1],1))
            return ESN

        # Fix boundary nans (replicate head/tail vals)
        @staticmethod
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
        @staticmethod
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

    esnt = ESNT()
    model = dict()
    with open('w.txt') as file:
        w = (np.loadtxt(file, skiprows=1))
        
    # Model parameters
    model['N_def'] = 100         # Neurons
    model['scale_def'] = 0.0001  # scaling
    model['mem_def'] = 1.0       # memory
    model['exponent_def'] = 1.0  # sigmoid exponent

    # Thresholds
    model['th_max'] = 0.0575
    model['th_min'] = -0.4868
    model['th_scale'] = 17.0103
    
    # Model functions
    model['f'] = esnt
    model['type'] = 'ESN'
    model['w'] = w

    # Model Mask
    model['M'] = 2*np.random.rand(40, model['N_def'])-1
    model['Mb'] = 2*np.random.rand(1, model['N_def'])-1

    return model
