#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#ENSEMBLES keep function
def keep_ensembles(i, ensemble_results, results, ensemble_target, target, ensemble_patient, patient):
    if i==0:
        return ensemble_results.append(results), ensemble_target.append(target), ensemble_patient.append(patient)
    else: 
        return ensemble_results.append(results), ensemble_target, ensemble_patient

# ...
# ...
# ...
# ...
# ...




    ## ENSEMBLE LOOP
    ensemble_results = []
    ensemble_target = [] 
    ensemble_patient = []    
    get_ensembles = True
    if get_ensembles:
        ensemble_max = 10
    else:
        ensemble_max = 1
    
    for ESNi in range(ensemble_max):
        print('ESN: ')
        allocateESN = True

# ...
# ...
# ...
# ...
# ...


        ## Evaluate results
        results = np.concatenate(results)
        target = np.concatenate(target)
        auc = roc_auc_score(target,results)
        print('auc: ', auc)

        ## ENSEMBLES keep
        ensemble_results, ensemble_target, ensemble_patient = keep_ensembles(ESNi, ensemble_results, results, ensemble_target, target, ensemble_patient, patient)    
        ensemble_results = np.array(ensemble_results)
        ensemble_target = np.concatenate(ensemble_target)
        ensemble_patient = np.concatenate(ensemble_patient)
        auc = roc_auc_score(ensemble_target,ensemble_results) #ensembles substitute the original auc
