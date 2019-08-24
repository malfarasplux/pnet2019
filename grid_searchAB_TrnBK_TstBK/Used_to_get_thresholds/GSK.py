import numpy as np
def GroupStratifiedKFold(dataset, n_split=10):
    '''
	Split the data in Stratified K folds considering the groups.
	Namely, the same group can not appear in the training and testing set at the same time (iteration).
	Note: This function only works with binary tasks.
	
	:params:
	    dataset: (numpy.array)
		    The dataset to split in train and test sets. It is expected to have the format of: features, 
			labels per sample, labels per patient, patient ID.
	    n_split: (int)
			The number of folds to use in cross validation.
	
	:returns:
	    train_indexes: (numpy.array)
		    Indexes of the trainning samples in order.
			It should be iterated such as: X_train, y_train = X[train_indexes[i]], y[train_indexes[i]]
		test_indexes: (numpy.array)
		    Indexes of the trainning samples in order.
			It should be iterated such as: X_test, y_test = X[test_indexes[i]], y[test_indexes[i]]
			
		Note that the iteration of train_indexes[i] and test_indexes[i] should always be the same!
		
	:Example:
		
		...
		
		train_index, test_index = GroupStratifiedKFold(np.hstack([features, labels.reshape(-1,1), labels_patient, patient_id]), 10)
    
		for j in range(len(train_index)):
			print("TRAIN:", train_index[j], "TEST:", test_index[j])
			X_train, X_test = X[train_index[j]], X[test_index[j]]
			y_train, y_test = y[train_index[j]], y[test_index[j]]
			
			...
	'''
    old_id = 0
    patient = []
    patients_healthy_ind = []
    patients_sepsis_ind = []
    aux_ind = []
    for i, sample in enumerate(dataset):
        if sample[-1] == old_id:
            patient.append(sample[-2])
            aux_ind.append(i)
        else:
            old_id = sample[-1]
            if 1 in np.array(patient):
                patients_sepsis_ind.append(np.array(aux_ind, dtype=int))
            else:
                patients_healthy_ind.append(np.array(aux_ind, dtype=int))
            patient = [sample[-2]]
            aux_ind = [i]
    if 1 in np.array(patient):
        patients_sepsis_ind.append(np.array(aux_ind, dtype=int))
    else:
        patients_healthy_ind.append(np.array(aux_ind, dtype=int))

    patients_healthy = np.array(patients_healthy_ind)
    patients_sepsis = np.array(patients_sepsis_ind)

    print(patients_healthy.shape)
    print(patients_sepsis.shape)

    len_sepsis = len(patients_sepsis)
    len_healthy = len(patients_healthy)

    sepsis_nbr = int(len_sepsis // n_split)
    healthy_nbr = int(len_healthy // n_split)

    test_sepsis = []
    test_healthy = []
    train_sepsis = []
    train_healthy = []

    ind = 0
    while ind in range(n_split):
        first_in_sepsis = ind*sepsis_nbr
        second_in_sepsis = first_in_sepsis + sepsis_nbr

        print("Sepsis: ", first_in_sepsis, second_in_sepsis)

        first_in_healthy = ind * healthy_nbr
        second_in_healthy = first_in_healthy + healthy_nbr

        print("Normal: ", first_in_healthy, second_in_healthy)

        test_sepsis.append(np.concatenate(patients_sepsis[range(first_in_sepsis, second_in_sepsis)]))
        train_sepsis.append(np.concatenate(patients_sepsis[
            np.concatenate([range(first_in_sepsis), range(second_in_sepsis, len_sepsis)]).astype(int)]))
        test_healthy.append(np.concatenate(patients_healthy[range(first_in_healthy, second_in_healthy)]))
        train_healthy.append(np.concatenate(patients_healthy[
            np.concatenate([range(first_in_healthy),range(second_in_healthy, len_healthy)]).astype(int)]))
        ind += 1

    return np.array([np.concatenate([train_healthy[j], train_sepsis[j]]) for j in range(n_split)]),\
           np.array([np.concatenate([test_healthy[i], test_sepsis[i]]) for i in range(n_split)])

