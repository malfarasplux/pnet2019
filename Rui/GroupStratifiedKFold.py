import numpy as np
from os import listdir


path = "D:\Physionet Challenge\old_data\\"
# path = r"D:\Physionet Challenge\GitHub\pnet2019\training\\"


def generate_new_dataset(path, name='new_dataset_AB.npy'):
    dataset = np.zeros(shape=(1, 43))
    for i, file in enumerate(listdir(path)):
        print(i)
        if True: #'p1' in file:
            with open(path+file, 'r') as f:
                header = f.readline().strip()
                column_names = header.split('|')
                data = np.loadtxt(f, delimiter='|')
                if 1 in data[:, -1]:
                    num = np.ones
                else:
                    num = np.zeros
                data = np.hstack([data, num(shape=(data.shape[0], 1))])
                id = int(file.strip('p.psv'))
                id_set = np.array([id]*data.shape[0]).reshape(-1, 1)
                data = np.hstack([data, id_set])
                dataset = np.vstack([dataset, data])
    print(dataset.shape)
    np.save(name, dataset[1:])


def GroupStratifiedKFold(dataset, n_split=10):
    # TODO: comment on the dataset input
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


# generate_new_dataset(path, 'new_dataset_B.npy')
# dataset = np.load('new_dataset.npy')
# train_indexes, test_indexes = GroupStratifiedKFold(dataset, 10)
