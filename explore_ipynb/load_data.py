import numpy as np
from os import listdir

path = "../training/"
paths = [path + "p0" + str(10000+i)[1:] + ".psv" for i in range(1, 5001)]
#data = np.array([np.loadtxt(open(path + file), delimiter='|', skiprows=1) for file in listdir(path)])

# Modified data load using ../ paths
data = np.array([np.loadtxt(open(path + file), delimiter='|', skiprows=1) for file in paths])
# data = np.array([np.loadtxt(open(path), delimiter='|', skiprows=1) for path in paths])
keys = open(paths[0]).readline().rstrip().split('|')


def delete_columns_nan(data,keys):
    df = {}
    for i, column in enumerate(data.T):
        if not np.isnan(column).all():
            df[keys[i]] = column
    return df


def replace_nan_by_value(data, value=None):
    for j, patient in enumerate(data):
        for key in patient.keys():
            if value == 'normal':
                for i in range(len(patient[key])):
                    p = patient[key].copy()[~np.isnan(patient[key])]
                    if np.isnan(patient[key][i]):
                        if value == 'mean':
                            data[j][key][i] = np.mean(p)
                        if value == 'normal':
                            data[j][key][i] = np.random.normal(np.mean(p), np.std(p))
            else:
                patient[key] = np.nan_to_num(patient[key])
    return data


# Remove nan-ONLY columns?
data_aux = []
for patient in data:
    data_aux.append(delete_columns_nan(patient, keys))

# Count non-EMPTY entries throughout patients
df = {}
for key in keys:
    df[key] = 0

for patient in data_aux:
    for key in patient.keys():
        df[key] += 1
  
for key,val in df.items():
    print (key,":", val)
    
    
data_new = replace_nan_by_value(data_aux, None)

# Get the list of patients that have feature X
pat_feat = {}
for i in range(len(keys)):
    L = []
    for j in range(5000):
        if keys[i] in data_new[j]:
            print(i, j) 
            L.append(j)              
    pat_feat[i] = L
                