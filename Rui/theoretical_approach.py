import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

dataset = np.load("./Datasets/training_AB.npy")
labels = np.load("./Datasets/training_AB_Y.npy")

# Glasgow Coma Scale score of 13 or less,' systolic blood pressure of 100 mm Hg or less,'
# and respiratory rate 22/min or greater

# HR >90 beats per minute,' RR >20 breaths per minute (or partial pressure of arterial CO2 < 32mm Hg),' 
# temperature either >38°C or <36°C,' and WBC either >12,'000 or <4000 cells/mm3

# ['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride
# ','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb
# ','PTT','WBC','Fibrinogen','Platelets','Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS','SepsisLabel']

indexes = {'HR': 0, 'Temp': 2,'SBP': 3, 'Resp': 6, 'etCO2': 7, 'WBC': 31}
limits = {'SBP': 100, 'Resp': 22, 'WBC': [4, 12], 'etCO2': 32, 'HR': 90, 'Temp': [36, 38]}

hr = dataset[:, indexes['HR']]
temp = dataset[:, indexes['Temp']]
sbp = dataset[:, indexes['SBP']]
resp = dataset[:, indexes['Resp']]
etco = dataset[:, indexes['etCO2']]
wbc = dataset[:, indexes['WBC']]

results = []

for sample in range(len(dataset)):
    print(sample)
    com = [hr[sample] > limits['HR'],
           temp[sample] > limits['Temp'][1] or temp[sample] < limits['Temp'][0],
           sbp[sample] < limits['SBP'],
           resp[sample] > limits['Resp'],
           etco[sample] < limits['etCO2'],
           wbc[sample] < limits['WBC'][0] or wbc[sample] > limits['WBC'][1]
           ]
    results.append(int(np.sum(com) > 1))

accuracy = accuracy_score(labels, results)
f1 = f1_score(labels, results)
recall = recall_score(labels, results)
precision = precision_score(labels, results)

print("Accuracy: ", accuracy)
print("F1-Score: ", f1)
print("Recall: ", recall)
print("Precision: ", precision)


import time
# write to report file
output_file = './Results/report_' + 'theoretical' + '.txt'
with open(output_file, 'w') as f:
    f.write(time.strftime("%Y-%m-%d %H:%M") + '\n')
    f.write('Dataset: new_dataset_AB.npy\n')
    f.write('%2.4f \t Pr\n' % precision)
    f.write('%2.4f \t Re\n' % recall)
    f.write('%2.4f \t F1\n' % f1)
    f.write('%2.4f \t ACC\n' % accuracy)
