import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, GroupShuffleSplit, ShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from GroupStratifiedKFold import GroupStratifiedKFold


def block_hour(dataset, labels, patient_id_array, hours=4):
    '''
	Function to build the dataset with blocks of hours of acquisition.
	The blocks are built for each patient, so, there are no blocks consisting of various patients.
	The number of columns of the returned dataset will be number_features * hours.
	The number of lines will be number_lines_dataset - (num_patients * (hours - 1))
	
	:params:
	    dataset: (numpy.array)
		    The set of features of the dataset.
		labels: (numpy.array)
		    The labels per patient of the dataset.
		patient_id_array: (numpy.array)
		    Patient ID of the original dataset.
		hours: (int)
		    Number of hour for each block.
	
	:return:
	    new_dataset: (numpy.array)
		    The new set of features containing the blocks of hours.
		new_labels: (numpy.array)
		    The labels of the newly constructed dataset.
			A block is considered positive (1) if any hour of the block has the positive label (1).
	'''
    new_dataset = []
    new_labels = []
    for patient_id in np.unique(patient_id_array):
        print("Patient ID: ", patient_id)
        patient_index = np.where(patient_id_array == patient_id)[0]
        patient = dataset[patient_index]
        patient_labels = labels[patient_index]
        for i in range(len(patient) - hours):
            if 1 in patient_labels[i:i + hours]:
                num = 1
            else:
                num = 0
            new_dataset.append(np.concatenate([np.concatenate(patient[i:i + hours, :-3])]))
            new_labels.append(num)
    return np.array(new_dataset), np.array(new_labels)


'''
Code to load the interpolated  dataset and perform the cross validation using GradientBoostingClassifier.
It is using the GroupStratifiedKFold and optimises the threshold for classification after cross-validation.
'''
features = np.nan_to_num(np.load('Datasets/training_setA_nanfill_mm.npy'))
labels = np.load('Datasets/training_setA_Y.npy')
patient_id = np.load('Datasets/training_setA_patient.npy').reshape(-1, 1)
labels_patient = np.load("Datasets/dataset_A_normal_subs.npy")[:, -2].reshape(-1,1)

# X, y = block_hour(features, labels, patient_id, 5)
X, y = features, labels
subs = ['MEAN', 'NORMAL', 'ZERO']
i = 0
acc = {}
f1 = {}
auc = {}
# skf = StratifiedKFold(n_splits=10)

acc[subs[i]] = []
f1[subs[i]] = []
auc[subs[i]] = []
res = []
y_test_all = []

print("Features: ", np.shape(features))
print("Labels: ", np.shape(labels))
print("labels_patient: ", np.shape(labels_patient))
print("Patient ID: ", np.shape(patient_id))

train_index, test_index = GroupStratifiedKFold(np.hstack([features, labels.reshape(-1,1), labels_patient, patient_id]), 10)

# for train_index, test_index in skf.split(X, y):
for j in range(len(train_index)):
	print("TRAIN:", train_index[j], "TEST:", test_index[j])
	X_train, X_test = X[train_index[j]], X[test_index[j]]
	y_train, y_test = y[train_index[j]], y[test_index[j]]
	patients_id_train, patients_id_test = patient_id[train_index[j]], patient_id[test_index[j]]

	# X_train, y_train = SMOTE().fit_resample(X_train, y_train)

	# scaler = MinMaxScaler(feature_range=(0, 1))
	# scaler = scaler.fit(X_train)
	# X_train = scaler.transform(X_train)
	# X_test = scaler.transform(X_test)


	# elf = RandomForestClassifier(n_estimators=20, class_weight='balanced')
	elf = GradientBoostingClassifier(n_estimators=200)
	print("Start training...")

	elf = elf.fit(X_train, y_train)
	print("Start testing...")
	pred = elf.predict_proba(X_test)[:, 1]
	results = elf.predict(X_test)

	# l = 0
	# previous_id = patients_id_test[l]
	# previous_results = results[l]
	# previous_pred = pred[l]
	# while l in range(len(patients_id_test)):
	#     print("Changing patient: ", patients_id_test[l])
	#     if patients_id_test[l] == previous_id and previous_results == 1:
	#         results[l] = 1
	#     if patients_id_test[l] == previous_id and previous_pred > .5:
	#         pred[l] = 1
	#     previous_id = patients_id_test[l]
	#     previous_results = results[l]
	#     l += 1

	res.append(pred)
	y_test_all.append(y_test)
	acc[subs[i]].append(accuracy_score(results, y_test))
	f1[subs[i]].append(f1_score(results, y_test))
	auc[subs[i]].append(roc_auc_score(y_test, pred))
	print(subs[i], " Accuracy: ", accuracy_score(results, y_test))
	print(subs[i], "F1-Score: ", f1_score(results, y_test))
	print(subs[i], "AUC: ", auc)

res = np.concatenate(res)
y_test_all = np.concatenate(y_test_all)

fpr, tpr, thresholds = roc_curve(y_test_all, res, pos_label=1)

threshold = 0
accuracy = []
f1_score_list = []
step = 0.001

for threshold in np.arange(0, 1, step):
	print(threshold)
	new_results = np.zeros(len(res))
	new_results[np.where(res > threshold)[0]] = 1
	new_results[np.where(res <= threshold)[0]] = 0
	accuracy.append(accuracy_score(y_test_all, new_results))
	f1_score_list.append(f1_score(y_test_all, new_results))
print(accuracy_score(y_test_all, new_results))
print(f1_score(y_test_all, new_results))

new_threshold = np.array(f1_score_list).argmax() * step
print(subs[i], "Threshold:", new_threshold)

new_results = np.zeros(len(res))
new_results[np.where(res > new_threshold)[0]] = 1
new_results[np.where(res <= new_threshold)[0]] = 0

acc[subs[i]].append(accuracy_score(y_test_all, new_results))
f1[subs[i]].append(f1_score(y_test_all, new_results))
print(subs[i], "Accuracy th: ", accuracy_score(y_test_all, new_results))
print(subs[i], "F1-Score th: ",f1_score(y_test_all, new_results))