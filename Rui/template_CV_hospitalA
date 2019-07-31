import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.ensemble import GradientBoostingClassifier
from GroupStratifiedKFold import GroupStratifiedKFold
import pickle


features = np.nan_to_num(np.load('Datasets/training_setA_nanfill_mm.npy'))
labels = np.load('Datasets/training_setA_Y.npy')
patient_id = np.load('Datasets/training_setA_patient.npy').reshape(-1, 1)
labels_patient = np.load("Datasets/dataset_A_normal_subs.npy")[:, -2].reshape(-1,1)

model = pickle.load( open( "grid_search_object.p", "rb" ) )
best_parameters = model.best_params_

X, y = features, labels
subs = ['MEAN', 'NORMAL', 'ZERO']
i = 0
acc = {}
f1 = {}
auc = {}

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

for j in range(len(train_index)):
	print("TRAIN:", train_index[j], "TEST:", test_index[j])
	X_train, X_test = X[train_index[j]], X[test_index[j]]
	y_train, y_test = y[train_index[j]], y[test_index[j]]
	patients_id_train, patients_id_test = patient_id[train_index[j]], patient_id[test_index[j]]

	elf = GradientBoostingClassifier(best_parameters)
	print("Start training...")

	elf = elf.fit(X_train, y_train)
	print("Start testing...")
	pred = elf.predict_proba(X_test)[:, 1]
	results = elf.predict(X_test)

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
step = 0.0005

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

try:
    np.savetxt("threshold.csv", np.array([new_threshold]))
except:
    pickle.dump(new_threshold, open("threshold.p", "wb"))

acc[subs[i]].append(accuracy_score(y_test_all, new_results))
f1[subs[i]].append(f1_score(y_test_all, new_results))
print(subs[i], "Accuracy th: ", accuracy_score(y_test_all, new_results))
print(subs[i], "F1-Score th: ",f1_score(y_test_all, new_results))
