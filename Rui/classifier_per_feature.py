import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, GroupShuffleSplit, ShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier


def block_hour(dataset, labels, patient_id_array, hours=4):
    new_dataset = []
    for column in dataset.T:
        new_feature = []
        new_labels = []
        for patient_id in np.unique(patient_id_array):
            print("Patient ID: ", patient_id)
            patient_index = np.where(patient_id_array == patient_id)[0]
            patient = dataset[patient_index]
            patient_labels = labels[patient_index]
            for i in range(len(patient_labels) - hours):
                if 1 in patient_labels[i:i + hours]:
                    num = 1
                else:
                    num = 0
                new_feature.append(column[i:i + hours])
                new_labels.append(num)
        new_dataset.append(np.array(new_feature))
    return np.array(new_dataset), np.array(new_labels)


features = np.nan_to_num(np.load('Datasets/training_1_nanfill.npy'))
labels = np.load('Datasets/training_1_Y.npy')
patient_id = np.load('Datasets/training_1_patient.npy')

X, y = block_hour(features, labels, patient_id, 4)
# X, y = features, labels

subs = ['MEAN', 'NORMAL', 'ZERO']
i = 0
acc = {}
f1 = {}
auc = {}
skf = StratifiedKFold(n_splits=10)

acc[subs[i]] = []
f1[subs[i]] = []
auc[subs[i]] = []
res = []
y_test_all = []

for train_index, test_index in skf.split(X[0], y):
    print("TRAIN:", train_index, "TEST:", test_index)
    aux_res = []
    for feature in X:
        X_train, X_test = feature[train_index], feature[test_index]
        y_train, y_test = y[train_index], y[test_index]

        elf = RandomForestClassifier(n_estimators=20)
        print("Start training...")

        elf = elf.fit(X_train, y_train)
        print("Start testing...")
        pred = elf.predict_proba(X_test)[:, 1]
        results = elf.predict(X_test)
        aux_res.append(pred)
        print(subs[i], " Accuracy: ", accuracy_score(results, y_test))
        print(subs[i], "F1-Score: ", f1_score(results, y_test))

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