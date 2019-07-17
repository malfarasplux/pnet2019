import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

def replace_nan_by_value(data, value=None):
    data_copy = data.copy()
    if value not in ['mean', 'normal']:
        for n, patient in enumerate(data):
            for l, line in enumerate(patient):
                data_copy[n][l] = np.nan_to_num(line)
        return data_copy
    for i, patient in enumerate(data):
        print(i)
        # patient = data[np.where(data[:, -1] == patient_id)[0]][:, :-3]
        for j, column in enumerate(patient.T):
            new_column = column[np.where(~np.isnan(column))[0]]
            if value == 'mean':
                data_copy[i][:, j][np.where(np.isnan(column))[0]] = np.mean(new_column)
            elif value == 'normal' and ~np.isnan(new_column).all():
                data_copy[i][:, j][np.where(np.isnan(column))[0]] = np.random.normal(np.mean(new_column),
                                                                                     np.std(new_column))
            data_copy[i][:, j] = np.nan_to_num(column)
    return data_copy


def late_onset_sepsis(dataset, hours_to_onset=6):
    flag = False
    i = 0
    while i in range(len(dataset[:-hours_to_onset, -3])):
        print(i)
        if dataset[i+hours_to_onset, -3] == 1 and dataset[i, -3] == 1 and not flag:
            flag = True
            j = i
            while j < i+hours_to_onset:
                print("######################3  HERE  #############################3")
                dataset[j, -3] = 0
                j += 1
            i = j
        if dataset[i, -3] == 0:
            flag = False
        i += 1
    return dataset


datasets = ["Datasets/dataset_A_mean_subs_sepsis_late.npy", "Datasets/dataset_A_normal_subs_sepsis_late.npy",
            "Datasets/dataset_A_zero_subs_sepsis_late.npy"]
subs = ['MEAN', 'NORMAL', 'ZERO']

acc = {}
f1 = {}
auc = {}
skf = StratifiedKFold(n_splits=10)
for i, dataset_name in enumerate(datasets):
    dataset = np.load(dataset_name)
    X = dataset[:, :-3]
    y = dataset[:, -3]
    acc[subs[i]] = []
    f1[subs[i]] = []
    auc[subs[i]] = []
    res = []
    y_test_all = []
    for train_index, test_index in skf.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        elf = VotingClassifier(estimators=[('RF', RandomForestClassifier(n_estimators=20)),
                                           ('ETC', ExtraTreesClassifier(n_estimators=20)),
                                           ('GBC', GradientBoostingClassifier(n_estimators=20)), ('GB', GaussianNB()),
                                           ('DT', DecisionTreeClassifier())
                                           ], n_jobs=-1, voting='soft')

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

        print(subs[i], " Accuracy: ", acc)
        print(subs[i], "F1-Score: ", f1)
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
    f1[subs[i]].append(accuracy_score(y_test_all, new_results))
    print(subs[i], "Accuracy th: ", accuracy_score(y_test_all, new_results))
    print(subs[i], "F1-Score th: ",f1_score(y_test_all, new_results))

# dataset = np.load(r"D:\Physionet Challenge\new_dataset_A.npy")[1:]
#
# new_dataset = []
# for patient in np.unique(dataset[:, -1]):
#     print(patient)
#     patient_features = dataset[np.where(dataset[:, -1] == patient)[0]]
#     new_dataset.append(patient_features)
#
# np.save("dataset_A_patient.npy", np.array(new_dataset))
#
# dataset = np.load("dataset_A_patient.npy")
#
# new_dataset = np.vstack(replace_nan_by_value(zero_dataset))
#
# print(new_dataset.shape)
#
# np.save("dataset_A_zero_subs", new_dataset)
