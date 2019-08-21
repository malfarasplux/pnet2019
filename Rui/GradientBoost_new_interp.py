from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, roc_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from iterative_interpolation import *
from GroupStratifiedKFold import GroupStratifiedKFold


backward_interpolation = False

print("Loading datasets...", flush=True)

dataset_interp = np.nan_to_num(np.load('./Datasets/training_setA_nanfill.npy'))
dataset = np.load('./Datasets/training_setA.npy')
patients_id = np.load('./Datasets/training_setA_patient.npy')
labels, patients_labels = np.load('./Datasets/dataset_A_mean_subs.npy').T[-3:-1]

print("Group Stratified K Fold...", flush=True)

train_index, test_index = GroupStratifiedKFold(np.hstack(
    [dataset, labels.reshape(-1, 1), patients_labels.reshape(-1, 1), patients_id.reshape(-1, 1)]), 10)

X = dataset[:, :-1]
X_interp = dataset_interp[:, :-1]
y_interp = labels

acc, f1, auc, res, y_test_all = [], [], [], [], []

patients_id_samples = []
for id in np.unique(patients_id):
    patients_id_samples.append(np.where(patients_id == id)[0])

print("Start Cross Validation...", flush=True)

for i in range(len(train_index)):

    print("TRAIN:", train_index[i], "TEST:", test_index[i])
    X_train, X_test = X_interp[train_index[i]], np.nan_to_num(X[test_index[i]])
    y_train, y_test = y_interp[train_index[i]], y_interp[test_index[i]]
    patients_id_train, patients_id_test = patients_id[train_index[i]], patients_id[test_index[i]]

    print("Build Classifier.", flush=True)

    elf = GradientBoostingClassifier(n_estimators=200)
    # elf = RandomForestClassifier(n_estimators=1, n_jobs=-1)

    print("Start training...", flush=True)

    elf = elf.fit(X_train, y_train)
    print("Start testing...", flush=True)

    aux_pred = []
    aux_result = []
    if backward_interpolation:
        for id in np.unique(patients_id_test):
            patients_features = X[patients_id_samples[id]]
            for h, hour in enumerate(patients_features):
                features = patients_features[:h+1]
                for f in range(features.shape[1]):
                        if np.sum(np.isnan(features[:, f])) < len(features[:, f]):
                            nan_bounds(features[:, f])
                            nan_interpolate(features[:, f])
                        else:
                            features[:, f] = np.nan_to_num(features[:, f], -1)
                pred = elf.predict_proba(features[-1].reshape(1, -1))[:, 1]
                results = elf.predict(features[-1].reshape(1, -1))
                aux_pred.append(pred)
                aux_result.append(results)
    else:
        pred = elf.predict_proba(X_test)[:, 1]
        results = elf.predict(X_test)

    print("Finished Testing.\n Next!", flush=True)
    res.append(pred)
    y_test_all.append(y_test)
    acc.append(accuracy_score(results, y_test))
    f1.append(f1_score(results, y_test))
    auc.append(roc_auc_score(y_test, pred))

    print("Accuracy: ", accuracy_score(results, y_test), flush=True)
    print("F1-Score: ", f1_score(results, y_test), flush=True)
    print("AUC: ", auc, flush=True)

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
print("Threshold:", new_threshold, flush=True)

new_results = np.zeros(len(res))
new_results[np.where(res > new_threshold)[0]] = 1
new_results[np.where(res <= new_threshold)[0]] = 0

acc.append(accuracy_score(y_test_all, new_results))
f1.append(f1_score(y_test_all, new_results))
print("Accuracy th: ", accuracy_score(y_test_all, new_results), flush=True)
print("F1-Score th: ", f1_score(y_test_all, new_results), flush=True)
