from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, roc_curve
from ESNtools import *
from iterative_interpolation import nan_bounds, nan_interpolate
from GroupStratifiedKFold import GroupStratifiedKFold


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

print("Build Classifier.", flush=True)

# Build the Net
N = 100
ESN = np.zeros((X_interp.shape[0], N+1))

print("Start Cross Validation...", flush=True)

patients_id_samples = []
for id in np.unique(patients_id):
    patients_id_samples.append(np.where(patients_id == id)[0])
    features_patient = X_interp[patients_id_samples[id]]
    ESN[patients_id_samples[id], :] = feedESN(features_patient, N, scale=.001, mem=.1, func=sigmoid, f_arg=10, silent=False)

for i in range(len(train_index)):

    print("TRAIN:", train_index[i], "TEST:", test_index[i], flush=True)
    X_train, X_test = ESN[train_index[i]], X[test_index[i]]
    y_train, y_test = y_interp[train_index[i]], y_interp[test_index[i]]
    patients_id_train, patients_id_test = patients_id[train_index[i]], patients_id[test_index[i]]

    print("Start training...", flush=True)

    w = get_weights_lu_biasedNE(X_train, y_train)

    print("Start testing...", flush=True)

    aux = []
    for id_ in np.unique(patients_id_test):
        patients_features = X[patients_id_samples[id_]]
        for h, hour in enumerate(patients_features):
            features = patients_features[:h+1]
            for f in range(features.shape[1]):
                    if np.sum(np.isnan(features[:, f])) < len(features[:, f]):
                        nan_bounds(features[:, f])
                        nan_interpolate(features[:, f])
                    else:
                        features[:, f] = np.nan_to_num(features[:, f], -1)
            ESN_test = feedESN(features, 100, scale=.001, mem=.1, func=sigmoid, f_arg=10, silent=True)[-1]
            pred = np.matmul(ESN_test, w)
            aux.append(pred)

    print(np.shape(aux))
    print("Finished Testing.\n Next!", flush=True)
    res.append(aux)
    y_test_all.append(y_test)
    auc.append(roc_auc_score(y_test, aux))

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
