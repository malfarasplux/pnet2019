from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, roc_curve
from ESNtools import *
from iterative_interpolation import *
from GroupStratifiedKFold import GroupStratifiedKFold


print("Loading datasets...")

dataset_interp = np.nan_to_num(np.load('./Datasets/training_setA_nanfill.npy'))
dataset = np.load('./Datasets/training_setA.npy')
patients_id = np.load('./Datasets/training_setA_patient.npy')
patients_labels = np.load(r'.\Datasets\dataset_A_mean_subs.npy')[:, -2]

print("Group Stratified K Fold...")

train_index, test_index = GroupStratifiedKFold(np.hstack(
    [dataset, patients_labels.reshape(-1, 1), patients_id.reshape(-1, 1)]), 10)

X = dataset[:, :-1]
X_interp = dataset_interp[:, :-1]
y_interp = dataset_interp[:, -1]

acc, f1, auc, res, y_test_all = [], [], [], [], []

print("Start Cross Validation...")

for i in range(len(train_index)):

    print("TRAIN:", train_index[i], "TEST:", test_index[i])
    X_train, X_test = X_interp[train_index[i]], X[test_index[i]]
    y_train, y_test = y_interp[train_index[i]], y_interp[test_index[i]]
    patients_id_train, patients_id_test = patients_id[train_index[i]], patients_id[test_index[i]]

    print("Build Classifier.")

    # Build the Net
    ESN = feedESN(X_train, 100, scale=.001, mem=.1, func=sigmoid, f_arg=10, silent=True)

    print("Start training...")

    w = get_weights_lu_biasedNE(ESN, y_train)

    print("Start testing...")

    for id in np.unique(patients_id_test):
        patients_features = X_test[np.where(patients_id_test == id)[0]]
        for h, hour in enumerate(patients_features):
            features = patients_features[:h]
            for f in range(features.shape[1]):
                if h > 1:
                    if np.sum(np.isnan(features[:, f])) < len(features[:, f]):
                        nan_bounds(features[:, f])
                        nan_interpolate(features[:, f])
                    else:
                        features[:, f] = np.nan_to_num(features[:, f], -1)
            ESN_test = feedESN(features, 100, scale=.001, mem=.1, func=sigmoid, f_arg=10, silent=True)
            pred = np.matmul(ESN_test, w)

    print("Finished Testing.\n Next!")
    res.append(pred)
    y_test_all.append(y_test)
    auc.append(roc_auc_score(y_test, pred))

    print("AUC: ", auc)

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
print("Threshold:", new_threshold)

new_results = np.zeros(len(res))
new_results[np.where(res > new_threshold)[0]] = 1
new_results[np.where(res <= new_threshold)[0]] = 0

acc.append(accuracy_score(y_test_all, new_results))
f1.append(f1_score(y_test_all, new_results))
print("Accuracy th: ", accuracy_score(y_test_all, new_results))
print("F1-Score th: ", f1_score(y_test_all, new_results))
