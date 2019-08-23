from sklearn.model_selection import cross_val_score
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

scores = cross_val_score(GradientBoostingClassifier(n_estimators=200), X=X, y=y_interp, groups=patients_id,
            scoring='f1', cv=(train_index, test_index), n_jobs=-3, verbose=2, pre_dispatch=4)

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
