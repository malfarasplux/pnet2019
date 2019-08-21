from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, roc_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from iterative_interpolation import *
from GroupStratifiedKFold import GroupStratifiedKFold
import multiprocessing
from ESNtools import *
from numba import njit


def cross_validation(train_index, test_index, X_interp, X, y_interp, patients_id, patients_id_samples, ESN, res,
                     y_test_all, backward_interpolation, acc, f1, auc, return_dict, num):
    print("TRAIN:", train_index, "TEST:", test_index)
    # X_train, X_test = np.nan_to_num(X[train_index]), np.nan_to_num(X[test_index])
    y_train, y_test = y_interp[train_index], y_interp[test_index]
    patients_id_train, patients_id_test = patients_id[train_index], patients_id[test_index]

    elf = GradientBoostingClassifier(n_estimators=200)
    # elf = RandomForestClassifier(n_estimators=1, n_jobs=-1)

    print("Start training...", flush=True)

    elf = elf.fit(ESN[train_index], y_train)
    print("Start testing...", flush=True)

    aux_pred = []
    aux_result = []
    if backward_interpolation:
        print("Backwards Interpolation Running.", flush=True)
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
                pred = elf.predict_proba(features[-1].reshape(1, -1))[:, 1]
                results = elf.predict(features[-1].reshape(1, -1))
                aux_pred.append(pred)
                aux_result.append(results)
    else:
        print("Regular Running.", flush=True)
        pred = elf.predict_proba(ESN[test_index])[:, 1]
        results = elf.predict(ESN[test_index])

    print("Finished Testing.\n Next!", flush=True)

    res.append(pred)
    y_test_all.append(y_test)

    acc.append(accuracy_score(results, y_test))
    f1.append(f1_score(results, y_test))
    auc.append(roc_auc_score(y_test, pred))

    print("Accuracy: ", accuracy_score(results, y_test), flush=True)
    print("F1-Score: ", f1_score(results, y_test), flush=True)
    print("AUC: ", auc, flush=True)

    return_dict['res' + str(num)] = pred
    return_dict['y_test_all' + str(num)] = y_test


# @jit(debug=True)
def threshold_optimization(step, res, y_test_all):
    accuracy, f1_score_list = [], []
    for threshold in np.arange(start=0, stop=1, step=step):
        print(threshold)
        new_results = np.zeros(len(res))
        new_results[np.where(res > threshold)[0]] = 1
        new_results[np.where(res <= threshold)[0]] = 0
        accuracy.append(accuracy_score(y_test_all, new_results))
        f1_score_list.append(f1_score(y_test_all, new_results))
    return res, y_test_all, new_results, accuracy, f1_score_list


# @njit(debug=True)
def build_ESN(patients_id, X, N, ESN, feedESN):
    patients_id_samples = []
    for id_ in np.unique(patients_id):
        patients_id_samples.append(np.where(patients_id == id_)[0])
        features_patient = X[patients_id_samples[id_]]
        ESN[patients_id_samples[id_], :] = 0
        np.array(feedESN(features_patient, N, scale=.001, mem=.1, func=sigmoid, f_arg=10,
                                                  silent=True))
    return patients_id_samples, ESN


if __name__ == '__main__':
    processes = []
    backward_interpolation = False

    print("Loading datasets...", flush=True)

    dataset_interp = np.nan_to_num(np.load('./Datasets/training_setA_nanfill.npy'))
    dataset = np.load('./Datasets/training_setA.npy')
    patients_id = np.load('./Datasets/training_setA_patient.npy')
    labels, patients_labels = np.load('./Datasets/dataset_A_mean_subs.npy').T[-3:-1]

    print("Group Stratified K Fold...", flush=True)

    train_index, test_index = GroupStratifiedKFold(np.hstack(
        [dataset, labels.reshape(-1, 1), patients_labels.reshape(-1, 1), patients_id.reshape(-1, 1)]), 2)

    X = np.nan_to_num(dataset[:, :-1])
    X_interp = dataset_interp[:, :-1]
    y_interp = labels

    acc, f1, auc, res, y_test_all = [], [], [], [], []

    # Build the Net
    N = 100
    ESN = np.zeros((X_interp.shape[0], N + 1))

    patients_id_samples, ESN = build_ESN(patients_id, X, N, ESN, feedESN)

    print("Start Cross Validation...", flush=True)

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    for i in range(len(train_index)):
        # cross_validation(train_index[i], test_index[i], X_interp, X, y_interp,
        #                   patients_id, patients_id_samples, ESN, res, y_test_all)
        p = multiprocessing.Process(
            target=cross_validation,
            args=(train_index[i], test_index[i], X_interp, X, y_interp, patients_id, patients_id_samples, ESN, res,
                y_test_all, backward_interpolation, acc, f1, auc, return_dict, i))

        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    for j in range(len(train_index)):
        res.append(return_dict['res' + str(j)])
        y_test_all.append(return_dict['y_test_all' + str(j)])

    res = np.concatenate(res)
    y_test_all = np.concatenate(y_test_all)

    print(res.shape)
    print(y_test_all.shape)

    fpr, tpr, thresholds = roc_curve(y_test_all, res, pos_label=1)

    threshold = 0
    accuracy = []
    f1_score_list = []
    step = 0.001

    res, y_test_all, new_results, accuracy, f1_score_list = threshold_optimization(step, res, y_test_all)
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
