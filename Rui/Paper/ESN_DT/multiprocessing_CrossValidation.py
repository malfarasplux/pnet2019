from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, roc_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from GroupStratifiedKFold import GroupStratifiedKFold
import multiprocessing
from ESNtools import *
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


def cross_validation(train_index, test_index, X_interp, X, y_interp, patients_id, patients_id_samples, ESN, res,
                     y_test_all, backward_interpolation, acc, f1, auc, return_dict, num, n_estimators, classifier):
    print("\nTRAIN:", train_index, "TEST:", test_index)
    # X_train, X_test = np.nan_to_num(X[train_index]), np.nan_to_num(X[test_index])
    y_train, y_test = y_interp[train_index], y_interp[test_index]
    patients_id_train, patients_id_test = patients_id[train_index], patients_id[test_index]

    print(classifier)
    if classifier == 'GB':
        elf = GradientBoostingClassifier(n_estimators=n_estimators)
    elif classifier == 'RF':
        elf = RandomForestClassifier(n_estimators=n_estimators)
    elif classifier == 'GNB':
        elf = GaussianNB()
    elif classifier == 'DT':
        elf = DecisionTreeClassifier()
    else:
        elf = KNeighborsClassifier(weights='distance', n_jobs=1)

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
        print("\n Regular Running.", flush=True)
        #pred = elf.predict_proba(ESN[test_index])[:, 1]
        results = elf.predict(ESN[test_index])
        pred = elf.predict_proba(ESN[test_index])
        if len(np.shape(pred)) == 2:
            pred = pred[:, 1]

    print("\nFinished Testing.\n Next!", flush=True)

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
def build_ESN(patients_id, X, N, ESN, feedESN, patients_id_samples):
    for id_ in np.unique(patients_id):
        features_patient = X[patients_id_samples[id_]]
        ESN[patients_id_samples[id_], :] = np.array(feedESN(features_patient, N, scale=.0001, mem=1, func=sigmoid, f_arg=1, silent=True))
    return ESN


def build_patients_id_samples(patients_id):
    patients_id_samples = []
    for i, id_ in enumerate(np.unique(patients_id)):
        patients_id_samples.append(np.where(patients_id == id_)[0])
    return patients_id_samples


if __name__ == '__main__':
    processes = []
    backward_interpolation = False

    print("Loading datasets...", flush=True)

    # data = np.nan_to_num(np.load('./Datasets/new_dataset_AB.npy'))
    # dataset = data[:, :40]
    # patients_id = data[:, -1]
    # labels, patients_labels = data[:, -3:-1].T

    dataset = np.nan_to_num(np.load('../../Datasets/training_AB.npy'))
    #dataset_interp = np.zeros(np.shape(dataset))
    dataset_interp = dataset # np.nan_to_num(np.load('./Datasets/training_AB_nanfill.npy'))
    patients_id = np.load('../../Datasets/training_AB_patient.npy')
    labels = np.concatenate(np.load('../../Datasets/training_AB_Y.npy'))
    patients_labels = np.zeros(np.shape(labels))

    for id_ in np.unique(patients_id):
        index = np.where(patients_id==id_)[0]
        if 1 in labels[index]:
            patients_labels[id_] = np.ones(np.shape(labels[id_]))

    print("\nGroup Stratified K Fold...", flush=True)

    train_index, test_index = GroupStratifiedKFold(np.hstack(
        [dataset, labels.reshape(-1, 1), patients_labels.reshape(-1, 1), patients_id.reshape(-1, 1)]), 10)

    X = dataset
    X_interp = dataset_interp[:, :-1]
    y_interp = labels

    ESN = X

    acc, f1, auc, res, y_test_all = [], [], [], [], []

    print("\nBuilding patients_id_samples", flush=True)

    patients_id_samples = build_patients_id_samples(patients_id)

    for ESN_bool in [True]:
        if ESN_bool:
            # Build the Net
            N = 100
            ESN = np.zeros((X_interp.shape[0], N + 1))

            ESN = build_ESN(patients_id, X, N, ESN, feedESN, patients_id_samples)

        print("\nStart Cross Validation...", flush=True)

        n_estimators = 200
        classifiers = ['GB', 'RF', 'kNN']
        return_dict = {}
        classifiers = ['DT']

        for classifier in classifiers:
            for i in range(len(train_index)):
                cross_validation(train_index[i], test_index[i], X_interp, X, y_interp, patients_id, patients_id_samples,
                                ESN, res, y_test_all, backward_interpolation, acc, f1, auc, return_dict, i,
                                n_estimators, classifier)

            for j in range(len(train_index)):
                res.append(return_dict['res' + str(j)])
                y_test_all.append(return_dict['y_test_all' + str(j)])

            res = np.concatenate(res)
            y_test_all = np.concatenate(y_test_all)

            print(res.shape)
            print(y_test_all.shape)

            fpr, tpr, thresholds = roc_curve(y_test_all, res, pos_label=1)

            threshold = 0
            step = 0.001

            res, y_test_all, new_results, accuracy, f1_score_list = threshold_optimization(step, res, y_test_all)
            print(accuracy_score(y_test_all, new_results))
            print(f1_score(y_test_all, new_results))

            new_threshold = np.array(f1_score_list).argmax() * step
            print("\nThreshold:", new_threshold, flush=True)

            new_results = np.zeros(len(res))
            new_results[np.where(res > new_threshold)[0]] = 1
            new_results[np.where(res <= new_threshold)[0]] = 0

            acc.append(accuracy_score(y_test_all, new_results))
            f1.append(f1_score(y_test_all, new_results))

            print("\nAccuracy th: ", accuracy_score(y_test_all, new_results), flush=True)
            print("F1-Score th: ", f1_score(y_test_all, new_results), flush=True)

            Pr = precision_score(y_test_all, new_results)
            Re = recall_score(y_test_all, new_results)
            ACC = accuracy_score(y_test_all, new_results)
            f1 = f1_score(y_test_all, new_results)
            auc = roc_auc_score(y_test_all, res)

            # write to report file
            output_file = './Results/report_' + str(n_estimators) + '_' + classifier + '_'+ str(ESN_bool) + '.txt'
            with open(output_file, 'w') as f:
                f.write(__file__ + '\n')
                f.write(time.strftime("%Y-%m-%d %H:%M") + '\n')
                f.write('Dataset: new_dataset_AB.npy\n')
                f.write('Using ESN: ' + str(ESN_bool) + '\n')
                f.write('(%2.4f) \t threshold\n' % new_threshold)
                f.write('%2.4f \t Pr\n' % Pr)
                f.write('%2.4f \t Re\n' % Re)
                f.write('%2.4f \t F1\n' % f1)
                f.write('%2.4f \t ACC\n' % ACC)
                f.write('%2.4f \t AUC\n' % auc)

            print(time.strftime("%Y-%m-%d %H:%M"))
            print('\nDataset: new_dataset_AB.npy\n')
            print('Using ESN: ' + str(ESN_bool) + '\n')
            print('(%2.4f) \t threshold\n' % new_threshold)
            print('Pr: %2.4f' % Pr)
            print('Re: %2.4f' % Re)
            print('F1: %2.4f' % f1)
            print('ACC: %2.4f' % ACC)
            print('AUC: %2.4f' % auc)
