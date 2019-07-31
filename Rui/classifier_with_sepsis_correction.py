import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, GroupShuffleSplit, ShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier, BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC, OneClassSVM, SVC
from sklearn.neural_network import MLPClassifier
from scipy import linalg
from imblearn.over_sampling import SMOTE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr


def replace_nan_by_value(data, value=None):
    '''
	Function to replace the NaNs in data for numerical values. Data should have 3 dimensions instead of 2.
	The first is patient, so, patient = data[i]. The second is the sample, so, sample = patient[j].
	The third is feature_value, so, feature_value = sample[n].
	
	:params:
	    data: (list) or (numpy.array)
		    Dataset as described above.
		value: (str) or None
		    The value to substitute. If value is 'mean', nan are replaced by the mean value of the column of each patient.
			If value is 'normal', nan are replaced by a random value with a normal distribution with the mean value of 
			the column of the patient and variance of the same column. Else, nan are replaced by 0.
	
	:return:
	    data_copy: (numpy.array) or (list)
		    Copy of input data with replaced nan.
	'''
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
	'''
	Function to change the labels of the dataset. The dataset is expected to be in the form of: features,
	labels per sample, labels per patient, patient ID. The labels are delayed for each patient by "hours_to_onset".
	
	:params:
	    dataset: (numpy.array)
		    Dataset in the form described above.
		hours_to_onset: (int)
		    Hours to delay the label. For example, if the original labels are:
			[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
			and hours_to_onset=6, then, the result will be:
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
	
	:return:
	    dataset: (numpy.array)
		    Dataset containing the delayed labels where the original ones were.
	'''
    flag = False
    i = 0
    while i in range(len(dataset[:-hours_to_onset, -3])):
        print(i)
        if dataset[i+hours_to_onset, -3] == 1 and dataset[i, -3] == 1 and not flag:
            flag = True
            j = i
            while j < i+hours_to_onset:
                dataset[j, -3] = 0
                j += 1
            i = j
        if dataset[i, -3] == 0:
            flag = False
        i += 1
    return dataset


def mean_wave(segments):
    '''
	Calculate the mean wave of segments.
	'''
    organized_segment = segments.copy().T
    mean_wave = [np.mean(seg) for seg in organized_segment]

    return np.array(mean_wave)

def compute_distance_to_mean_wave_patient(dataset):
    '''
	Compute the mean wave of each patient and then calculate the distance (in this case cosine_similarity) 
	between each sample and the mean wave. You can change the used metric in the for below by uncommenting one
	of the other lines.
	
	(It is only using the identified by eye best features)
	
	:params:
	    dataset: (numpy.array)
		    dataset is expected to be in the form of: features,
			labels per sample, labels per patient, patient ID.
			
	:return:
	    comparison_matrix: (list)
		    Matrix containing the results of the comparison between each sample of each patient and the
			corresponding mean wave.
	'''
    comparison_matrix = []
    index = [0, 8, 14, 15, 20, 26, 27, 32, -3, -2, -1]

    for patient_id in np.unique(dataset[:, -1]):
        print(patient_id)
        patient = dataset[np.where(dataset[:, -1] == patient_id)[0]] # [:, :-3]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(patient)
        patient = scaler.transform(patient)
        patient = patient[:, index][:, :-3]
        # aux_mean_wave = mean_wave(patient[:, :-6])
        aux_mean_wave = mean_wave(patient)
        for line in patient:
            # comparison_matrix.append(linalg.norm(line[:, :-6]-aux_mean_wave))
            # comparison_matrix.append(linalg.norm(line[:-3] - aux_mean_wave))
            # comparison_matrix.append(pearsonr(line, aux_mean_wave))
            comparison_matrix.append(cosine_similarity(line.reshape(1, -1), aux_mean_wave.reshape(1, -1)))
            # comparison_matrix.append(np.abs(line[:, :-6]-aux_mean_wave))
    return comparison_matrix


from numba import jit

@jit()
def compute_distance_to_mean_wave(dataset):
    '''
	Compute the mean wave of all patients and then calculate the distance between each sample and the
	mean wave. You can change the used metric in the for below by uncommenting one of the other lines.
	
	(It is only using the identified by eye best features)

	:params:
	    dataset: (numpy.array)
		    dataset is expected to be in the form of: features,
			labels per sample, labels per patient, patient ID.
			
	:return:
	    comparison_matrix: (list)
		    Matrix containing the results of the comparison between each sample of each patient and the
			corresponding mean wave.
	'''
    index = [0, 8, 14, 15, 20, 26, 27, 32, -3, -2, -1]
    comparison_matrix = []
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler = scaler.fit(dataset)
    # dataset = scaler.transform(dataset)
    aux_mean_wave = mean_wave(dataset[:, index][np.where(dataset[:, -2] == 0)[0]][:, :-3])
    for i, line in enumerate(dataset[:, index][:, :-3]):
        print(i)
        comparison_matrix.append(linalg.norm(line - aux_mean_wave))
        # comparison_matrix.append(pearsonr(line, aux_mean_wave))
        # comparison_matrix.append(cosine_similarity(line[:-9].reshape(1, -1), aux_mean_wave.reshape(1, -1)))
        # comparison_matrix.append(np.abs(line[:, :-6]-aux_mean_wave))
    return comparison_matrix


def get_patients_by_health(dataset):
    '''
	Function to divide the patients in sepsis and healthy. This can be used to get a balanced dataset
	in terms of healthy/unhealthy patient by using healthy_patients = healthy_patients[len((sepsis_patients)].
	
	:params:
	    dataset: (numpy.array)
		    dataset is expected to be in the form of: features,
			labels per sample, labels per patient, patient ID.
			
	:return:
	    sepsis: (numpy.array)
		    dataset containing only the patients that present sepsis at some time.
		healthy: (numpy.array)
		    dataset containing only the patients that do not present sepsis at any time.
	'''
    healthy = []
    sepsis = []
    for patient_id in np.unique(dataset[:, -1]):
        print(patient_id)
        patient = dataset[np.where(dataset[:, -1] == patient_id)[0]]
        if np.sum(patient[:, -2]) == 0:
            healthy.append(patient)
        else:
            sepsis.append(patient)

    healthy = np.vstack(healthy)
    sepsis = np.vstack(sepsis)
    print(np.shape(healthy))

    # dataset = np.vstack([sepsis, healthy])
    # dataset = healthy
    return sepsis, healthy  # dataset


def block_hour(dataset, hours=4):
    '''
	Function to build the dataset with blocks of hours of acquisition.
	The blocks are built for each patient, so, there are no blocks consisting of various patients.
	The number of columns of the returned dataset will be number_features * hours.
	The number of lines will be number_lines_dataset - (num_patients * (hours - 1))
	
	:params:
	    dataset: (numpy.array)
            dataset is expected to be in the form of: features,
			labels per sample, labels per patient, patient ID.
	
	:return:
	    new_dataset: (numpy.array)
		    The new set of features containing the blocks of hours.
		    The labels of the newly constructed dataset are as follows:
			A block is considered positive (1) if any hour of the block has the positive label (1).
	'''
    new_dataset = []
    for patient_id in np.unique(dataset[:, -1]):
        print("Patient ID: ", patient_id)
        patient_index = np.where(dataset[:, -1] == patient_id)[0]
        patient = dataset[patient_index]
        for i in range(len(patient) - hours):
            print(i)
            if 1 in patient[i:i + hours, -3]:
                num = 1
            else:
                num = 0
            new_dataset.append(np.concatenate([np.concatenate(patient[i:i + hours, :-3]), [num, num, num]]))
    return np.array(new_dataset)


# dataset = get_patients_by_health(dataset)
# np.save('healthy_patients_normal_subs_A_sepsis_late', dataset)
# dataset = np.load('sepsis_patients_normal_subs_A_sepsis_late.npy')
dataset = np.load("Datasets/dataset_A_normal_subs.npy")
patients_id = dataset[:, -1]
# indexes = [16, 38, 37, 36, 35]
indexes = [0, 8, 14, 15, 20, 26, 27, 32, -3, -2, -1]
dataset = block_hour(dataset, hours=5)
# dataset = dataset[:, index]
print(dataset.shape)

# datasets = ["Datasets/dataset_A_mean_subs_sepsis_late.npy", "Datasets/dataset_A_normal_subs_sepsis_late.npy",
#             "Datasets/dataset_A_zero_subs_sepsis_late.npy"]
subs = ['MEAN', 'NORMAL', 'ZERO']
i = 0
acc = {}
f1 = {}
auc = {}
skf = StratifiedKFold(n_splits=10)
# for i, dataset_name in enumerate(datasets):
#     dataset = np.load(dataset_name)
X = dataset[:, :-3]

# scaler = MinMaxScaler(feature_range=(0, 1))
# scaler = scaler.fit(X)
# X = scaler.transform(X)
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
    patients_id_train, patients_id_test = patients_id[train_index], patients_id[test_index]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(X)
    X = scaler.transform(X)

    # X_train, y_train = SMOTE().fit_resample(X_train, y_train)

    # elf = VotingClassifier(estimators=[('RF', RandomForestClassifier(n_estimators=20)),
    #                                    ('ETC', ExtraTreesClassifier(n_estimators=20)),
    #                                    ('GBC', GradientBoostingClassifier(n_estimators=20)), ('GB', GaussianNB()),
    #                                    ('DT', DecisionTreeClassifier())
    #                                    ], n_jobs=-1, voting='soft')

    # elf = GaussianNB()
    # elf = OneClassSVM(gamma='auto')
    elf = RandomForestClassifier(n_estimators=20, class_weight='balanced')
    # elf = AdaBoostClassifier(base_estimator=GaussianNB(), n_estimators=1000)
    # elf = ExtraTreesClassifier(n_estimators=20)
    # elf = GradientBoostingClassifier(n_estimators=20)
    # elf = MLPClassifier()

    # n_estimators = 10
    # elf = OneVsRestClassifier(
    #     BaggingClassifier(SVC(kernel='linear', probability=True, class_weight={0: 100, 1: .1}), max_samples=1.0 / n_estimators,
    #                       n_estimators=n_estimators, n_jobs=-1))
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
    # for r, value in results:
    #     if value == -1:
    #         results[r] = 1
    #     else:
    #         results[r] = 0
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
