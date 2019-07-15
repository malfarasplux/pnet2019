import numpy as np
new_dataset = np.load('new_dataset.npy')

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import LeaveOneOut, KFold, ShuffleSplit, LeavePGroupsOut, GroupShuffleSplit,\
    GroupKFold, GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle

x = new_dataset[:, 1:-1]
patients_id = new_dataset[:, 0]
y = new_dataset[:, -1]

# hospA = np.where(patients_id<=20336)[0]

# index = int(0.7*len(x))
# x_train, x_test, y_train, y_test = [x[:index], x[index:], y[:index], y[index:]]

# param_grid = {
    # 'n_estimators': [10, 20, 50, 100],
    # 'max_features': ['auto', 'sqrt', 'log2'],
    # 'max_depth': [4, 6, 8, 10],
    # 'criterion': ['gini', 'entropy']
# }
# random_forest = RandomForestClassifier(n_jobs=1)
# CV_rfc = GridSearchCV(estimator=random_forest, scoring='balanced_accuracy', param_grid=param_grid, verbose=2,
                      # cv=StratifiedKFold(n_splits=10).split(x_train, y_train, patients_id[:index]), n_jobs=1)
# CV_rfc.fit(x_train, y_train)
# parameters = CV_rfc.best_params_


# x_A, y_A = x[:hospA], y[:hospA]
# x_B, y_B = x[hospA:], y[hospA:]
# random_forest = RandomForestClassifier(n_estimators=20, n_jobs=-1)
# random_forest = random_forest.fit(x_B, y_B)
# results = random_forest.predict_proba(x_A)[:, -1]

# fpr, tpr, thresholds = roc_curve(y_A, results, pos_label=1)

# import matplotlib.pylab as plt
# plt.plot(fpr, tpr)

# threshold = 0
# accuracy = []
# f1_score_list = []
# step = 0.001

# for threshold in np.arange(0, 1, step):
    # print(threshold)
    # new_results = np.zeros(len(results))
    # new_results[np.where(results>threshold)[0]] = 1
    # new_results[np.where(results<=threshold)[0]] = 0
    # accuracy.append(accuracy_score(y_A, new_results))
    # f1_score_list.append(f1_score(y_A, new_results))
# print(accuracy_score(y_A, new_results))
# print(f1_score(y_A, new_results))

# plt.plot(f1_score_list)
# new_threshold = np.array(f1_score_list).argmax() * step
# print(new_threshold)

# new_results = np.zeros(len(results))
# new_results[np.where(results>new_threshold)[0]] = 1
# new_results[np.where(results<=new_threshold)[0]] = 0

# print(accuracy_score(y_A, new_results))
# print(f1_score(y_A, new_results))
# print(confusion_matrix(y_A, new_results))


# Used to avoid overfit

# rs = GroupShuffleSplit(n_splits=10, train_size=.9, test_size=.1)
# lpgo = LeavePGroupsOut(n_groups=.1)
# print("# Groups: " + str(lpgo.get_n_splits(x, y, groups=patients_id)))

rs = GroupKFold(n_splits=10)
x_shuffled, y_shuffled, patients_id_shuffled = shuffle(x, y, patients_id, random_state=0)

auc = []
f_score = []
acc = []
res = []
y_test_all = []

for train_index, test_index in rs.split(x_shuffled, y_shuffled, patients_id_shuffled): # lpgo.split(x, y, groups=patients_id):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = x_shuffled[train_index], x_shuffled[test_index]
    y_train, y_test = y_shuffled[train_index], y_shuffled[test_index]
#     scaler = MinMaxScaler()
#     scaler = scaler.fit(X_train)
#     X_train = scaler.transform(X_train)
#     X_test = scaler.transform(X_test)
    
    X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    
    random_forest = RandomForestClassifier(n_estimators=20, class_weight = 'balanced', n_jobs=-2)  # n_estimators=100, n_jobs=-1)
    random_forest = random_forest.fit(X_train, y_train)
    results = random_forest.predict_proba(X_test)[:,1]
    
    #adaboost = AdaBoostClassifier(RandomForestClassifier(n_estimators=10, n_jobs=-1))
    #adaboost = adaboost.fit(X_train, y_train)
    #results = adaboost.predict_proba(X_test)[:,1]
    
#     decisiontree = DecisionTreeClassifier(max_depth=1000, max_leaf_nodes=1000)
#     decisiontree = decisiontree.fit(X_train, y_train)
#     results = decisiontree.predict_proba(X_test)[:,1]
    
#     qda = QuadraticDiscriminantAnalysis()
#     qda = qda.fit(X_train, y_train)
#     results = qda.predict_proba(X_test)[:,1]
    
#     gauss = GaussianNB()
#     gauss = gauss.fit(X_train, y_train)
#     results = gauss.predict_proba(X_test)[:,1]
    
# x_shuffled.shape[1], 2
#     clf = MLPClassifier(solver='adam', activation='logistic', alpha=1e-2, hidden_layer_sizes=(x_shuffled.shape[1], 1), random_state=42)
#     clf = clf.fit(X_train, y_train)
#     results = clf.predict_proba(X_test)[:,1]
    
#     svm_c = svm.OneClassSVM(gamma='auto', verbose=True, max_iter=1)
#     svm_c = svm_c.fit(X_train, y_train)
#     results = svm_c.predict_proba(X_test)[:, 1]
#     print(f1_score(y_test, results, average='micro'))

    res.append(results)
    y_test_all.append(y_test)
    auc.append(roc_auc_score(y_test, results))
#     f_score.append(f1_score(y_test, results))
#     acc.append(accuracy_score(y_test, results))
    print(auc[-1])
    
import joblib
joblib.dump(random_forest, "rf20_noSMOTE.joblib")

print("Finished Validation!")

print("Mean: " + str(100*np.mean(auc)) + " +- " + str(100*np.std(auc)))
print(np.concatenate(res))
res = np.concatenate(res)
y_test_all = np.concatenate(y_test_all)

fpr, tpr, thresholds = roc_curve(y_test_all, res, pos_label=1)

import matplotlib.pylab as plt
plt.plot(fpr, tpr)

threshold = 0
accuracy = []
f1_score_list = []
step = 0.001

for threshold in np.arange(0, 1, step):
    print(threshold)
    new_results = np.zeros(len(res))
    new_results[np.where(res>threshold)[0]] = 1
    new_results[np.where(res<=threshold)[0]] = 0
    accuracy.append(accuracy_score(y_test_all, new_results))
    f1_score_list.append(f1_score(y_test_all, new_results))
print(accuracy_score(y_test_all, new_results))
print(f1_score(y_test_all, new_results))

plt.plot(f1_score_list)
new_threshold = np.array(f1_score_list).argmax() * step
print(new_threshold)

new_results = np.zeros(len(res))
new_results[np.where(res>new_threshold)[0]] = 1
new_results[np.where(res<=new_threshold)[0]] = 0

print(accuracy_score(y_test_all, new_results))
print(f1_score(y_test_all, new_results))
print(confusion_matrix(y_test_all, new_results))

print("RandomForest without SMOTE 100!")