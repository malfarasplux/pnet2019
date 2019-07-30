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
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pylab as plt


dataset = np.load("Datasets/dataset_A_normal_subs_sepsis_late.npy")
# variance = .9
# sel = VarianceThreshold(threshold=(variance * (1 - variance)))
# dataset = np.hstack([sel.fit_transform(dataset[:, :-3]), dataset[:, -3:]])

print("Shape of the new dataset: ", dataset.shape)

X = dataset[:, :-3]
y = dataset[:, -3]


def compute_correlations(dataset):
    correlations_final = []
    for i, feature in enumerate(dataset.T):
        print(i)
        correlations = []
        for other_feature in dataset.T:
            correlations.append(pearsonr(feature, other_feature)[0])
        correlations_final.append(correlations)
    return correlations_final


def compute_correlations_to_target(dataset):
    correlations_final = []
    for i, feature in enumerate(dataset.T):
        correlations = pearsonr(feature, dataset[:, -3])[0]
        correlations_final.append(correlations)
    return correlations_final
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaler = scaler.fit(X)
# X = scaler.transform(X)


def forward_selection(dataset, X, y):
    model = GroupShuffleSplit(1)
    np.array(list(model.split(X, y, dataset[:, -1])))
    index_train, index_test = np.array(list(model.split(X, y, dataset[:, -1])))[0]

    indexes = [16]
    all_feature_results = []
    for i in range(40):
        new_X = X[:, indexes]
        features_results = []

        for j, feature in enumerate(X.T):
            print(indexes)
            if j not in indexes:
                if len(indexes) > 0:
                    feature = np.vstack([new_X.T, feature]).T
                    # feature = feature.reshape(-1, 2)
                    X_train, y_train = feature[index_train], y[index_train]
                    X_test, y_test = feature[index_test], y[index_test]
                    print(X_train.shape)

                    elf = VotingClassifier(estimators=[('RF', RandomForestClassifier(n_estimators=10)),
                                                        ('ETC', ExtraTreesClassifier(n_estimators=10)),
                                                        ('GBC', GradientBoostingClassifier(n_estimators=10)), ('GB', GaussianNB()),
                                                        ('DT', DecisionTreeClassifier())
                                                        ], n_jobs=-1, voting='hard')

                    # elf = GaussianNB()
                    # elf = OneClassSVM(gamma='auto')
                    # elf = RandomForestClassifier(n_estimators=20)
                    # elf = AdaBoostClassifier(base_estimator=GaussianNB(), n_estimators=1000)
                    # elf = ExtraTreesClassifier(n_estimators=20)
                    # elf = GradientBoostingClassifier(n_estimators=20)
                    # elf = MLPClassifier()

                    # n_estimators = 10
                    # elf = OneVsRestClassifier(
                    #     BaggingClassifier(SVC(kernel='linear', probability=True,
                    #     class_weight={0: 100, 1: .1}), max_samples=1.0 / n_estimators,
                    #     n_estimators=n_estimators, n_jobs=-1))
                    print("Start training...")

                    elf = elf.fit(X_train, y_train)
                    print("Start testing...")
                    results = elf.predict(X_test)
                    f1 = f1_score(y_test, results)
                    features_results.append(f1)

                    print("F1-Score th: ", f1)
                else:
                    features_results.append(-1)
        if len(features_results) > 0:
            indexes.append(features_results.index(np.max(features_results)))
            all_feature_results.append(features_results)
    return indexes, all_feature_results


def backward_elimination(dataset, X, y):
    model = GroupShuffleSplit(1)
    index_train, index_test = np.array(list(model.split(X, y, dataset[:, -1])))[0]

    indexes = [np.arange(40)]
    all_feature_results = []

    for i in range(40):
        if len(indexes) > 0:
            new_X = X[:, indexes]
            features_results = []

            for j, feature in enumerate(X.T):
                feature = np.vstack(np.vstack(new_X[:, np.delete(indexes, [j])]))
                # feature = feature.reshape(-1, 2)
                X_train, y_train = feature[index_train], y[index_train]
                X_test, y_test = feature[index_test], y[index_test]
                print(X_train.shape)
                print(X_train)

                # elf = VotingClassifier(estimators=[('RF', RandomForestClassifier(n_estimators=10)),
                #                                    ('ETC', ExtraTreesClassifier(n_estimators=10)),
                #                                    ('GBC', GradientBoostingClassifier(n_estimators=10)),
                #                                    ('GB', GaussianNB()),
                #                                    ('DT', DecisionTreeClassifier())
                #                                    ], n_jobs=-1, voting='hard')

                # elf = GaussianNB()
                # elf = OneClassSVM(gamma='auto')
                elf = RandomForestClassifier(n_estimators=20)
                # elf = AdaBoostClassifier(base_estimator=GaussianNB(), n_estimators=1000)
                # elf = ExtraTreesClassifier(n_estimators=20)
                # elf = GradientBoostingClassifier(n_estimators=20)
                # elf = MLPClassifier()

                # n_estimators = 10
                # elf = OneVsRestClassifier(
                #     BaggingClassifier(SVC(kernel='linear', probability=True,
                #     class_weight={0: 100, 1: .1}), max_samples=1.0 / n_estimators,
                #     n_estimators=n_estimators, n_jobs=-1))
                print("Start training...")

                elf = elf.fit(X_train, y_train)
                print("Start testing...")
                results = elf.predict(X_test)
                f1 = f1_score(y_test, results)
                features_results.append(f1)

                print("F1-Score th: ", f1)
            indexes = np.delete(indexes(np.max(features_results)))
            all_feature_results.append(features_results)
    return indexes, all_feature_results
