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


'''
Load the dataset. The part in the comments can be used to select features based on the variance 
(check the sklearn site for more information). The results were not satisfactory.
'''
dataset = np.load("Datasets/dataset_A_normal_subs_sepsis_late.npy")
# variance = .9
# sel = VarianceThreshold(threshold=(variance * (1 - variance)))
# dataset = np.hstack([sel.fit_transform(dataset[:, :-3]), dataset[:, -3:]])

print("Shape of the new dataset: ", dataset.shape)

X = dataset[:, :-3]
y = dataset[:, -3]


def compute_correlations(dataset):
    '''
	Function to calculate the correlation between every pair of features in dataset.
	It returns a 2D array to check those correlations (can be shown using imshow).
	It can also be used to check the correlation between each feature and the labels 
	if you only pay atention to the correct column/line of the matrix.
	
	:params:
	    dataset: (numpy.array)
		    dataset containing whatever features you want to compare. The features 
			should be in the columns, while the lines correspond to different samples.
			(You can just pass the whole dataset (with labels and all) as input.
	
	:return:
	    correlations_final: (numpy.array)
		    The matrix containing the pearson correlation values of pairs of columns 
			of dataset.
	'''
    correlations_final = []
    for i, feature in enumerate(dataset.T):
        print(i)
        correlations = []
        for other_feature in dataset.T:
            correlations.append(pearsonr(feature, other_feature)[0])
        correlations_final.append(correlations)
    return np.array(correlations_final)


def compute_correlations_to_target(feature_matrix, target):
    '''
	Function to calculate the correlation between the features and the target.
	It returns a 1D array to check those correlations.
	
	:params:
	    feature_matrix: (numpy.array)
		    dataset containing whatever features you want to compare. The features 
			should be in the columns, while the lines correspond to different samples.
	
	:return:
	    correlations_final: (numpy.array)
		    The array containing the pearson correlation values of columns and target.
	'''
    correlations_final = []
    for i, feature in enumerate(feature_matrix.T):
        correlations = pearsonr(feature, target)[0]
        correlations_final.append(correlations)
    return numpy.array(correlations_final)


def forward_selection(patient_id, X, y):
    '''
	Function for feature selection (it takes a long time to run). It performs classification for each feature.
	Then, keeps the best performing feature (based on the f1 score) and reruns appending each of the remaining features.
	Then keeps the best performing set of features and does this iteratively until it runs out of features.
	
	The function returns the set of indexes of the features in order of "quality". So, if you want to use the best 5 features, 
	you just need to use indexes, results = forward_selection(patient_id, features, labels); chosen_indexes = indexes[:5].
	
	:params:
	    patient_id: (numpy.array)
		    Patient ID to use in the GroupShuffleSplit method.
		X: (numpy.array)
		    Feature matrix.
	    y: (numpy.array)
		    Target of classification.
			
	:returns:
	    indexes: (numpy.array)
		    Indexes of the features in order of quality (in a cummulative fashion).
		all_feature_results: (list)
		    Results of F1-Score for the set of features. 
			The results of all_feature_results[2] correspond to the results of features[indexes[:1]] + each of the remaining features.
	'''
    
    model = GroupShuffleSplit(1)
    index_train, index_test = np.array(list(model.split(X, y, patient_id)))[0]

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

                    # elf = VotingClassifier(estimators=[('RF', RandomForestClassifier(n_estimators=10)),
                    #                                     ('ETC', ExtraTreesClassifier(n_estimators=10)),
                    #                                     ('GBC', GradientBoostingClassifier(n_estimators=10)), ('GB', GaussianNB()),
                    #                                     ('DT', DecisionTreeClassifier())
                    #                                     ], n_jobs=-1, voting='hard')

                    # elf = GaussianNB()
                    # elf = OneClassSVM(gamma='auto')
                    # elf = RandomForestClassifier(n_estimators=20)
                    # elf = AdaBoostClassifier(base_estimator=GaussianNB(), n_estimators=1000)
                    # elf = ExtraTreesClassifier(n_estimators=20)
                    elf = GradientBoostingClassifier(n_estimators=20)
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
    return np.array(indexes), all_feature_results


def backward_elimination(dataset, X, y):
    # TODO: This is not working!
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
                X_train, y_train = feature[index_train], y[index_train]
                X_test, y_test = feature[index_test], y[index_test]
                print(X_train.shape)
                print(X_train)

                elf = GradientBoostingClassifier(n_estimators=20)

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
