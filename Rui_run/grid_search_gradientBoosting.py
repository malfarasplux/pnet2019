import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve


parameters = {
    "loss":["deviance", "exponential"],
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth":[3, 5, 8, None],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse",  "mae"],
    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators": [10, 20, 100, 200, 500, 1000]
    }


parameters_ = {
    "loss":["deviance", "exponential"],
    "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 4),
    "min_samples_leaf": np.linspace(0.1, 0.5, 4),
    "max_depth":[3, 5, 8, None],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse",  "mae"],
    "subsample":[0.5, .75, 0.8, 0.9,  1.0],
    "n_estimators": [10, 20, 100, 200]
    }

features = np.nan_to_num(np.load('../npy/training_setA_nanfill_mm.npy'))
labels = np.load('../npy/training_setA_Y.npy')

clf = GridSearchCV(GradientBoostingClassifier(), parameters_, cv=10, n_jobs=-1, verbose=2, scoring="f1")

print("Begin training...")
clf = clf.fit(features, labels)
print("Done training!")

import pickle

print("Begin dumping to file...")
pickle.dump(clf, open("grid_search_object_lessParam.p", "wb"))
print("Finish")
