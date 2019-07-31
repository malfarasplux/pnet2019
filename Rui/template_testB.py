import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score


model = pickle.load( open( "grid_search_object.p", "rb" ) )

best_model = model.best_estimator_
best_parameters = model.best_params_

# Change the path to the file of the interpolated features of Hospital B
features = np.load(path_to_HospitalB)
# Change the path to the file of the targets of Hospital B
labels = np.load(path_to_HospitalB)

try:
    threshold = np.loadtxt("threshold.csv")[0]
except:
    threshold = pickle.load(open("threshold.p", "rb"))

predictions = best_model.predict_proba(features)[:, 1]

results = predictions.copy()
results[np.where(results > threshold)[0]] = 1
results[np.where(results <= threshold)[0]] = 0

print("Accuracy: ", accuracy_score(results, labels))
print("F1-Score: ", f1_score(results, labels))
print("AUC: ", roc_auc_score(predictions, labels))
print("Precision: ", precision_score(results, labels))
print("Recall: ", recall_score(results, labels))
