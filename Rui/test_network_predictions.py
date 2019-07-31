from ESNtools import *
from pandas import DataFrame, concat
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.preprocessing import MinMaxScaler

dataset = np.load("Datasets/dataset_A_normal_subs_sepsis_late.npy")
# features = dataset[:-6, :-3]
# labels = dataset[6:, :-3]


def timeseries_to_supervised(data, lag=1):
    '''
	Construct the problem as supervised. I did NOT do this function, it was taken from:
	https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
	'''
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

'''
In the rest of the script, I assumed that the dataset was in the form of features, labels per sample, labels per patient, patient ID.
'''

rmse_list = []
for patient_id in np.unique(dataset[:, -1]):
    print(patient_id)
	# Get patient
    patient = dataset[np.where(dataset[:, -1] == patient_id)[0]][:, :-3]
	
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(patient)
    patient = scaler.transform(patient)
    
	# The framing of the problem is done for each patient
	train = timeseries_to_supervised(patient[:, :-3], 6).values
    
	# The labels/target are now future samples. The transformation done by timeseries_to_supervised()
	# allow us to divide the input an output as such:
	X, y = train[:, 0:-patient.shape[1]], train[:, -patient.shape[1]:]
	
	# Build the Net
    ESN = feedESN(X, 200, scale=.001, mem=.1, func=sigmoid, f_arg=10, silent=True)
    w = get_weights_lu_biasedNE(ESN, y)
	
	# The predictions should correspond to a future sample, so, the shape is (1, 40)
    predictions = np.matmul(ESN, w)
	
	# Evaluate the performance.
    rmse = np.sqrt(mean_squared_error(y, predictions))
    print("RMSE: ", rmse)
    rmse_list.append(rmse)


# I don't remember why I did this...
sepsis_label = []
for patient_id in np.unique(dataset[:, -1]):
    print(patient_id)
    sepsis_label.append(dataset[np.where(dataset[:, -1] == patient_id)[0]][:, -2][0])


# This was a tentative based on the comparison of what was expected and what we got.
# The threshold was used to optimise the f1-score.
f1_list = []
new_rmse_list = rmse_list
for value in np.arange(0, 100, .001):
    print(value)
    new_rmse_list[np.where(rmse_list <= np.percentile(rmse_list, value))[0]] = 0
    new_rmse_list[np.where(rmse_list > np.percentile(rmse_list, value))[0]] = 1
    f1_list.append(f1_score(sepsis_label, new_rmse_list))


print("Mean RMSE: ", np.mean(rmse_list))
