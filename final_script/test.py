import numpy as np
import joblib
import sys


def get_sepsis_score(patient):

    threshold = 0.139
    randomforest = joblib.load('rf100_noSMOTE.joblib')

    scores = randomforest.predict_proba(patient[:, :-1])[:,-1]

    results = np.zeros(len(scores))
    results[np.where(scores > threshold)[0]] = 1
    results[np.where(scores <= threshold)[0]] = 0

    return scores.flatten(), results.flatten()


def read_challenge_data(input_file):
    with open(input_file) as file:
        f = np.nan_to_num(np.loadtxt(file, delimiter='|', skiprows=1))
    return f


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('Usage: %s input[.psv]' % sys.argv[0])

    record_name = sys.argv[1]
    if record_name.endswith('.psv'):
        record_name = record_name[:-4]

    # read input data
    input_file = record_name + '.psv'
    patient = read_challenge_data(input_file)

    # generate predictions
    (scores, labels) = get_sepsis_score(patient)

    # write predictions to output file
    output_file = record_name + '.out'
    with open(output_file, 'w') as f:
        for (s, l) in zip(scores, labels):
            f.write('%g|%d\n' % (s, l))
