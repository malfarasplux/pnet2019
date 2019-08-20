import numpy as np


def nan_bounds(feats):
    nanidx = np.where(np.isnan(feats))[0]
    pointer_left = 0
    pointer_right = len(feats) - 1
    fix_left = pointer_left in nanidx
    fix_right = pointer_right in nanidx
    while fix_left:
        if pointer_left in nanidx:
            pointer_left += 1
            # print("pointer_left:", pointer_left)
        else:
            val_left = feats[pointer_left]
            feats[:pointer_left] = val_left * np.ones((1, pointer_left), dtype=np.float)
            fix_left = False

    while fix_right:
        if pointer_right in nanidx:
            pointer_right -= 1
            # print("pointer_right:", pointer_right)
        else:
            val_right = feats[pointer_right]
            feats[pointer_right + 1:] = val_right * np.ones((1, len(feats) - pointer_right - 1), dtype=np.float)
            fix_right = False

        # nan interpolation


def nan_interpolate(feats):
    nanidx = np.where(np.isnan(feats))[0]
    nan_remain = len(nanidx)
    nanid = 0
    while nan_remain > 0:
        nanpos = nanidx[nanid]
        nanval = feats[nanpos - 1]
        nan_remain -= 1

        nandim = 1
        initpos = nanpos

        # Check whether it extends
        while nanpos + 1 in nanidx:
            nanpos += 1
            nanid += 1
            nan_remain -= 1
            nandim += 1
            # Average sides
            if np.isfinite(feats[nanpos + 1]):
                nanval = 0.5 * (nanval + feats[nanpos + 1])

        # Single value average
        if nandim == 1:
            nanval = 0.5 * (nanval + feats[nanpos + 1])
        feats[initpos:initpos + nandim] = nanval * np.ones((1, nandim), dtype=np.double)
        nanpos += 1
        nanid += 1


def test_interpolation():
    dataset = np.load('./Datasets/training_setA.npy')
    patients = np.load('./Datasets/training_setA_patient.npy')

    for id in np.unique(patients):
        patients_features = dataset[np.where(patients == id)[0]]
        for h, hour in enumerate(patients_features):
            features = patients_features[:h]
            for f in range(features.shape[1]):
                if h > 1:
                    if np.sum(np.isnan(features[:, f])) < len(features[:, f]):
                        nan_bounds(features[:, f])
                        nan_interpolate(features[:, f])
                    else:
                        features[:, f] = np.nan_to_num(features[:, f], -1)
