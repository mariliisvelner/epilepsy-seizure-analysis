import os

from sklearn import ensemble
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

"""
:param is_in_HPC (bool) -- if the program is to be run in HPC

:return a tuple containing the main directory, where the program is, and the data directory, where the data files reside 
"""


def get_directories(is_in_HPC):
    HPC_MAIN_DIR = "/gpfs/hpchome/velner/"
    HPC_DATA_DIR = "/gpfs/hpchome/velner/data/"
    HOME_MAIN_DIR = "C:\\Users\\MariLiis\\Documents\\Ylikool\\THESIS\\Thesis"
    HOME_DATA_DIR = "C:\\Users\\MariLiis\\Documents\\Ylikool\\THESIS\\Thesis\\data"
    if is_in_HPC:
        return HPC_MAIN_DIR, HPC_DATA_DIR
    return HOME_MAIN_DIR, HOME_DATA_DIR


MAIN_DIR, DATA_DIR = get_directories(False)
TO_PREDICT = "class"

"""
:param dir (str) -- the directory, where the data resides
:param filename (str) -- the name of the file to read the data from

:return the data (pandas DataFrame) from the given file
"""


def read_data(dir, filename):
    print("Reading data...")
    os.chdir(dir)
    data = pd.read_csv(filename, delimiter=";")
    os.chdir(MAIN_DIR)
    return data


"""
:param filename (str) -- the name of the file where the data resides
:param to_predict (str) -- the feature which is to be predicted

:return a tuple containing 1) the data (pandas DataFrame) used for training (and testing)
2) the labels (pandas DataFrame) of the samples in the data 3) the features (list) of data  
"""


def get_data_target_features(filename, to_predict):
    data = read_data(DATA_DIR, filename)
    features = [feature for feature in data if not feature in [TO_PREDICT, "seg"]]
    print("Features: ")
    print(features)
    data_preictal = data[data[to_predict] == 2]

    # Shuffle the interictals and then get the same number of interictals as there are preictals
    data_interictal = shuffle(data[data[to_predict] == 1])
    data_interictal = data_interictal[0: len(data_preictal)]

    data_features = pd.concat([data_interictal, data_preictal], ignore_index=True)[features]
    data_target = pd.concat([data_interictal, data_preictal], ignore_index=True)[to_predict]
    return data_features, data_target, features


def cross_validate_different_segs(filename, to_predict, splits):
    data = read_data(DATA_DIR, filename)
    features = [feature for feature in data if not feature in [TO_PREDICT, "seg"]]
    print("Features: ")
    print(features)

    # Get preictal and interictal data
    data_preictal = data[data[to_predict] == 2]
    data_interictal = data[data[to_predict] == 1]

    # Get the segment numbers of preictal and interictal segments
    preictal_segments = list(set(data_preictal["seg"]))
    interictal_segments = list(set(data_interictal["seg"]))

    # Number of preictal segments
    no_pr_seg = len(preictal_segments)
    # The number of segments of a certain class (interictal and preictal) to include in the train data
    no_cl_seg = int(no_pr_seg / 2)

    scores = []
    for i in range(splits):
        # Shuffle the segment numbers
        preictal_segments = shuffle(preictal_segments)
        interictal_segments = shuffle(interictal_segments)

        # Get the TRAIN SEGMENTS as the FIRST no_cl_seg segments of the shuffled preictal and interictal segments
        train_preictal_segments = preictal_segments[:no_cl_seg]
        train_interictal_segments = interictal_segments[:no_cl_seg]
        print("{}. Training set interictal segments: {}".format(str(i + 1), str(train_interictal_segments)))
        print("{}. Training set preictal segments: {}".format(str(i + 1), str(train_preictal_segments)))

        # Get the interictal windows, which's segments are in train_interictal_segments
        train_interictal_windows = data_interictal[data_interictal["seg"].isin(train_interictal_segments)]
        # Get the preictal windows, which's segments are in train_preictal_segments
        train_preictal_windows = data_preictal[data_preictal["seg"].isin(train_preictal_segments)]
        print("Training set no of interictal windows: " + str(len(train_interictal_windows)))
        print("Training set no of preictal windows: " + str(len(train_preictal_windows)))

        # Get the training set data and targets
        train_data = pd.concat([train_interictal_windows, train_preictal_windows], ignore_index=True)[features]
        train_target = pd.concat([train_interictal_windows, train_preictal_windows], ignore_index=True)[to_predict]

        print("Training data length: " + str(len(train_data)))

        # Get the TEST SEGMENTS as the LAST no_cl_seg segments of the shuffled preictal and interictal segments
        test_preictal_segments = preictal_segments[no_cl_seg:]
        test_interictal_segments = interictal_segments[no_cl_seg:no_cl_seg * 2]
        print("{}. Test set interictal segments: {}".format(str(i + 1), str(test_interictal_segments)))
        print("{}. Test set preictal segments: {}".format(str(i + 1), str(test_preictal_segments)))

        # Get the interictal windows, which's segments are in test_interictal_segments
        test_interictal_windows = data_interictal[data_interictal["seg"].isin(test_interictal_segments)]
        # Get the interictal windows, which's segments are in test_preictal_segments
        test_preictal_windows = data_preictal[data_preictal["seg"].isin(test_preictal_segments)]
        print("Test set no of interictal windows: " + str(len(test_interictal_windows)))
        print("Test set no of preictal windows: " + str(len(test_preictal_windows)))

        # Get the test set data and targets
        test_data = pd.concat([test_interictal_windows, test_preictal_windows], ignore_index=True)[features]
        test_target = pd.concat([test_interictal_windows, test_preictal_windows], ignore_index=True)[to_predict]

        print("Test data length: " + str(len(test_data)))

        # Fit and predict
        rf = ensemble.RandomForestClassifier(n_estimators=10, max_features=None, class_weight="balanced")
        rf.fit(train_data, train_target)
        scores.append(rf.score(test_data, test_target))

    print("All scores: " + str(scores))
    print("Average score: " + str(sum(scores) / len(scores)))


def cross_validate(data, target, splits, is_shuffled):
    rf = ensemble.RandomForestClassifier(n_estimators=10, max_features=None, class_weight="balanced")
    skf = StratifiedKFold(n_splits=splits, shuffle=is_shuffled)

    scores = cross_val_score(rf, data, target, cv=skf)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def get_feature_importances(data, target):
    rf = ensemble.RandomForestClassifier(n_estimators=10, max_features=None, class_weight="balanced")
    rf.fit(data, target)

    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    print(data.shape)
    print("Feature ranking: ")
    for f in range(data.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(data.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(data.shape[1]), indices)
    plt.xlim([-1, data.shape[1]])
    plt.show()


# data_target_features = get_data_target_features("patient1_10sec_data_vol2.csv", TO_PREDICT)
#
# data = data_target_features[0]
# target = data_target_features[1]
# features = data_target_features[2]

# get_feature_importances(data, target)
#
# cross_validate(data, target, 5, True)

cross_validate_different_segs("patient1_minute_data_vol2.csv", TO_PREDICT, 5)
