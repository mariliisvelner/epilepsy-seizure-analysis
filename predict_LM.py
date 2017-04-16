import os
from collections import defaultdict
import numpy as np
import sys
from sklearn import linear_model
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle


MAIN_DIR = sys.argv[1]
DATA_DIR = sys.argv[2]
# Name of the file
FNM = sys.argv[3]
TO_PREDICT = "class"

print("File: " + FNM)

"""
:param filename (str) -- the name of the file to read the data from
                         the file must reside in the directory DATA_DIR

:return the data (pandas DataFrame) from the given file
"""
def read_data(filename):
    print("Reading data...")
    os.chdir(DATA_DIR)
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
    data = read_data(filename)
    features = [feature for feature in data if not feature in [to_predict, "seg"]]
    print("Features: ")
    print(features)
    data_preictal = data[data[to_predict] == 2]

    # Shuffle the interictals and then get the same number of interictals as there are preictals
    data_interictal = shuffle(data[data[to_predict] == 1])
    data_interictal = data_interictal[0: len(data_preictal)]

    data_features = pd.concat([data_interictal, data_preictal], ignore_index=True)[features]
    data_target = pd.concat([data_interictal, data_preictal], ignore_index=True)[to_predict]
    return data_features, data_target, features


def recursive_feature_elimination(data, target, features, splits, is_shuffled, steps):
    lm = linear_model.LogisticRegressionCV(solver="liblinear", penalty="l1")
    skf = StratifiedKFold(n_splits=splits, shuffle=is_shuffled)
    selector = RFECV(lm, step=steps, cv=skf)
    selector = selector.fit(data, target)

    # The mask contains a boolean for each feature
    # A True value indicates that the feature was chosen for predicting, False indicates otherwise
    mask = selector.support_
    print("Feature masks: ")
    for i in range(len(features)):
        print("{}: {}".format(features[i], mask[i]))

    # Rankings of the features
    # All of the chosen features have a ranking of 1, others have a lower ranking
    ranking = selector.ranking_
    print("Features with rankings:")
    for i in range(len(ranking)):
        print("{}: {}".format(str(i), features[i]))

    print("Score: " + str(selector.score))


def cross_validate(data, target, splits, is_shuffled):
    lm = linear_model.LogisticRegressionCV(solver="liblinear", penalty="l1")
    skf = StratifiedKFold(n_splits=splits, shuffle=is_shuffled)

    scores = cross_val_score(lm, data, target, cv=skf)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def get_coef_values(data, target, features, iterations):
    coef_dict = defaultdict(float)
    for i in range(iterations):
        lm = linear_model.LogisticRegressionCV(solver="liblinear", penalty="l1")
        lm.fit(data, target)
        # print(i)
        # print(lm.coef_)
        coef = lm.coef_[0]
        for j in range(len(features)):
            coef_dict[features[j]] += abs(coef[j])

    coef_list = list(zip(list(coef_dict.values()), list(coef_dict.keys())))
    coef_list = sorted(coef_list, reverse=True)
    for pair in coef_list:
        print(pair)

"""
:param filename (str) -- the name of the file where the data resides
:param to_predict (str) -- the name of the value which is to be predicted
:param tries (int) -- number of times for the model to be fitted and test targets predicted 

Splits the data between train and test sets so that they contain different (but the same number of) segments,
then fits the model and predicts <tries> times
"""
def predicting_with_different_segs(filename, to_predict, tries):
    data = read_data(filename)
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

    # Contains the accuracies for each prediction
    scores = []
    # Contains the sum of importances for every feature
    feature_coefficients = [0 for feature in features]

    for i in range(tries):
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
        lm = linear_model.LogisticRegressionCV(solver="liblinear", penalty="l1")
        lm.fit(train_data, train_target)
        scores.append(lm.score(test_data, test_target))

        coef = lm.coef_[0]
        feature_coefficients = [abs(coef[i]) + feature_coefficients[i] for i in range(len(feature_coefficients))]

    assert len(features) == len(feature_coefficients), "Length of features and feature coefficients do not match!"

    # Calculate the average feature importance for every feature and print them out with rankings
    feature_coefficients = [coef / tries for coef in feature_coefficients]
    indices = np.argsort(feature_coefficients)[::-1]
    print("Feature coef. ranking (averaged): ")
    for j in range(len(features)):
        print("%d. feature %s (%f)" % (j + 1, features[indices[j]], feature_coefficients[indices[j]]))

    print("All scores: " + str(scores))
    print("Average score: " + str(sum(scores) / len(scores)))

# data_target_features = get_data_target_features("patient1_minute_data_vol2.csv", TO_PREDICT)
#
# data = data_target_features[0]
# target = data_target_features[1]
# features = data_target_features[2]
#
# get_coef_values(data, target, features, 5)
#
# cross_validate(data, target, 5, True)

predicting_with_different_segs(FNM, TO_PREDICT, 5)