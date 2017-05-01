import os
import pandas as pd
import operator

import sys
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
import numpy as np

# Name of the data file
DATA_DIR = sys.argv[1]
# Name of the data file
FNM = sys.argv[2]
TO_PREDICT = "class"

print("File: " + FNM)

"""
:param dir (str) -- the directory, where the data resides
:param filename (str) -- the name of the file to read the data from

:return the data (pandas DataFrame) from the given file
"""
def read_data(filename):
    print("Reading data...")
    os.chdir(DATA_DIR)
    data = pd.read_csv(filename, delimiter=";")
    return data

"""
:param filename (str) -- the name of the file where the data resides
:param to_predict (str) -- the feature which is to be predicted

:return a tuple containing 1) the data (pandas DataFrame) used for training (and testing)
2) the labels (pandas DataFrame) of the samples in the data 3) the features (list) of data  
"""
def get_data_target_features(filename, to_predict):
    data = read_data(filename)
    features = [feature for feature in data if not feature in [to_predict, "seg"] and not "electrode" in feature]
    print("Features: ")
    print(features)
    data_preictal = data[data[to_predict] == 2]

    # Shuffle the interictals and then get the same number of interictals as there are preictals
    data_interictal = shuffle(data[data[to_predict] == 1])
    data_interictal = data_interictal[0: len(data_preictal)]

    data_features = pd.concat([data_interictal, data_preictal], ignore_index=True)[features]
    data_target = pd.concat([data_interictal, data_preictal], ignore_index=True)[to_predict]
    return data_features, data_target, features

"""
:param top_dict (dict<str, float>) -- the dictionary containing the chosen features and their scores

Prints out the features and their scores in their ranking order.
"""
def print_chosen_features(top_dict):
    print("Chosen features: ")
    for i in range(len(top_dict)):
        print("{}. feature {} ({})".format(str(i + 1), top_dict[i][0], str(top_dict[i][1])))

"""
:param data (array) -- the data that contains the windows with the values of the features in <features>
:param target (array) -- an array that corresponds to <data> and contains the values of the feature "class" (1 for 
                         interictals and 2 for preictals)
:param features (list) -- the features used in training
:param k (int) -- the number of features to select using SelectKBest
                        
Selects k best features using SelectKBest and returns the data with only the values of those features.                         
"""
def fit_with_SelectKBest(data, target, features, k):
    selector = SelectKBest(f_classif, k=k)
    data = selector.fit_transform(data, target)

    all_scores = selector.scores_
    top_features = [features[i] for i in range(len(features)) if selector.get_support()[i]]
    top_scores = [all_scores[i] for i in range(len(all_scores)) if selector.get_support()[i]]
    top_dict = sorted(dict(zip(top_features, top_scores)).items(), key=operator.itemgetter(1), reverse=True)
    print_chosen_features(top_dict)

    return data


"""
:param filename (str) -- the name of the file where the data resides
:param to_predict (str) -- the name of the value which is to be predicted
:param tries (int) -- number of times for the model to be fitted and test targets predicted 

Splits the data between train and test sets so that they contain different (but the same number of) segments,
then fits the model and predicts <tries> times
"""
def predicting_with_different_segs(filename, to_predict, tries, k):
    data = read_data(filename)
    features = [feature for feature in data if not feature in [to_predict, "seg"] and not "electrode" in feature]
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
    feature_rankings = [0 for feature in features]

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

        selector = SelectKBest(f_classif, k=k)
        train_data = selector.fit_transform(train_data, train_target)

        all_scores = selector.scores_
        feature_rankings = [feature_rankings[i] + all_scores[i] for i in range(len(all_scores))]
        new_features = [features[i] for i in range(len(features)) if selector.get_support()[i]]

        print(type(test_data))
        test_data = test_data[new_features]

        # Fit and predict
        gnb = GaussianNB()
        gnb.fit(train_data, train_target)
        scores.append(gnb.score(test_data, test_target))

    # Calculate the average feature importance for every feature and print them out with rankings
    feature_rankings = [ranking / tries for ranking in feature_rankings]
    indices = np.argsort(feature_rankings)[::-1]
    print("Feature ranking (averaged): ")
    for j in range(len(features)):
        print("%d. feature %s (%f)" % (j + 1, features[indices[j]], feature_rankings[indices[j]]))

    print("All scores: " + str(scores))
    print("Average score: " + str(sum(scores) / len(scores)))


"""
:param filename (str) -- the name of the file where the data resides
:param to_predict (str) -- the name of the value which is to be predicted
:param splits (int) -- the number of splits for the cross validation
:param is_shuffled (bool) -- True, if data is to be shuffled before splitting in cross validation, False otherwise
:param use_select_k (bool) -- True, if SelectKBest is used to fit the data before fitting with GaussianNB
:param k (int) -- the number of features to select with SelectKBest

Predicts using the cross_val_score method and uses SelectKBest to fit the data beforehand, if the use_select_k is set to
True. 
"""
def predicting_with_cross_validation(filename, to_predict, splits, is_shuffled, use_select_k, k):
    data_target_features = get_data_target_features(filename, to_predict)

    data = data_target_features[0]
    target = data_target_features[1]
    features = data_target_features[2]

    if use_select_k:
        data = fit_with_SelectKBest(data, target, features, k)

    clf = GaussianNB()
    skf = StratifiedKFold(n_splits=splits, shuffle=is_shuffled)
    print("Predicting...")
    scores = cross_val_score(clf, data, target, cv=skf)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

predicting_with_cross_validation(FNM, TO_PREDICT, 5, True, True, 20)

# predicting_with_different_segs(FNM, TO_PREDICT, 5, 20)