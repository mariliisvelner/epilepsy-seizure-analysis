import os

import sys
from collections import defaultdict

from sklearn import ensemble
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


# Name of the data file
DATA_DIR = sys.argv[1]
# Name of the data file
FNM = sys.argv[2]
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
    return data

"""
:param filename (str) -- the name of the file where the data resides
:param to_predict (str) -- the name of the value which is to be predicted

:return a tuple containing 1) the data (pandas DataFrame) used for training (and testing)
2) the labels (pandas DataFrame) of the samples in the data 3) the features (list) of data  
"""
def get_data_target_features(filename, to_predict):
    data = read_data(filename)
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

"""
:param data (array) -- the data to plot
:param target (array) -- the target values (1 (interictal) or 2 (preictal) corresponding to the data

Plots the data. Pink dots represent interictal windows and green dots preictal windows. 
"""
def show_tsne_plot(data, target):
    data_reduced = TruncatedSVD(n_components=2, random_state=0).fit_transform(data)
    data_tsne = TSNE().fit_transform(data_reduced)
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    colors = ["green" if target[i] == 2 else "pink" for i in range(len(target))]
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=colors)
    plt.show()

"""
:param importance_tuples (list) -- a list of tuples, which contain features and their importances; is ordered 
                                   decreasingly by the importance value
:param stds (list) -- a list containing the standard deviations of the features; standard deviation with index i 
                      corresponds to the feature in importance_tuples[i]
:param features (list) -- a list that contains all of the features in the standard order

Plots the features with their importances and standard deviations starting from the feature with the largest importance.
"""
def plot_feature_importances(importance_tuples, stds, features):
    features_ordered = [pair[1] for pair in importance_tuples]
    importances = [pair[0] for pair in importance_tuples]
    indices = [features.index(feature) for feature in features_ordered]

    print("Feature ranking: ")
    for i in range(len(features_ordered)):
        print("{}. {} ({})".format(str(i + 1), features_ordered[i], str(importances[i])))


    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(len(features_ordered)), importances, color="r", yerr=stds, align="center")
    plt.xticks(range(len(features)), indices)
    plt.xlim([-1, len(features)])
    plt.show()

"""
:param filename (str) -- the name of the file where the data resides
:param to_predict (str) -- the name of the value which is to be predicted
:param tries (int) -- number of times for the model to be fitted and test targets predicted 

Splits the data between train and test sets so that they contain different (but the same number of) segments,
then fits the model and predicts <tries> times
"""
def predicting_with_different_segs(filename, to_predict, tries):
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
    # Number of interictal segments
    no_int_seg = len(interictal_segments)

    preictals_in_train = int(no_pr_seg / 2)
    preictals_in_test = int(no_pr_seg / 2)
    interictals_in_train = int(no_pr_seg / 2)
    interictals_in_test = int(no_pr_seg / 2)

    # Contains the accuracies for each prediction
    scores = []
    # Contains the sum of importances for every feature
    feature_importances = [0 for feature in features]

    for i in range(tries):
        # Shuffle the segment numbers
        preictal_segments = shuffle(preictal_segments)
        interictal_segments = shuffle(interictal_segments)

        # Get the TRAIN SEGMENTS as the FIRST no_cl_seg segments of the shuffled preictal and interictal segments
        train_preictal_segments = preictal_segments[:preictals_in_train]
        train_interictal_segments = interictal_segments[:interictals_in_train]
        print("{}. iteration".format(str(i + 1)))
        print("Training set interictal segments: " + str(train_interictal_segments))
        print("Training set preictal segments: " + str(train_preictal_segments))

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
        test_preictal_segments = preictal_segments[preictals_in_test:]
        test_interictal_segments = interictal_segments[interictals_in_test:]
        print("Test set interictal segments: " + str(test_interictal_segments))
        print("Test set preictal segments: " + str(test_preictal_segments))

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
        rfc = ensemble.RandomForestClassifier(n_estimators=10, max_features=None, class_weight="balanced")
        rfc.fit(train_data, train_target)

        predictions = rfc.predict(test_data)
        correct_data = []
        correct_target = []
        incorrect_data = []
        incorrect_target = []
        correctly_predicted = 0
        for i in range(len(predictions)):
            if predictions[i] == test_target[i]:
                correct_data.append(test_data.iloc[i])
                correct_target.append(test_target.iloc[i])
                correctly_predicted += 1
            else:
                incorrect_data.append(test_data.iloc[i])
                incorrect_target.append(test_target.iloc[i])
        # score = rfc.score(test_data, test_target)
        scores.append(correctly_predicted / len(test_target))

        print("Test interictal segments: " + str(test_interictal_segments))
        print("Test preictal segments: " + str(test_preictal_segments))

        # Comment in in order to visualize correctly and incorrectly predicted datawith t-SNE
        # print("Incorrectly predicted tSNE")
        # show_tsne_plot(incorrect_data, incorrect_target)
        #
        # print("Correctly predicted tSNE")
        # show_tsne_plot(correct_data, correct_target)

        # Get the feature importances for this iteration and add the results to the sums in feature_importances
        importances = rfc.feature_importances_
        feature_importances = [importances[i] + feature_importances[i] for i in range(len(feature_importances))]

    assert len(features) == len(feature_importances), "Length of features and feature importances do not match!"

    # Calculate the average feature importance for every feature and print them out with rankings
    feature_importances = [importance / tries for importance in feature_importances]
    indices = np.argsort(feature_importances)[::-1]
    print("Feature ranking (averaged): ")
    for j in range(len(features)):
        print("%d. feature %s (%f)" % (j + 1, features[indices[j]], feature_importances[indices[j]]))

    print("All scores: " + str(scores))
    print("Average score: " + str(sum(scores) / len(scores)))

"""
:param filename (str) -- the name of the file where the data resides
:param to_predict (str) -- the name of the value which is to be predicted
:param splits (int) -- the number of splits to be made in cross validation
:param is_shuffled (bool) -- True, if the data is to be shuffled before split into training and test data,
otherwise False

Performs cross-validation on the given data using StratifiedKFold and prints out the scores along with the standard 
deviation. Also fits the data once and displays the corresponding feature importances. 
"""
def predicting_with_cross_validation(filename, to_predict, splits):
    data = read_data(filename)
    features = [feature for feature in data if not feature in [to_predict, "seg"] and not "electrode" in feature]
    print("Features: ")
    print(features)
    interictal_data = data[data[to_predict] == 1]
    preictal_data = data[data[to_predict] == 2]

    # The number of windows of each class (interictal and preictal) to include both in the training and test data
    no_of_class_windows = int(len(preictal_data) / 2)

    importances_dict = defaultdict(float)
    std_dict = defaultdict(float)
    predictions = []
    for i in range(splits):
        print("Iteration ", str(i + 1))
        rfc = ensemble.RandomForestClassifier(n_estimators=10, max_features=None, class_weight="balanced")

        # Shuffle the data first
        preictal_data = shuffle(preictal_data)
        interictal_data = shuffle(interictal_data)

        # Take the first no_of_class_windows preictal and interictal windows
        train_X_preictal = preictal_data[features][:no_of_class_windows]
        train_y_preictal = preictal_data[:no_of_class_windows][to_predict]
        train_X_interictal = interictal_data[:no_of_class_windows][features]
        train_y_interictal = interictal_data[:no_of_class_windows][to_predict]

        train_data = pd.concat([train_X_preictal, train_X_interictal], ignore_index=True)
        train_target = pd.concat([train_y_preictal, train_y_interictal], ignore_index=True)

        # Take the next no_of_class_windows preictal and interictal windows
        test_X_preictal = preictal_data[no_of_class_windows:][features]
        test_y_preictal = preictal_data[no_of_class_windows:][to_predict]
        test_X_interictal = interictal_data[no_of_class_windows:no_of_class_windows * 2][features]
        test_y_interictal = interictal_data[no_of_class_windows:no_of_class_windows * 2][to_predict]

        test_data = pd.concat([test_X_preictal, test_X_interictal], ignore_index=True)
        test_target = pd.concat([test_y_preictal, test_y_interictal], ignore_index=True)

        rfc.fit(train_data, train_target)

        importances = rfc.feature_importances_
        std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)

        for i in range(len(features)):
            importances_dict[features[i]] += abs(importances[i])
            std_dict[features[i]] += abs(std[i])

        # Predict and append the score to the predictions list
        score = rfc.score(test_data, test_target)
        predictions.append(score)

    print("Average prediction score: ", str(round(sum(predictions)/splits, 3)))

    importances_list = list(zip([round(impt/splits, 4) for impt in list(importances_dict.values())],
                                list(importances_dict.keys())))
    importances_list = sorted(importances_list, reverse=True)
    std_list = [round(std_dict[pair[1]]/splits, 4) for pair in importances_list]

    plot_feature_importances(importances_list, std_list, features)

"""
:param data (pandas DataFrame) -- the data used in predicting
:param features (list) -- the features used in predicting
:rfc (RandomForestClassifier) -- a previously fitted RandomForestClassifier model

Calculates the feature importances of the fitted model and plots them.
"""
def get_feature_importances(data, features, rfc):
    importances = rfc.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)
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


# Comment in with the necessary parameters to get results
# predicting_with_cross_validation(FNM, TO_PREDICT, 5)
# predicting_with_different_segs(FNM, TO_PREDICT, 5)
