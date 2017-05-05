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

# The directory of the mat files
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
:param filename (str) -- the name of the file where the data resides
:param to_predict (str) -- the name of the value which is to be predicted
:param splits (int) -- the number of splits for the recursive feature elimination
:param is_shuffled (bool) -- True, if the data is to be shuffled before splitting in the recursive feature elimination,
                             False otherwise
:param steps (int) -- corresponds to the number of features to remove at each iteration
                          
Performs a recursive feature elimination with automatic tuning of the number of features selected with cross-validation.
Prints out the prediction accuracy score and the feature rankings. 
"""
def predicting_with_RFECV(filename, to_predict, splits, is_shuffled, steps):
    data_target_features = get_data_target_features(filename, to_predict)

    data = data_target_features[0]
    target = data_target_features[1]
    features = data_target_features[2]

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

"""
:param filename (str) -- the name of the file where the data resides
:param to_predict (str) -- the feature which is to be predicted
:param iterations (int) -- the number of iterations for the cross-validation

Fits and predicts shuffled data for <iterations> iterations, averages the scores and coefficients and prints them out.
"""
def predicting_with_cross_validation(filename, to_predict, iterations):
    data = read_data(filename)
    features = [feature for feature in data if not feature in [to_predict, "seg"] and not "electrode" in feature]
    print("Features: ")
    print(features)
    interictal_data = data[data[to_predict] == 1]
    preictal_data = data[data[to_predict] == 2]

    # The number of windows of each class (interictal and preictal) to include both in the training and test data
    no_of_class_windows = int(len(preictal_data) / 2)

    coef_dict = defaultdict(float)
    predictions = []
    for i in range(iterations):
        print("Iteration ", str(i + 1))
        lm = linear_model.LogisticRegressionCV(solver="liblinear", penalty="l1")

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

        lm.fit(train_data, train_target)
        coef = lm.coef_[0]
        # Increase each feature's coefficient sum in the coef_dict dictionary
        for j in range(len(features)):
            coef_dict[features[j]] += abs(coef[j])

        # Predict and append the score to the predictions list
        score = lm.score(test_data, test_target)
        predictions.append(score)

    print("Average prediction score: ", str(round(sum(predictions)/len(predictions), 3)))

    print("Feature ranking with and averaged coefficients: ")
    coef_list = list(zip(list(coef_dict.values()), list(coef_dict.keys())))
    coef_list = sorted(coef_list, reverse=True)
    for i in range(len(coef_list)):
        avg_coef = round(coef_list[i][0]/iterations, 4)
        print("{}. {} ({})".format(str(i + 1), coef_list[i][1], str(avg_coef)))

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
    # Number of interictal segments
    no_int_seg = len(interictal_segments) / 2

    preictals_in_train = int(no_pr_seg / 2)
    preictals_in_test = int(no_pr_seg / 2)
    interictals_in_train = int(no_int_seg / 2)
    interictals_in_test = int(no_int_seg / 2)

    # Contains the accuracies for each prediction
    scores = []
    # Contains the sum of importances for every feature
    feature_coefficients = [0 for feature in features]

    for i in range(tries):
        # Shuffle the segment numbers
        preictal_segments = shuffle(preictal_segments)
        interictal_segments = shuffle(interictal_segments)

        # Get the TRAIN SEGMENTS as the FIRST no_cl_seg segments of the shuffled preictal and interictal segments
        train_preictal_segments = preictal_segments[:preictals_in_train]
        train_interictal_segments = interictal_segments[:preictals_in_train]
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
        test_preictal_segments = preictal_segments[preictals_in_test:]
        test_interictal_segments = interictal_segments[preictals_in_test:preictals_in_test*2]
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


# Comment in with the necessary parameters to get results
# predicting_with_different_segs(FNM, TO_PREDICT, 5)
# predicting_with_cross_validation(FNM, TO_PREDICT, 5)

