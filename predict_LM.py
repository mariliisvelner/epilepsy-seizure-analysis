import os
from collections import defaultdict

import sys
from sklearn import linear_model
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle


MAIN_DIR = sys.argv[1]
DATA_DIR = sys.argv[2]
TO_PREDICT = "class"


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


data_target_features = get_data_target_features("patient1_10sec_data_vol2.csv", TO_PREDICT)

data = data_target_features[0]
target = data_target_features[1]
features = data_target_features[2]

get_coef_values(data, target, features, 5)

cross_validate(data, target, 5, True)
