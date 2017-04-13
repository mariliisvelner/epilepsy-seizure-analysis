import os
import pandas as pd
import operator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
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


def print_top_features(top_dict):
    for i in range(len(top_dict)):
        print("{}. feature {} ({})".format(str(i + 1), top_dict[i][0], str(top_dict[i][1])))


def print_top_n_features(dict, n):
    for i in range(n):
        print("{}. feature {} ({})".format(str(i + 1), dict[i][0], str(dict[i][1])))


def fit_with_SelectKBest(data, target, k):
    selector = SelectKBest(f_classif, k=k)
    data = selector.fit_transform(data, target)

    all_scores = selector.scores_
    top_features = [features[i] for i in range(len(features)) if selector.get_support()[i]]
    top_scores = [all_scores[i] for i in range(len(all_scores)) if selector.get_support()[i]]
    top_dict = sorted(dict(zip(top_features, top_scores)).items(), key=operator.itemgetter(1), reverse=True)
    all_dict = sorted(dict(zip(features, all_scores)).items(), key=operator.itemgetter(1), reverse=True)

    print_top_features(top_dict)
    print_top_n_features(all_dict, 20)

    # Plot the scores
    plt.bar(range(len(features)), all_scores)
    plt.xticks(range(len(features)), features, rotation='vertical')
    plt.show()

    return data


def cross_validate(data, target, splits, is_shuffled):
    clf = GaussianNB()
    skf = StratifiedKFold(n_splits=splits, shuffle=is_shuffled)
    scores = cross_val_score(clf, data, target, cv=skf)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


data_target_features = get_data_target_features("patient1_10sec_data_vol2.csv", TO_PREDICT)

data = data_target_features[0]
target = data_target_features[1]
features = data_target_features[2]

data = fit_with_SelectKBest(data, target, 5)

cross_validate(data, target, 5, True)
