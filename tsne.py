import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

# Name of the data file
DATA_DIR = sys.argv[1]
# Name of the data file
FNM = sys.argv[2]
TO_PREDICT = "class"

"""
:param dir (str) -- the directory, where the data resides
:param filename (str) -- the name of the file to read the data from

:return the data (pandas DataFrame) from the given file
"""
def read_data(dir, filename):
    os.chdir(dir)
    data = pd.read_csv(filename, delimiter=";")
    return data

"""
:param filename (str) -- the name of the file where the data resides
:param to_predict (str) -- the feature which is to be predicted

:return a tuple containing 1) the data (pandas DataFrame) used for training (and testing) 
2) the labels (pandas DataFrame) of the samples in the data 3) the features (list) of the data  
"""
def get_data_target_features(filename, to_predict):
    data = read_data(DATA_DIR, filename)
    features = [feature for feature in data if not feature in [to_predict, "seg"]]

    print(features)
    data_preictal = data[data[to_predict] == 2]
    data_interictal = data[data[to_predict] == 1]
    data_features = pd.concat([data_interictal, data_preictal], ignore_index=True)[features]
    data_target = pd.concat([data_interictal, data_preictal], ignore_index=True)[to_predict]
    return data_features, data_target, features


"""
:param data (array) -- the data to plot
:param target (array) -- the target values of the data to plot (target is either 1 (interictal) or 2 (preictal))

Displays the TSNE plot of the data.
"""
def show_tsne_plot(data, target):
    data_reduced = TruncatedSVD(n_components=30, random_state=0).fit_transform(data)
    data_tsne = TSNE(perplexity=30).fit_transform(data_reduced)
    print(data_tsne)
    print(len(data_tsne))
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    colors = target.replace(2, "green")
    colors = colors.replace(1, "pink")
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=colors)
    plt.show()

fnm = FNM
data_target_features = get_data_target_features(fnm, TO_PREDICT)
data = data_target_features[0]
target = data_target_features[1]
features = data_target_features[2]

show_tsne_plot(data, target)