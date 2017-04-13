import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

MAIN_DIR = "C:\\Users\\MariLiis\\Documents\\Ylikool\\THESIS\\Thesis"
DATA_DIR = "C:\\Users\\MariLiis\\Documents\\Ylikool\\THESIS\\Thesis\\data"
TO_PREDICT = "class"

def read_data(dir, filename):
    os.chdir(dir)
    data = pd.read_csv(filename, delimiter=";")
    os.chdir(MAIN_DIR)
    return data

def get_data_and_label(filename, to_predict):
    data = read_data(DATA_DIR, filename)
    features = [feature for feature in data if not feature in [to_predict, "seg"]]
    print(features)
    data_preictal = data[data[to_predict] == 2]
    data_interictal = data[data[to_predict] == 1]
    data_features = pd.concat([data_interictal, data_preictal], ignore_index=True)[features]
    data_target = pd.concat([data_interictal, data_preictal], ignore_index=True)[to_predict]
    return data_features, data_target, features


def show_tsne_plot(data, target):
    data_reduced = TruncatedSVD(n_components=2, random_state=0).fit_transform(data)
    data_tsne = TSNE().fit_transform(data_reduced)
    print(data_tsne)
    print(len(data_tsne))
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    colors = target.replace(2, "green")
    colors = colors.replace(1, "pink")
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=colors)
    plt.show()

fnm = "patient2_10sec_data_vol2.csv"
things = get_data_and_label(fnm, TO_PREDICT)
data = things[0]
target = things[1]
features = things[2]

show_tsne_plot(data, target)