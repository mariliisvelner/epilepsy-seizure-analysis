import csv
import pandas as pd
import os


FEATURES = ["electrode", "activity", "mobility", "complexity", "hfd", "skewness", "kurtosis", "ps_delta",
            "ps_theta", "ps_alpha", "ps_beta", "ps_lowgamma", "ps_highgamma", "psr_delta", "psr_theta", "psr_alpha",
            "psr_beta", "psr_lowgamma", "psr_highgamma"]
MAIN_DIR = "C:\\Users\\MariLiis\\Documents\\Ylikool\\THESIS\\Thesis"
DATA_DIR = "C:\\Users\\MariLiis\\Documents\\Ylikool\\THESIS\\Thesis\\data"

old_fnm = "dog5_10sec_data.csv"
new_fnm = "dog5_10sec_data_vol2.csv"
ELEC_COL = "electrode"
SEG_COL = "seg"
STATE_COL = "class"

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
file_name (str) -- the name of the file to write the data to
headings (list<str>) -- a list of headers for the columns in the file
data (list<list<float>>) -- the data to be written to the file; the length of a small list is
                          equal to the length of the headings list

Writes the headings and data to a file with the given name.
"""
def write_to_file(file_name, headings, data):
    print("writing to file " + file_name)
    os.chdir(MAIN_DIR)
    result = open(file_name, 'w')
    writer = csv.writer(result, dialect='excel', delimiter=";")
    writer.writerow(headings)
    print("len(data) = ", len(data))
    for row in data:
        writer.writerow(row)
    result.close()

def get_state_data(state, old_data, electrodes):
    new_state_data = []
    old_state_data = old_data[old_data[STATE_COL] == state]
    segments = max(old_state_data[SEG_COL])
    for seg in range(segments):
        seg_data = old_state_data[old_state_data[SEG_COL] == seg + 1]
        electrode_data = []
        for elec in range(electrodes):
            electrode_data.append(seg_data[seg_data[ELEC_COL] == elec + 1])

        assert len(set([len(data) for data in electrode_data])) == 1, "Electrodes have different number of windows"

        windows = len(electrode_data[0])
        for i in range(windows):
            new_line = []
            for elec in range(electrodes):
                new_line += electrode_data[elec].iloc[i][FEATURES].tolist()
            new_state_data.append([state, seg + 1] + new_line)
    return new_state_data

old_data = read_data(DATA_DIR, old_fnm)
electrodes = max(old_data[ELEC_COL])
new_features = ["class", "seg"]

for j in range(electrodes):
    new_features += [feature + str(j + 1).zfill(2) for feature in FEATURES]

interictal_data = get_state_data(1, old_data, electrodes)
preical_data = get_state_data(2, old_data, electrodes)

new_data = interictal_data + preical_data

write_to_file(new_fnm, new_features, new_data)