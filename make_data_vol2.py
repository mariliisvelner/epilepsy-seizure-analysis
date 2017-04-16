import csv
import pandas as pd
import os

import sys

# The directory of the py files
MAIN_DIR = sys.argv[1]
# The directory of the csv files
DATA_DIR = sys.argv[2]
# The name of the file containing the old data
OLD_FNM = sys.argv[3]
# The file where the new data will be written
NEW_FNM = sys.argv[4]
# The file containing the features
FEATURE_FILE = sys.argv[5]

# Column names for electrode number, segment number and class (interictal (=1) or preictal (=2))
ELEC_COL = "electrode"
SEG_COL = "seg"
CLASS_COL = "class"


# Read in the features
features_f = open(FEATURE_FILE, encoding="UTF-8")
FEATURES = [line.strip() for line in features_f.readlines()]
features_f.close()
# These features cannot contain the class and segment column
FEATURES.remove(CLASS_COL)
FEATURES.remove(SEG_COL)

assert ELEC_COL in FEATURES, "Column {} is not among the features!".format(ELEC_COL)

"""
:param filename (str) -- the name of the file to read the data from

:return the data (pandas DataFrame) from the given file
"""
def read_data(filename):
    print("Reading data...")
    os.chdir(DATA_DIR)
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
    print("Writing to file " + file_name)
    os.chdir(MAIN_DIR)
    result = open(file_name, 'w')
    writer = csv.writer(result, dialect='excel', delimiter=";")
    writer.writerow(headings)
    print("len(data) = ", len(data))
    for row in data:
        writer.writerow(row)
    result.close()


"""
:param class_ (int) -- 1 (interictal) or 2 (preictal)
:param old_data (pandas DataFrame) -- the data which was previously calculated using make_data.py
:param electrodes (int) -- the number of electrodes in the data

Gets the data from old_data corresponding to class_, transforms it and returns the new data.
In old_data, each sample represents a time window for a certain electrode. This means that there are "concurrent" 
samples (sample for time window 1 and electrode 1; sample for time window 1 and electrode 2; ...; sample for time window 
1 and electrode n; sample for time window 2 and electrode 1; ...).

In the new data, each sample represents a time window -- the features of every electrode for this time window are 
concatenated to a single row. This means that there are NO "concurrent" samples (sample for time window 1; sample for 
time window 2; ...).
"""
def get_class_data(class_, old_data, electrodes):
    print("Calculating new data for class {}...".format(class_))
    new_class_data = []
    # Get the <class_> data (interictal or preictal)
    old_class_data = old_data[old_data[CLASS_COL] == class_]
    segments = max(old_class_data[SEG_COL])

    for seg in range(segments):
        # Gets the data of segment <seg + 1>
        seg_data = old_class_data[old_class_data[SEG_COL] == seg + 1]

        electrode_data = []
        # The data of segment <seg + 1> and electrode <elec> is in electrode_data[x-1]
        for elec in range(electrodes):
            electrode_data.append(seg_data[seg_data[ELEC_COL] == elec + 1])

        # Checks if every list in electrode_data is of same length
        # a.k.a has the same number of windows
        assert len(set([len(data) for data in electrode_data])) == 1, "Electrodes have different number of windows"

        # The number of windows for each electrode in a segment
        windows = len(electrode_data[0])

        # For every time window, concat the data that belongs to this time window
        for i in range(windows):
            new_line = []
            for elec in range(electrodes):
                # electrode[elec] selects the data of the electrode <elec>
                # .iloc[i] selects the data of the time window <i>
                # [FEATURES] selects the data of every feature in FEATURES
                new_line += electrode_data[elec].iloc[i][FEATURES].tolist()

            # At last, add the class and segment number to the features of the new data row
            new_class_data.append([class_, seg + 1] + new_line)
    return new_class_data

old_data = read_data(OLD_FNM)
electrodes = max(old_data[ELEC_COL])
new_features = [CLASS_COL, SEG_COL]

# Add feature names for every electrode
for j in range(electrodes):
    new_features += [feature + str(j + 1).zfill(2) for feature in FEATURES]


interictal_data = get_class_data(1, old_data, electrodes)
preical_data = get_class_data(2, old_data, electrodes)

new_data = interictal_data + preical_data

write_to_file(NEW_FNM, new_features, new_data)