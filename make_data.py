import os
import csv
import scipy.io as sio
import numpy as np
from math import ceil
import scipy.stats as st
import pyeeg
import matplotlib.pyplot as plt
import sys

# The directory of the py files
MAIN_DIR = sys.argv[1]
# The directory of the mat files
MAT_FILE_DIR = sys.argv[2]
# The number of interictal segments of the given object
INTERICTALS = int(sys.argv[3])
# The number of preictal segments of the given object
PREICTALS = int(sys.argv[4])
# The length of the time windows to be extracted from the data (in seconds)
WINDOW_LENGTH = int(sys.argv[5])
# The file which contains templates for mat files
TEMPLATE_FILE = sys.argv[6]
# The file which contains the feature names
FEATURE_FILE = sys.argv[7]
# The object (Dog_n (n = 1..5) or Patient_m (m = 1..2))
OBJECT = sys.argv[8]
# The file where the extracted features will be written
RESULT_FNM = sys.argv[9]


# Read in the templates
temp_f = open(TEMPLATE_FILE, encoding="UTF-8")
templates = [line.split(";")[1].strip() for line in temp_f.readlines()]
temp_f.close()

INTERICTAL_MAT_TEMPL = templates[0]
PREICTAL_MAT_TEMPL = templates[1]
INTERICTAL_SEG_TEMPL = templates[2]
PREICTAL_SEG_TEMPL = templates[3]

# Read in the features
features_f = open(FEATURE_FILE, encoding="UTF-8")
FEATURES = [line.strip() for line in features_f.readlines()]
features_f.close()


"""
Taken from the PyEEG module, added Hjorth activity (TP) to the returned values.
PyEEG: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3070217/
"""
def hjorth(X, D=None):
    """ Compute Hjorth mobility and complexity of a time series from either two
    cases below:
        1. X, the time series of type list (default)
        2. D, a first order differential sequence of X (if D is provided,
           recommended to speed up)

    In case 1, D is computed using Numpy's Difference function.

    Notes
    -----
    To speed up, it is recommended to compute D before calling this function
    because D may also be used by other functions whereas computing it here
    again will slow down.

    Parameters
    ----------

    X
        list

        a time series

    D
        list

        first order differential sequence of a time series

    Returns
    -------

    As indicated in return line

    Hjorth mobility and complexity

    """

    if D is None:
        D = np.diff(X)
        D = D.tolist()

    D.insert(0, X[0])  # pad the first difference
    D = np.array(D)

    n = len(X)

    M2 = float(sum(D ** 2)) / n
    TP = sum(np.array(X) ** 2)
    M4 = 0
    for i in range(1, len(D)):
        M4 += (D[i] - D[i - 1]) ** 2
    M4 = M4 / n

    return TP, np.sqrt(M2 / TP), np.sqrt(float(M4) * TP / M2 / M2)
    # Hjorth Activity, Mobility and Complexity. Changed: TP as Hjorth Activity is returned as well

def plot_and_show(x_axis, y_axis):
    plt.plot(x_axis, y_axis)
    plt.show()


"""
sample (list<float>) -- an EEG sample from a mat-file containing the time-domain values for a single electrode
window_length (int) -- the length of the windows to be computed in seconds
freq (int) -- the sampling frequency from the mat-file

Return (list<list<float>>) -- a list of lists, where each small list represents a window_length second long list containing
time-domain values from the given sample
"""
def make_windows(sample, window_length, freq):
    data_points = window_length * freq
    time_windows = []
    for n in range(0, len(sample), data_points):
        new = sample[n:n + data_points]
        time_windows.append(new)
    return time_windows


"""
classif (int) -- 1 if the data is interictal files, 2 if preictal files
mat_templ (str) -- the template for the mat-files
seg_templ (str) -- the template for the label of the data in the mat-file
file_no -- the number of mat-files

Returns (list<list<float>>) -- processed data from the mat files with the given template;
for every electrode in every mat-file, windows with length of WINDOW_LENGTH are calculated and
features 'classif', 'electrode', 'sequence', 'mobility', 'complexity', 'hfd', 'skewness' and
'kurtosis' are extracted.
"""
def get_state_data(classif, mat_templ, seg_templ, file_no, window_length):
    result = []
    for i in range(1, file_no + 1):
        print("mat_file: ", i)
        file = sio.loadmat(mat_templ.format(i))
        data = file[seg_templ.format(str(i))][0][0]

        # READ IN DATA
        array = data["data"]
        sampling_freq = ceil(data["sampling_frequency"][0][0])
        # ROW_DURATION = data["data_length_sec"][0][0]
        # sequence_no = data["sequence"][0][0]
        # data["channels"] -- electrode names

        array = array.astype(np.float, copy=False)

        for elc in range(len(array)):
            windows = make_windows(array[elc], window_length, sampling_freq)

            for window in windows:
                mean = round(sum(window) / len(window), 2)
                window = [el - mean for el in window]

                hjorth_params = hjorth(window)
                activity = hjorth_params[0]
                mobility = hjorth_params[1]
                complexity = hjorth_params[2]
                hfd = pyeeg.hfd(window, 5)
                skewness = st.skew(window)
                kurtosis = st.kurtosis(window)
                bins = pyeeg.bin_power(window, [0, 4, 8, 12, 30, 70, 180], sampling_freq)

                datarow = [classif, elc + 1, i, activity, mobility, complexity, hfd, skewness, kurtosis] + list(
                    bins[0]) + list(bins[1])
                result.append(datarow)
                break
            break
    return result


"""
directory (str) -- the directory of the mat-files of the object
object (str) -- "Dog_1", "Dog_2", ... or "Patient_2"
inter_fileno -- the number of interictal mat-files for the object
pre_fileno -- the number of preictal mat-files for the object

Returns (list<list<float>>) -- the preictal and interictal processed data as a list.
"""
def get_object_data(directory, object, inter_fileno, pre_fileno, window_length):
    os.chdir(directory)
    print("get_object_data: ", object)
    inter_mat_templ = object + INTERICTAL_MAT_TEMPL
    pre_mat_templ = object + PREICTAL_MAT_TEMPL
    interictal_data = get_state_data(1, inter_mat_templ, INTERICTAL_SEG_TEMPL, inter_fileno, window_length)
    preictal_data = get_state_data(2, pre_mat_templ, PREICTAL_SEG_TEMPL, pre_fileno, window_length)
    print("len(interictal_data) = ", len(interictal_data))
    print("len(preictal_data) = ", len(preictal_data))
    return interictal_data + preictal_data


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


data = get_object_data(MAT_FILE_DIR, OBJECT, INTERICTALS, PREICTALS, WINDOW_LENGTH)

write_to_file(RESULT_FNM, FEATURES, data)
