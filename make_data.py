import os
import csv
import scipy.io as sio
import numpy as np
from math import ceil
import scipy.stats as st
import pyeeg
import matplotlib.pyplot as plt

DOG1_DIR = "/gpfs/hpchome/velner/mat_files/Dog_1"
DOG2_DIR = "/gpfs/hpchome/velner/mat_files/Dog_2"
DOG3_DIR = "/gpfs/hpchome/velner/mat_files/Dog_3"
DOG4_DIR = "/gpfs/hpchome/velner/mat_files/Dog_4"
DOG5_DIR = "/gpfs/hpchome/velner/mat_files/Dog_5"
PATIENT1_DIR = "/gpfs/hpchome/velner/mat_files/Patient_1"
PATIENT2_DIR = "/gpfs/hpchome/velner/mat_files/Patient_2"

MAIN_DIR = "/gpfs/hpchome/velner/"
# sys.path.insert(0, "C:\\Users\\MariLiis\\Documents\\Ylikool\\THESIS\\pyEEG\\pyeeg")

DOG1_PREICTALS = 24
DOG1_INTERICTALS = 480
DOG2_PREICTALS = 42
DOG2_INTERICTALS = 500
DOG3_INTERICTALS = 1440
DOG3_PREICTALS = 72
DOG4_INTERICTALS = 804
DOG4_PREICTALS = 97
DOG5_PREICTALS = 30
DOG5_INTERICTALS = 450
PATIENT1_INTERICTALS = 50
PATIENT1_PREICTALS = 18
PATIENT2_INTERICTALS = 42
PATIENT2_PREICTALS = 18

# In seconds
WINDOW_LENGTH_1 = 10
WINDOW_LENGTH_2 = 60

INTERICTAL_MAT_TEMPL = "_interictal_segment_{0:04d}.mat"
PREICTAL_MAT_TEMPL = "_preictal_segment_{0:04d}.mat"
INTERICTAL_SEG_TEMPL = "interictal_segment_{}"
PREICTAL_SEG_TEMPL = "preictal_segment_{}"
FEATURES = ["class", "electrode", "seg", "activity", "mobility", "complexity", "hfd", "skewness", "kurtosis", "ps_delta",
            "ps_theta", "ps_alpha", "ps_beta", "ps_lowgamma", "ps_highgamma", "psr_delta", "psr_theta", "psr_alpha",
            "psr_beta", "psr_lowgamma", "psr_highgamma"]

"""
Taken from the pyEEG module, added Hjorth activity (TP) to the returned values.
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


# dog_1 = get_object_data(DOG1_DIR, "Dog_1", DOG1_INTERICTALS, DOG1_PREICTALS, WINDOW_LENGTH_1)
# dog_2 = get_object_data(DOG2_DIR, "Dog_2", DOG2_INTERICTALS, DOG2_PREICTALS, WINDOW_LENGTH_2)
# dog_3 = get_object_data(DOG3_DIR, "Dog_3", DOG3_INTERICTALS, DOG3_PREICTALS, WINDOW_LENGTH_2)
# dog_4 = get_object_data(DOG4_DIR, "Dog_4", DOG4_INTERICTALS, DOG4_PREICTALS, WINDOW_LENGTH_2)
dog_5 = get_object_data(DOG5_DIR, "Dog_5", DOG5_INTERICTALS, DOG5_PREICTALS, WINDOW_LENGTH_2)

# patient_1 = get_object_data(PATIENT1_DIR, "Patient_1", PATIENT1_INTERICTALS, PATIENT1_PREICTALS, WINDOW_LENGTH_1)
# patient_2 = get_object_data(PATIENT2_DIR, "Patient_2", PATIENT2_INTERICTALS, PATIENT2_PREICTALS, WINDOW_LENGTH_1)

dog_data = dog_5
write_to_file("dog5_minute_data.csv", FEATURES, dog_data)
