import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import sys
import os

# The directory where the .mat file resides
DATA_DIR = sys.argv[1]
# The object (Dog_n (n = 1..5) or Patient_m (m = 1..2))
OBJECT = sys.argv[2]
# Number of the .mat file
FILE_NO = sys.argv[3]
# The file where the templates are
TEMPLATE_FILE = sys.argv[4]
# True, if the .mat file represents an interictal segment, otherwise False
IS_INTERICTAL = True if sys.argv[5] == "True" else False

os.chdir(DATA_DIR + "\\" + OBJECT)

temp_f = open(TEMPLATE_FILE, encoding="UTF-8")
templates = [line.split(";")[1].strip() for line in temp_f.readlines()]
temp_f.close()

# Return the correct .mat file name and segment label templates from the templates file
mat_templ, seg_templ = "", ""
if IS_INTERICTAL:
    mat_templ = templates[0]
    seg_templ = templates[2]
else:
    mat_templ = templates[1]
    seg_templ = templates[3]

"""
:param fnm (str) -- the name of the file where the data resides
:param seg_templ (str) -- the template for the segment label 
:param file_no (str) -- the number of the mat file

Returns the data from the .mat file with the given name and number. 
"""
def get_data(fnm, seg_templ, file_no):
    file = sio.loadmat(fnm.format(int(file_no)))
    data = file[seg_templ.format(int(file_no))][0][0]
    array = data["data"]
    array = array.astype(np.float, copy=False)
    return array


"""
Plots the given EEG data. 
"""
def show_plot(data):
    # 15 subplots in one column, background color is white
    fig, ax = plt.subplots(nrows=15, ncols=1, facecolor='white')
    # 5000 Hz is the sampling frequency
    # 5000 * 60 is the amount of data points corresponding to one minute
    dp = 5000 * 60 * 5
    lines = [x for x in range(1, dp + 1, 5000 * 50)]
    i = 0
    for row in ax:
        row.plot(list(range(1, dp + 1)), data[i][0:dp])

        # Add red vertical lines for every point x in lines
        # for x in lines:
        #     row.axvline(x=x, color="red")

        # Remove the axes
        row.axis("off")
        i += 1
    # Reduce the space between the subplots
    plt.subplots_adjust(wspace=0)
    plt.show()


fnm = OBJECT + mat_templ
data = get_data(fnm, seg_templ, FILE_NO)
show_plot(data)