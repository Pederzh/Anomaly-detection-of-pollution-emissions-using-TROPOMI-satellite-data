import datetime
import io
import json
import math
from pathlib import Path
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import scipy as sp
from scipy import interpolate
import scipy.ndimage


import matplotlib.pyplot as plt

from PIL import Image
from numpy import array
import numpy as np
from oauthlib.oauth2 import BackendApplicationClient
import scipy.signal
from scipy.signal import savgol_filter, butter,filtfilt, bspline
from requests_oauthlib import OAuth2Session





def get_rgb_values(value):
    """
    [minVal, [0, 0, 0.5]],
    [minVal + 0.125 * diff, [0, 0, 1]],
    [minVal + 0.375 * diff, [0, 1, 1]],
    [minVal + 0.625 * diff, [1, 1, 0]],
    [minVal + 0.875 * diff, [1, 0, 0]],
    [maxVal, [0.5, 0, 0]]
    """
    rgb = [0, 0, 0, 255]
    if (value == -1): return [0, 0, 0, 0]
    precision = 1016  # given by the rgb composition
    prop = value / precision
    # [0, 0, 1]
    if prop <= 0.125:
        min = 0
        max = 0.125
        rangeVal = max - min
        new_value = round((prop - min) / rangeVal * 127)
        rgb[2] = 128 + new_value
        return rgb
    # [0, 1, 1]
    if prop <= 0.375:
        min = 0.125
        max = 0.375
        rangeVal = max - min
        new_value = round((prop - min)/rangeVal * 255)
        rgb[1] = new_value
        rgb[2] = 255
        return rgb
    # [1, 1, 0]
    if prop <= 0.625:
        min = 0.375
        max = 0.625
        rangeVal = max - min
        new_value = round((prop - min)/rangeVal * 255)
        rgb[0] = new_value
        rgb[1] = 255
        rgb[2] = 255 - new_value
        return rgb
    # [1, 0, 0]
    if prop <= 0.875:
        min = 0.625
        max = 0.875
        rangeVal = max - min
        new_value = round((prop - min) / rangeVal * 255)
        rgb[0] = 1
        rgb[1] = 255 - new_value
        return rgb
    # [0.5, 0, 0]
    min = 0.875
    max = 1
    rangeVal = max - min
    new_value = round(127 - (prop - min) / rangeVal * 127)
    rgb[0] = 128 + new_value
    return rgb


def create_image_from_matrix(data):
    image = []
    for y in range(len(data)):
        image.append([])
        for x in range(len(data)):
            image[y].append(get_rgb_values(data[y][x]))
    return image










def print_image_given_matrix(matrix):
    image = create_image_from_matrix(matrix)
    plt.imshow(image)
    plt.show()

def get_df_array(array, name):
    df_array = []
    for x in range(len(array)):
        df_array.append({
            "value": array[x],
            "x": x,
            "name": name
        })
    return df_array

def get_df_array_converted(df_array):
    return pd.DataFrame(df_array)

def float_array_to_int(array):
    new_array = []
    for y in range(len(array)):
        new_array.append(round(array[y]))
    return new_array

def get_matrix_column(matrix, column_number):
    column = []
    for y in range(len(matrix)):
        column.append(matrix[y][column_number])
    return column

def get_json_content(directory_path):
    with open(directory_path + "data.json") as json_file:
        data = json.load(json_file)
    return data

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#                       MAIN
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

values = {
    "product_types": ["NO2", "CO", "CH4", "SO2"],
    "locations_name": ["Bering Strait", "Sabetta Port"],
    "minQas": ["high", "all"],
    "image_types": ["unprocessed", "balanced"]
}
date = datetime.datetime.now()
date_start = date.replace(year=2021, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
date_end = date.replace(year=2021, month=10, day=1, hour=0, minute=0, second=0, microsecond=0)

product_type = values["product_types"][0]
location_name = values["locations_name"][1]
minQa = values["minQas"][1]
image_type = values["image_types"][1]

data = get_json_content("../Data/NO2/Sabetta Port/images/2021/05/05/balanced/")["4"]


n = len(data) # widht/height of the array
sigma_y = 3
sigma_x = 3
sigma = [sigma_y, sigma_x]
data_smoothed = sp.ndimage.filters.gaussian_filter(data, sigma, mode='constant')

#print_image_given_matrix(data)
print_image_given_matrix(data_smoothed)

"""for y in range(len(data)):
    for x in range(len(data[y])):
        data[y][x] = abs(data[y][x] - data_smoothed[y][x])
print_image_given_matrix(data)"""



signal = get_matrix_column(data_smoothed, 45)

"""signal_smoothed = savgol_filter(signal, 51, 5)
signal_smoothed = float_array_to_int(signal_smoothed)
print(signal_smoothed)"""



peaks = list(scipy.signal.find_peaks(signal, height=None, threshold=None, distance=None, prominence=0, width=0, wlen=None, rel_height=0.5, plateau_size=None))
# wlen
#       indica la dimensione della finestra da considerare per decidere se il punto deve
#       o non deve essere considerato un picco
# height
#       altezza minima che deve avere il picco
# treshold
#       distanza verticale minima dai due campioni vicini
# distance
#       distanza minima che ci deve essere tra due picchi
# prominence
#       valore minimo di risalto del picco
print(peaks[0])
print(list(peaks[1]["prominences"]))









def calculate_angle(x1, x2):
    return math.atan(x2-x1)

# CALCULATING RANGE
flag = True
point = 50

new_signal = []
for i in range(len(signal)): new_signal.append((0))
new_signal[50] = signal[50]

cc = 0
angle = 0
while flag:
    cc += 1
    if cc + point < len(signal):
        new_angle = calculate_angle(signal[cc + point], signal[cc-1 + point])
        if angle < new_angle:
            angle = new_angle
            new_signal[cc+point] = signal[cc + point]
        else:
            flag = False
    else:
        flag = False
cc = 0
angle = 0
flag = True
while flag:
    cc += 1
    if cc + point < len(signal):
        new_angle = calculate_angle(signal[point - cc], signal[point - cc + 1])
        if angle < new_angle:
            angle = new_angle
            new_signal[point - cc] = signal[point - cc]
        else:
            flag = False
    else:
        flag = False

original_signal =get_matrix_column(data, 45)
for i in range(len(new_signal)): new_signal[i] = abs(original_signal[i] - new_signal[i])
new_signal_smoothed = sp.ndimage.filters.gaussian_filter(new_signal, 1, mode='constant')
subtracted_smoothed = []
for i in range(len(signal)): subtracted_smoothed.append(abs(signal[i] - new_signal_smoothed[i]))

df_array = get_df_array(data[50], "normal") + get_df_array(data_smoothed[50], "smoothed")
#df_array = get_df_array(original_signal, "normal") + get_df_array(signal, "smoothed") + get_df_array(new_signal, "subtracted") + get_df_array(new_signal_smoothed, "subtracted smoothed") + get_df_array(subtracted_smoothed, "peak smoothed")

#df_array = get_df_array(get_matrix_column(data, 45), "normal") + get_df_array(signal, "smoothed")
df = get_df_array_converted(df_array)
fig = px.histogram(df, x="x", y="value", color="name", barmode="overlay", nbins=int((date_end - date_start).days))
fig.show()


"""df_array = []
for y in range(int(len(data)/2)):
    signal = get_matrix_column(data_smoothed, y*2)
    print(list(scipy.signal.find_peaks(signal, height=None, threshold=None, distance=None, prominence=5, width=[3, None], wlen=None, rel_height=0.5, plateau_size=None))[0])
"""