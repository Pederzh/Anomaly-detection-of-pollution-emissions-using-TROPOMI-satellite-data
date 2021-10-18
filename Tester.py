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
from scipy.ndimage import rotate
from scipy import interpolate
import scipy.ndimage
import scipy.fftpack
import scipy.signal


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










def reduce_image_dimension(matrix, factor):
    new_matrix = []
    for y in range(len(matrix)):
        if y%factor == 0: new_matrix.append([])
        for x in range(len(matrix)):
            if x%factor == 0: new_matrix[int(y/factor)].append(0)
            new_matrix[int(y / factor)][int(x / factor)] += matrix[y][x]
    for y in range(len(new_matrix)):
        for x in range(len(new_matrix[y])):
            new_matrix[y][x] = new_matrix[y][x]/(factor*factor)
    return new_matrix


def print_images_peaks(data):
    # decreasing image values precision
    #data = divide_image_in_two(data, 100)
    #data = decrease_image_precision(data, 20)
    #data = reduce_image_dimension(data,2)

    # smoothing the image
    sigma_y = 1
    sigma_x = 1
    sigma = [sigma_y, sigma_x]
    data_smoothed = sp.ndimage.filters.gaussian_filter(data, sigma, mode='constant')
    plot_data = sp.ndimage.filters.gaussian_filter(data, sigma, mode='constant')
    # print_image_given_matrix(data)
    # print_image_given_matrix(data_smoothed)

    # PEAKS
    min_height = 50
    prominence = 0
    perpendiculars_peaks = []
    """for i in range(len(data_smoothed[0])):
        signal = get_matrix_column(data_smoothed, i)
        signal = float_array_to_int(signal)
        peaks = list(
            scipy.signal.find_peaks(signal, height=min_height, threshold=None, distance=None, prominence=prominence,
                                    width=0, wlen=None,
                                    rel_height=0.5, plateau_size=None))
        perpendiculars_peaks.append(peaks)
    parallels_peaks = []
    for i in range(len(data_smoothed)):
        signal = float_array_to_int(data_smoothed[i])
        peaks = list(
            scipy.signal.find_peaks(signal, height=min_height, threshold=None, distance=None, prominence=prominence,
                                    width=0, wlen=None,
                                    rel_height=0.5, plateau_size=None))
        parallels_peaks.append(peaks)"""

    # VALLEYS
    min_height = -50
    prominence = 0
    perpendiculars_valleys = []
    for i in range(len(data_smoothed[0])):
        signal = get_matrix_column(data_smoothed * (-1), i)
        signal = float_array_to_int(signal)
        peaks = list(
            scipy.signal.find_peaks(signal, height=min_height, threshold=None, distance=None, prominence=prominence,
                                    width=0, wlen=None,
                                    rel_height=0.5, plateau_size=None))
        perpendiculars_valleys.append(peaks)
    parallels_valleys = []
    for i in range(len(data_smoothed)):
        signal = float_array_to_int((data_smoothed * (-1))[i])
        peaks = list(
            scipy.signal.find_peaks(signal, height=min_height, threshold=None, distance=None, prominence=prominence, width=0,
                                    wlen=None,
                                    rel_height=0.5, plateau_size=None))
        parallels_valleys.append(peaks)

    data_per_peaks = sp.ndimage.filters.gaussian_filter(data, sigma, mode='constant')
    data_par_peaks = sp.ndimage.filters.gaussian_filter(data, sigma, mode='constant')
    for y in range(len(data_smoothed)):
        for x in range(len(data_smoothed[y])):
            """if x in parallels_peaks[y][0]: data_par_peaks[y][x] = data_smoothed[y][x] = 1000
            if y in perpendiculars_peaks[x][0]: data_per_peaks[y][x] = data_smoothed[y][x] = 1000
            #if y in perpendiculars[x][0] or x in parallels[y][0]:
            if x in parallels_peaks[y][0] and y in perpendiculars_peaks[x][0]:
                data_smoothed[y][x] = 2000
                #data_par_peaks[y][x] = data_smoothed[y][x] = -1
                #data_per_peaks[y][x] = -1"""
            if x in parallels_valleys[y][0]: data_par_peaks[y][x] = data_smoothed[y][x] = -2
            if y in perpendiculars_valleys[x][0]: data_per_peaks[y][x] = data_smoothed[y][x] = -2

    #print_image_given_matrix(plot_data)
    #print_image_given_matrix(data_per_peaks)
    #print_image_given_matrix(data_par_peaks)
    #print_image_given_matrix(data_smoothed)
    #print_image_given_matrix(plot_data)
    #print_image_given_matrix(data)



    for y in range(len(plot_data)):
        for x in range(len(plot_data[y])):
            plot_data[y][x] = data_smoothed[y][x]
            if data_smoothed[y][x] == -2:
                plot_data[y][x] = 0
    return plot_data







def calculate_angle(x1, x2):
    return math.atan(x2-x1)

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

def get_json_content_w_name(directory_path, name):
    with open(directory_path + name + ".json") as json_file:
        data = json.load(json_file)
    return data

def decrease_image_precision(matrix, decrease_factor):
    for y in range(len(matrix)):
        for x in range(len(matrix)):
            matrix[y][x] = round(matrix[y][x]/decrease_factor)*decrease_factor
    return matrix

def decrease_image_precision_by_min(matrix, decrease_factor):
    for y in range(len(matrix)):
        for x in range(len(matrix)):
            matrix[y][x] = int(matrix[y][x]/decrease_factor)*decrease_factor
    return matrix

def divide_image_in_two(matrix, delimiter):
    new = []
    for y in range(len(matrix)):
        new.append([])
        for x in range(len(matrix)):
            new[y].append(1000)
            if matrix[y][x] < delimiter:
                new[y][x] = 0
    return new

def print_images_ridge(data):
    # decreasing image values precision
    min_height = 50
    prominence = 0
    data = decrease_image_precision(data, 100)
    plot_data = []
    for y in range(len(data)):
        plot_data.append([])
        for x in range(len(data)):
            plot_data[y].append(0)
    for i in range(len(data[0])):
        column_signal = get_matrix_column(data,i)
        column_signal = float_array_to_int(column_signal)
        peaks = list(scipy.signal.find_peaks(column_signal, height=min_height, prominence=prominence, rel_height=0.5))
        for peak_id in range(len(peaks[0])):
            item_id = peaks[0][peak_id]
            if data[item_id][i] == data[item_id][i-1]:
                plot_data[item_id][i] = 1000
    print_image_given_matrix(plot_data)













def cut_from_top(matrix, delimiter):
    new = []
    for y in range(len(matrix)):
        new.append([])
        for x in range(len(matrix)):
            new[y].append(matrix[y][x])
            if matrix[y][x] > delimiter:
                new[y][x] = delimiter
    return new

def get_image_range(matrix, min_value, max_value):
    new_matrix = []
    for y in range(len(matrix)):
        new_matrix.append([])
        for x in range(len(matrix[y])):
            new_matrix[y].append(0)
            if matrix[y][x] >= min_value and matrix[y][x] <= max_value:
                new_matrix[y][x] = 1000
    return new_matrix



def test(data):

    #decreasing image values precision
    # data = divide_image_in_two(data, 100)
    # data = reduce_image_dimension(data,2)
    # data = decrease_image_precision(data, 20)

    for y in range(len(data)):
        for x in range(len(data[y])):
            data[y][x] = data[y][x]-40
            if data[y][x]<0 and data[y][x]>=-40: data[y][x] = 0




    # smoothing the image
    sigma_y = 1
    sigma_x = 1
    sigma = [sigma_y, sigma_x]
    data_smoothed = sp.ndimage.filters.gaussian_filter(data, sigma, mode='constant')

    def remove_over_values(matrix, value):
        new_matrix = []
        for y in range(len(matrix)):
            new_matrix.append([])
            for x in range(len(matrix[y])):
                new_matrix[y].append(matrix[y][x])
                if matrix[y][x] >= value: new_matrix[y][x]=0
        return new_matrix


    """for i in range(10):
        print_image_given_matrix(divide_image_in_two(data_smoothed, 100-i*10))"""
    #data_smoothed = decrease_image_precision(data_smoothed, 10)
    print_image_given_matrix(data_smoothed)




    freq_scan = scipy.fftpack.fft2(data)
    keep_fraction = 0.15
    im_fft2 = freq_scan.copy()
    r, c = im_fft2.shape
    im_fft2[int(r * keep_fraction):int(r * (1 - keep_fraction))] = 0
    im_fft2[:, int(c * keep_fraction):int(c * (1 - keep_fraction))] = 0
    signal_lowpass = scipy.fftpack.ifft2(im_fft2).real
    #signal_lowpass = decrease_image_precision(signal_lowpass, 10)
    print_image_given_matrix(signal_lowpass)

    """for i in range(10):
        print_image_given_matrix(divide_image_in_two(signal_lowpass, 110-i*10))
    """


    #data_smoothed = sp.ndimage.filters.gaussian_filter(data, [3,3], mode='constant')
    #data_smoothed = decrease_image_precision(data_smoothed, 10)

    #print_image_given_matrix(data)

























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


# getting the image
#data = get_json_content("../Data/NO2/Sabetta Port/images/2021/05/05/balanced/")["4"]
#data = get_json_content_w_name("../Data/NO2/Sabetta Port/images/2021/05/04/balanced/", "mean")
#test(data)

for i in range(10):
    day_string = ""
    if i+1 <=9: day_string = "0"
    day_string += str(i+1)
    data = get_json_content("../Data/NO2/Sabetta Port/images/2021/05/" + day_string + "/balanced/")["4"]
    test(data)

#print_images_peaks(data)

"""def unify(m1, m2):
    for y in range(len(m1)):
        for x in range(len(m1[y])):
            if m2[y][x] > 0: m1[y][x] = (m2[y][x]+m1[y][x]) / 2
    return m1

print_image_given_matrix(data)
old_data = data
values = print_images_peaks(data)
values = rotate(values, angle=45, reshape=False, order=1)
values = rotate(values, angle=-45, reshape=False, order=1)
for i in range(32):
    angle = (i+1)*5
    data = rotate(old_data, angle=angle, reshape=False, order=1)
    values = unify(values, rotate(print_images_peaks(data), angle=-angle, reshape=False, order=1))
    print_image_given_matrix(values)

print_image_given_matrix(values)"""


#print_images_ridge(data)

#peaks = list(scipy.signal.find_peaks(signal, height=None, threshold=None, distance=None, prominence=0, width=0, wlen=None, rel_height=0.5, plateau_size=None))
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
#print(peaks[0])
#print(list(peaks[1]["prominences"]))









"""def calculate_angle(x1, x2):
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
"""




"""
df_array = get_df_array(data[50], "normal") + get_df_array(data_smoothed[50], "smoothed")
#df_array = get_df_array(original_signal, "normal") + get_df_array(signal, "smoothed") + get_df_array(new_signal, "subtracted") + get_df_array(new_signal_smoothed, "subtracted smoothed") + get_df_array(subtracted_smoothed, "peak smoothed")

#df_array = get_df_array(get_matrix_column(data, 45), "normal") + get_df_array(signal, "smoothed")
df = get_df_array_converted(df_array)
fig = px.histogram(df, x="x", y="value", color="name", barmode="overlay", nbins=int((date_end - date_start).days))
fig.show()
"""

"""df_array = []
for y in range(int(len(data)/2)):
    signal = get_matrix_column(data_smoothed, y*2)
    print(list(scipy.signal.find_peaks(signal, height=None, threshold=None, distance=None, prominence=5, width=[3, None], wlen=None, rel_height=0.5, plateau_size=None))[0])
"""