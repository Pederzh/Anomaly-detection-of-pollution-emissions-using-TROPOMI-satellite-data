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
def print_image_given_matrix(matrix):
    image = create_image_from_matrix(matrix)
    plt.imshow(image)
    plt.show()


def get_json_content(directory_path, file_name):
    with open(directory_path + file_name + ".json") as json_file:
        data = json.load(json_file)
    return data






def get_mean_given_percent_values(list, percent):
    tot_n = 0
    tot_value = 0
    for i in range(round(len(list) * percent)):
        tot_value += list[i]
        tot_n += 1
    tot_value = round(tot_value / tot_n)
    return tot_value

def remove_missing_data(list):
    list_new = list.copy()
    while None in list_new or -1 in list_new:
        if -1 in list_new:
            list_new.remove(-1)
        if None in list:
            list_new.remove(-1)
    return list_new

def get_sorted_list_from_matrix(matrix):
    list = []
    for y in range(len(data)):
        list += data[y]
    list = remove_missing_data(list)
    list.sort()
    return list

def get_data_w_lowed_noise(data):
    new_data = data.copy()
    list = get_sorted_list_from_matrix(data)
    min_range = get_mean_given_percent_values(list, 1)
    #min_range = get_percentile_value(list,0.50)
    for y in range(len(new_data)):
        for x in range(len(new_data[y])):
            if new_data[y][x] != None and new_data[y][x] != 0:
                if new_data[y][x] <= min_range: new_data[y][x] = min_range
                new_data[y][x] -= min_range
    return new_data

def get_data_w_lowed_noise_by_mean(data):
    new_data = data.copy()
    list = get_sorted_list_from_matrix(data)
    min_range = get_mean_given_percent_values(list, 1)
    #min_range = get_percentile_value(list,0.50)
    for y in range(len(new_data)):
        for x in range(len(new_data[y])):
            if new_data[y][x] != None and new_data[y][x] != 0:
                new_data[y][x] = new_data[y][x] / min_range
    return new_data

def get_percentile_value(list, fraction):
    return list[round(len(list) * fraction)]

def get_matrix_low_pass_filtered(data, keep_fraction):
    freq_scan = scipy.fftpack.fft2(data)
    im_fft2 = freq_scan.copy()
    r, c = im_fft2.shape
    im_fft2[int(r * keep_fraction):int(r * (1 - keep_fraction))] = 0
    im_fft2[:, int(c * keep_fraction):int(c * (1 - keep_fraction))] = 0
    signal_lowpass = scipy.fftpack.ifft2(im_fft2).real
    return signal_lowpass

def divide_image_in_two(matrix, delimiter):
    new = []
    for y in range(len(matrix)):
        new.append([])
        for x in range(len(matrix)):
            new[y].append(1000)
            if matrix[y][x] < delimiter:
                new[y][x] = 0
    return new

def get_signal_peaks(signal):
    return list(scipy.signal.find_peaks(signal, height=30, rel_height=0.5,))[0]

def get_matrix_column(matrix, column_number):
    column = []
    for y in range(len(matrix)):
        column.append(matrix[y][column_number])
    return column

def decrease_image_precision(matrix, decrease_factor):
    for y in range(len(matrix)):
        for x in range(len(matrix)):
            matrix[y][x] = int(round(matrix[y][x]/decrease_factor)*decrease_factor)
    return matrix

def get_peaks_positions(data):
    x_peaks = []
    y_peaks = []
    peaks = []
    data_smoothed = sp.ndimage.filters.gaussian_filter(data, [1, 1], mode='constant')
    for y in range(len(data_smoothed)):
        x_peaks.append(get_signal_peaks(data_smoothed[y]))
    for x in range(len(data_smoothed[0])):
        y_peaks.append(get_signal_peaks(get_matrix_column(data_smoothed, x)))
    for y in range(len(data_smoothed)):
        for x in range(len(data_smoothed[y])):
            if y in y_peaks[x] and x in x_peaks[y]:
                peaks.append([y, x])
    return peaks

def get_matrix_filled_w_value(data, value):
    new_data = []
    for y in range(len(data)):
        new_data.append([])
        for x in range(len(data[y])):
            new_data[y].append(value)
    return new_data

def cut_matrix_values_in_half(data, value):
    matrix = []
    for y in range(len(data)):
        matrix.append([])
        for x in range(len(data[y])):
            if data[y][x] >= value:
                matrix[y].append(1000)
            else:
                matrix[y].append(0)
    return matrix

def add_values_around_pixel(data, y, x, value):
    changed = []
    if y>0 and x>0:
        if data[y-1][x-1] == -1:
            data[y-1][x-1] = value
            changed.append([y-1, x-1])
    if y>0:
        if data[y - 1][x] == -1:
            data[y - 1][x] = value
            changed.append([y-1, x])
    if y>0 and x<len(data[0])-1:
        if data[y - 1][x + 1] == -1:
            data[y - 1][x + 1] = value
            changed.append([y - 1, x + 1])
    if x > 0:
        if data[y][x - 1] == -1:
            data[y][x - 1] = value
            changed.append([y, x-1])
    if x < len(data[0])-1:
        if data[y][x + 1] == -1:
            data[y][x + 1] = value
            changed.append([y, x+1])
    if y<len(data)-1 and x>0:
        if data[y+1][x-1] == -1:
            data[y+1][x-1] = value
            changed.append([y+1, x-1])
    if y<len(data)-1:
        if data[y + 1][x] == -1:
            data[y + 1][x] = value
            changed.append([y+1, x])
    if y<len(data)-1 and x<len(data[0])-1:
        if data[y + 1][x + 1] == -1:
            data[y + 1][x + 1] = value
            changed.append([y+1, x+1])
    return changed

def create_shape(shapes_matrix, y, x):
    #print(shapes_matrix)
    changed = add_values_around_pixel(shapes_matrix, y, x, shapes_matrix[y][x])
    for i in range(len(changed)):
        create_shape(shapes_matrix, changed[i][0], changed[i][1])

def differentiate_shapes(data):
    shapes_matrix = []
    shape_number = 0
    for y in range(len(data)):
        shapes_matrix.append([])
        for x in range(len(data)):
            if data[y][x] == 1000:
                shapes_matrix[y].append(-1)
            else:
                shapes_matrix[y].append(0)
    for y in range(len(data)):
        for x in range(len(data)):
            if data[y][x] == 1000 and shapes_matrix[y][x] == -1:
                shape_number += 50
                shapes_matrix[y][x] = shape_number
                create_shape(shapes_matrix, y, x)
    return shapes_matrix

def get_missing_id(data):
    list = []
    for y in range(len(data)):
        for x in range(len(data[y])):
            if data[y][x] != None:
                list.append(data[y][x]["id"])
    id = 1
    while id in list:
        id += 1
    return id

def update_found_shapes(found_shapes, shapes, shape_id, level):
    plumes = []
    for y in range(len(found_shapes)):
        for x in range(len(found_shapes[y])):
            if found_shapes[y][x] != None and shapes[y][x] == shape_id:
                if found_shapes[y][x] not in plumes:
                    plumes.append(found_shapes[y][x])
    count_over_limit = 0
    unify = True
    id = get_missing_id(found_shapes)
    max_peak ={"peak": level, "id": id}
    if len(plumes)>1:
        for i in range(len(plumes)):
            if (plumes[i]["peak"] - level)/plumes[i]["peak"] >= 0.7:
                count_over_limit += 1
            if plumes[i]["peak"]>max_peak["peak"]: max_peak = plumes[i]
    if (len(plumes)) == 1:
        max_peak = plumes[0]
    if max_peak["peak"]*0.15 > level: unify = False
    if count_over_limit > 1: unify = False

    for y in range(len(shapes)):
        for x in range(len(shapes[y])):
            if shapes[y][x] == shape_id:
                if unify:
                    found_shapes[y][x] = max_peak
                shapes[y][x] = 0




def unify_shapes(found_shapes, shapes, considered_level):
    for y in range(len(shapes)):
        for x in range(len(shapes[y])):
            if shapes[y][x]>0:
                update_found_shapes(found_shapes, shapes, shapes[y][x], considered_level)




def print_image_given_plumes(data):
    matrix = []
    for y in range(len(data)):
        matrix.append([])
        for x in range(len(data[y])):
            if data[y][x] == None:
                matrix[y].append(0)
            else:
                matrix[y].append(data[y][x]["id"] * 50)
    print_image_given_matrix(matrix)

def get_matrix_shapes_from_peak(data, peak_level):
    level = peak_level
    found_shapes = get_matrix_filled_w_value(data, None)
    while level >= 50: #peak_level*0.5:
        cutted_data = cut_matrix_values_in_half(data, level)
        shapes_data = differentiate_shapes(cutted_data)
        unify_shapes(found_shapes, shapes_data, level)
        level = level - 5
    return found_shapes



def get_plumes(data, peak_levels):
    #for i in range(len(peak_levels)):
    plumes = get_matrix_shapes_from_peak(data, peak_levels[0])
    return(plumes)


def detect_shapes(image_data):
    # lowering data noise
    #print_image_given_matrix(image_data)
    #data = get_data_w_lowed_noise_by_mean(image_data)
    data = get_data_w_lowed_noise(image_data)
    #data = get_matrix_filled_w_value(image_data, 0)
    """data_tmp = sp.ndimage.filters.gaussian_filter(image_data, [10, 10], mode='constant')
    for y in range(len(data_tmp)):
        for x in range(len(data_tmp[y])):
            data[y][x] = image_data[y][x] - data_tmp[y][x]
            if data[y][x] < 0: data[y][x] = 0"""
    # getting peak levels
    peaks = get_peaks_positions(data)
    # low pass filtering the image
    data = get_matrix_low_pass_filtered(data, 0.15)
    data = sp.ndimage.filters.gaussian_filter(data, [1, 1], mode='constant')
    # reducing image range quality
    data = decrease_image_precision(data, 5)
    # getting peaks value levels
    peak_levels = []
    for i in range(len(peaks)):
        if data[peaks[i][0]][peaks[i][1]] not in peak_levels:
            peak_levels.append(data[peaks[i][0]][peaks[i][1]])
    peak_levels.sort()
    peak_levels.reverse()


    # !!!!!!!!!!!!!!!!!!!!!!!
    #data = sp.ndimage.filters.gaussian_filter(data, [1, 1], mode='constant')
    """data1 = sp.ndimage.filters.gaussian_filter(data, [1, 1], mode='constant')
    data2 = sp.ndimage.filters.gaussian_filter(data, [2, 2], mode='constant')
    data3 = sp.ndimage.filters.gaussian_filter(data, [3.5, 3.5], mode='constant')
    data4 = sp.ndimage.filters.gaussian_filter(data, [5, 5], mode='constant')
    for y in range(len(data1)):
        for x in range(len(data1[y])):
            #data[y][x] = round(pow(abs(data1[y][x]) * abs(data2[y][x]) * abs(data3[y][x]) * abs(data4[y][x]), 1/4))
            #data[y][x] = round(((abs(data1[y][x]) + abs(data2[y][x]) + abs(data3[y][x]) + abs(data4[y][x]))/4))
            data[y][x] = round(pow((data1[y][x]*data1[y][x] + data2[y][x]*data2[y][x] +data3[y][x]*data3[y][x] + data4[y][x]*data4[y][x])/4, 1/2))
    # !!!!!!!!!!!!!!!!!!!!!!!
    print_image_given_matrix(data1)"""
    print_image_given_matrix(data)

    #print_image_given_matrix(data)
    # getting plumes shapes
    plumes = get_plumes(data, peak_levels)
    plumes_list = []
    for i in range(len(peaks)):
        if plumes[peaks[i][0]][peaks[i][1]] != None:
            if plumes[peaks[i][0]][peaks[i][1]] not in plumes_list:
                plumes_list.append(plumes[peaks[i][0]][peaks[i][1]])
    for y in range(len(plumes)):
        for x in range(len(plumes[y])):
            if plumes[y][x] != None:
                if plumes[y][x] not in plumes_list:
                    plumes[y][x] = None
    print_image_given_plumes(plumes)














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


#getting the image
"""data = get_json_content("../Data/NO2/Sabetta Port/images/2021/05/05/balanced/", "mean")
#data = data["4"]
print_image_given_matrix(data)
shapes = detect_shapes(data)"""

for i in range(10):
    if i!= 2:
        day_string = ""
        if i+1 <=9: day_string = "0"
        day_string += str(i+1)
        data = get_json_content("../Data/NO2/Sabetta Port/images/2021/05/" + day_string + "/balanced/", "mean")#["4"]
        shapes = detect_shapes(data)














