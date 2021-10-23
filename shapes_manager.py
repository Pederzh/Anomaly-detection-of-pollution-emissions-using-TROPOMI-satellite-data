import datetime
import io
import json
import math
import sys
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
from scipy.interpolate import griddata
import statistics as stst
import numpy


import matplotlib.pyplot as plt

from PIL import Image
from numpy import array
import numpy as np
from oauthlib.oauth2 import BackendApplicationClient
import scipy.signal
from scipy.signal import savgol_filter, butter,filtfilt, bspline
from requests_oauthlib import OAuth2Session




#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#                               PLOTTING IMAGES
#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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






#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#                               DATA MANIPULATION
#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def reduce_image_dimension(matrix, factor):
    new_matrix = []
    for y in range(len(matrix)):
        if y%factor == 0: new_matrix.append([])
        for x in range(len(matrix[y])):
            if x%factor == 0: new_matrix[int(y/factor)].append(0)
            new_matrix[int(y / factor)][int(x / factor)] += matrix[y][x]
    for y in range(len(new_matrix)):
        for x in range(len(new_matrix[y])):
            new_matrix[y][x] = new_matrix[y][x]/(factor*factor)
    return new_matrix

def increment_image_dimension(matrix, factor):
    new_matrix = []
    for y in range(len(matrix)):
        for i in range(factor): new_matrix.append([])
        for x in range(len(matrix[y])):
            for i in range(factor):
                for j in range(factor):
                    new_matrix[y*factor+i].append(matrix[y][x])
    return new_matrix


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
    for y in range(len(matrix)):
        list += matrix[y]
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

def divide_image_in_two(matrix, delimiter):
    new = []
    for y in range(len(matrix)):
        new.append([])
        for x in range(len(matrix)):
            new[y].append(1000)
            if matrix[y][x] < delimiter:
                new[y][x] = 0
    return new

def decrease_image_precision(matrix, decrease_factor):
    for y in range(len(matrix)):
        for x in range(len(matrix)):
            matrix[y][x] = int(round(matrix[y][x]/decrease_factor)*decrease_factor)
    return matrix




#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#                               LOW PASS FILTER
#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def get_matrix_low_pass_filtered(data, keep_fraction):
    freq_scan = scipy.fftpack.fft2(data)
    im_fft2 = freq_scan.copy()
    r, c = im_fft2.shape
    im_fft2[int(r * keep_fraction):int(r * (1 - keep_fraction))] = 0
    im_fft2[:, int(c * keep_fraction):int(c * (1 - keep_fraction))] = 0
    signal_lowpass = scipy.fftpack.ifft2(im_fft2).real
    return signal_lowpass





#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#                               PEAK FUNCTIONS
#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def get_signal_peaks(signal):
    return list(scipy.signal.find_peaks(signal, height=5, rel_height=0.5, prominence=30))[0]

def get_matrix_column(matrix, column_number):
    column = []
    for y in range(len(matrix)):
        column.append(matrix[y][column_number])
    return column

def get_peaks_positions(data):
    x_peaks = []
    y_peaks = []
    peaks = []
    data_smoothed = sp.ndimage.filters.gaussian_filter(data, [0, 0], mode='constant')
    for y in range(len(data_smoothed)):
        x_peaks.append(get_signal_peaks(data_smoothed[y]))
    for x in range(len(data_smoothed[0])):
        y_peaks.append(get_signal_peaks(get_matrix_column(data_smoothed, x)))
    for y in range(len(data_smoothed)):
        for x in range(len(data_smoothed[y])):
            if y in y_peaks[x] and x in x_peaks[y]:
                peaks.append([y, x])
    return peaks







#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#                                 INTERPOLATION
#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def get_interpolated_image(points, data):
    all_points = []
    for y in range(len(data)):
        for x in range(len(data[y])):
            all_points.append([y, x])
    values = []
    for i in range(len(points)):
        values.append(data[points[i][0]][points[i][1]])
    values = numpy.array(values)
    new_values = griddata(points, values, all_points, method='cubic')
    new_data = []
    count = 0
    for y in range(len(data)):
        new_data.append([])
        for x in range(len(data[y])):
            if numpy.isnan(new_values[count]):
                new_data[y].append(0)
            else:
                if new_values[count] < 0:
                    new_data[y].append(0)
                else:
                    new_data[y].append(round(new_values[count]))
            count += 1
    return new_data

def get_sub_points(data, side_len, n):
    points = []
    for y in range(len(data)):
        for x in range(len(data[y])):
            if (y-int(n/side_len)) % side_len == 0 and (x-(n-int(n/side_len)*side_len)) % side_len == 0:
                if (data[y][x] != None and data[y][x] != -1):
                    points.append([y, x])
    return points

def get_interpolated_data(data_set, side_len, what_to_return):
    data_mean = []
    final_data = []
    list_of_data_set = []
    tot = []
    for y in range(len(data_set)):
        data_mean.append([])
        tot.append([])
        final_data.append([])
        for x in range(len(data_set[y])):
            data_mean[y].append(0)
            tot[y].append(0)
            final_data[y].append(0)
    for i in range(side_len*side_len):
        points = get_sub_points(data_set, side_len, i)
        new_data_set = get_interpolated_image(points, data_set)
        for y in range(len(data_mean)):
            for x in range(len(data_mean[y])):
                if (y - int(i / side_len)) % side_len != 0 or (x - (i - int(i / side_len) * side_len)) % side_len != 0:
                    data_mean[y][x] += new_data_set[y][x]
                    tot[y][x] += 1
                else:
                    new_data_set[y][x] = -1
        list_of_data_set.append(new_data_set)
    for y in range(len(data_mean)):
        for x in range(len(data_mean[y])):
            data_mean[y][x] = data_mean[y][x]/tot[y][x]
    if what_to_return == "mean": return data_mean

    if what_to_return == "closer_to_mean":
        for y in range(len(data_mean)):
            for x in range(len(data_mean[y])):
                closer = 0
                for i in range(len(list_of_data_set)):
                    if abs(list_of_data_set[i][y][x] - data_mean[y][x]) < abs(closer - data_mean[y][x]) and list_of_data_set[i][y][x] != -1:
                        closer = list_of_data_set[i][y][x]
                final_data[y][x] = closer
        return final_data

    if what_to_return == "median":
        for y in range(len(data_mean)):
            for x in range(len(data_mean[y])):
                values_list = []
                for i in range(len(list_of_data_set)):
                    if list_of_data_set[i][y][x] != -1: values_list.append(list_of_data_set[i][y][x])
                median = stst.median(values_list)
                final_data[y][x] = median
        return final_data

    if what_to_return == "mode":
        for y in range(len(data_mean)):
            for x in range(len(data_mean[y])):
                values_list = []
                for i in range(len(list_of_data_set)):
                    if list_of_data_set[i][y][x] != -1: values_list.append(list_of_data_set[i][y][x])
                median = stst.mode(values_list)
                final_data[y][x] = median
        return final_data

    if what_to_return == "min":
        for y in range(len(data_mean)):
            for x in range(len(data_mean[y])):
                values_list = []
                for i in range(len(list_of_data_set)):
                    if list_of_data_set[i][y][x] != -1: values_list.append(list_of_data_set[i][y][x])
                min_val = min(values_list)
                final_data[y][x] = min_val
        return final_data

    if what_to_return == "max":
        for y in range(len(data_mean)):
            for x in range(len(data_mean[y])):
                values_list = []
                for i in range(len(list_of_data_set)):
                    if list_of_data_set[i][y][x] != -1: values_list.append(list_of_data_set[i][y][x])
                min_val = max(values_list)
                final_data[y][x] = min_val
        return final_data

def get_mean_of_interpolated_data(data_set, side_lens, mean_type):
    final_data = []
    tot = 0
    for y in range(len(data_set)):
        final_data.append([])
        for x in range(len(data_set[y])):
            final_data[y].append(0)
    for i in range(len(side_lens)):
        data_mean = get_interpolated_data(data_set, side_lens[i], mean_type)
        for y in range(len(final_data)):
            for x in range(len(final_data[y])):
                final_data[y][x] += data_mean[y][x]
        tot += 1
    for y in range(len(final_data)):
        for x in range(len(final_data[y])):
            final_data[y][x] = final_data[y][x] / tot
    return final_data

def get_final_interpolated_data(data_set):
    side_lens = [3, 4, 5]
    mean_types = ["mean", "median"]
    final_data = []
    for y in range(len(data_set)):
        final_data.append([])
        for x in range(len(data_set[y])):
            final_data[y].append(0)
    for i in range(len(mean_types)):
        data_tmp = get_mean_of_interpolated_data(data_set, side_lens, mean_types[i])
        # print_image_given_matrix(data_tmp)
        for y in range(len(final_data)):
            for x in range(len(final_data[y])):
                final_data[y][x] += data_tmp[y][x]
    for y in range(len(final_data)):
        for x in range(len(final_data[y])):
            final_data[y][x] = final_data[y][x] / len(mean_types)
    return final_data








#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#                               SHAPES FUNCTIONS
#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

"""min_peaks_distance = 0.25
min_shape_border = 0.5"""

min_peaks_distance = 0.3
min_shape_border = 0.1
distance_level = 30

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
            #if (plumes[i]["peak"] - level)/plumes[i]["peak"] >= min_peaks_distance:
            if plumes[i]["peak"] - level > distance_level:
                count_over_limit += 1
            if plumes[i]["peak"]>max_peak["peak"]: max_peak = plumes[i]
    if (len(plumes)) == 1:
        max_peak = plumes[0]
    if max_peak["peak"]*min_shape_border > level: unify = False
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
    while level >= 10: #peak_level*0.5:
        cutted_data = cut_matrix_values_in_half(data, level)
        shapes_data = differentiate_shapes(cutted_data)
        unify_shapes(found_shapes, shapes_data, level)
        level = level - 10
    return found_shapes



def get_plumes(data, peak_levels):
    #for i in range(len(peak_levels)):
    plumes = get_matrix_shapes_from_peak(data, peak_levels[0])
    return(plumes)


def detect_shapes(image_data):
    # lowering data noise
    data = image_data.copy()
    #print_image_given_matrix(image_data)
    #data = get_data_w_lowed_noise_by_mean(image_data)
    #data = get_data_w_lowed_noise(image_data)
    #print_image_given_matrix(data)
    #data = reduce_image_dimension(data, 5)
    #data = increment_image_dimension(data, 5)
    #print_image_given_matrix(data)
    #data = get_matrix_filled_w_value(image_data, 0)
    """data_tmp = sp.ndimage.filters.gaussian_filter(image_data, [10, 10], mode='constant')
    for y in range(len(data_tmp)):
        for x in range(len(data_tmp[y])):
            data[y][x] = image_data[y][x] - data_tmp[y][x]
            if data[y][x] < 0: data[y][x] = 0"""
    # getting peak levels
    #peaks = get_peaks_positions(data)
    # low pass filtering the image
    #data = get_final_interpolated_data(data)
    data = get_matrix_low_pass_filtered(data, 0.2)
    #data = sp.ndimage.filters.gaussian_filter(data, [1, 1], mode='constant')
    print_image_given_matrix(data)
    # reducing image range quality
    #data = decrease_image_precision(data, 20)
    peaks = get_peaks_positions(data)
    # getting peaks value levels
    peak_levels = []
    sys.setrecursionlimit(10000)
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
    return plumes














#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#                                   MAIN
#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
"""path_day = "../Data/NO2/Sabetta Port/images/2021/05/05/"
print_image_given_matrix(get_json_content(path_day + "balanced/", "data")["4"])
data = get_json_content(path_day + "interpolated/", "data")["4"]
#print_image_given_matrix(data)

shapes = detect_shapes(data)

points = []
for y in range(len(data)):
    for x in range(len(data[y])):
        if shapes[y][x] != None:
            points.append([y, x])
            #data[y][x] = shapes[y][x]["peak"]
        else:
            if data[y][x] < 20:
                points.append([y, x])
new_data = get_interpolated_image(points, data)
print_image_given_matrix(new_data)"""

for counter in range(10):
    i = counter
    if i!= 2:
        day_string = ""
        if i+1 <=9: day_string = "0"
        day_string += str(i+1)
        path_day = "../Data/NO2/Sabetta Port/images/2021/05/" + day_string + "/"
        print_image_given_matrix(get_json_content(path_day + "balanced/", "data")["4"])
        data = get_json_content(path_day + "interpolated/", "data")["4"]
        shapes = detect_shapes(data)
        print_image_given_plumes(shapes)

        """points = []
        for y in range(len(data)):
            for x in range(len(data[y])):
                if shapes[y][x] != None:
                    points.append([y, x])
                else:
                    if data[y][x] <= 5:
                        data[y][x] = 0
                        points.append([y, x])
        new_data = get_interpolated_image(points, data)
        print_image_given_matrix(new_data)"""


        """mean_list = {}
        tot_list = {}
        for y in range(len(data)):
            for x in range(len(data[y])):
                if shapes[y][x] != None:
                    if shapes[y][x]["id"] not in mean_list:
                        mean_list[str(shapes[y][x]["id"])] = data[y][x]
                        tot_list[str(shapes[y][x]["id"])] = 1
                    else:
                        mean_list[str(shapes[y][x]["id"])] += data[y][x]
                        tot_list[str(shapes[y][x]["id"])] += 1
                    data[y][x] = 0
        for y in range(len(data)):
            for x in range(len(data[y])):
                if shapes[y][x] != None:
                    data[y][x] = mean_list[str(shapes[y][x]["id"])] / tot_list[str(shapes[y][x]["id"])]
        data = get_interpolated_data(data, 2, "mean")
        print_image_given_matrix(data)"""
        #shapes = detect_shapes(data)














