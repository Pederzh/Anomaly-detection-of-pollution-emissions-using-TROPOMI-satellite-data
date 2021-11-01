import datetime
import io
import json
import math
import sys
from pathlib import Path

import copy
import numpy
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import skimage
from scipy.ndimage import gaussian_filter
import scipy as sp
from scipy.ndimage import rotate
from scipy import interpolate, ndimage
from scipy.interpolate import griddata, interp2d
import scipy.ndimage
import scipy.fftpack
import scipy.signal
from scipy import interpolate
from scipy.interpolate import griddata
import statistics as stst
from skimage import measure
from skimage.filters import threshold_otsu
import skimage.color
import skimage.filters
import glob

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import reconstruction
from skimage.util import random_noise
from skimage import feature

from skimage import data, io, filters, feature

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
    if value == -1 or value == None: return [0, 0, 0, 0]
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
    for y in range(len(new_matrix)):
        for x in range(len(new_matrix[y])):
            closer = 0
            for yy in range(factor):
                for xx in range(factor):
                    if y*factor+yy < len(matrix) and x*factor+xx < len(matrix[0]):
                        if abs(matrix[y*factor+yy][x*factor+xx] - new_matrix[y][x]) < abs(closer - new_matrix[y][x]):
                            closer = matrix[y*factor+yy][x*factor+xx]
            new_matrix[y][x] = closer

    return new_matrix

























def get_all_peaks(data_set, min_height, prominence):
    found_peaks = []
    for y in range(len(data_set)):
        found_peaks.append([])
        for x in range(len(data_set[y])):
            found_peaks[y].append(0)
    # PEAKS
    data_smoothed = copy.deepcopy(data_set)
    perpendiculars_peaks = []
    for i in range(len(data_smoothed[0])):
        signal = get_matrix_column(data_smoothed, i)
        signal = float_array_to_int(signal)
        peaks = list(
            scipy.signal.find_peaks(signal, height=min_height, prominence=prominence))
        perpendiculars_peaks.append(peaks)
    parallels_peaks = []
    for i in range(len(data_smoothed)):
        signal = float_array_to_int(data_smoothed[i])
        peaks = list(
            scipy.signal.find_peaks(signal, height=min_height, prominence=prominence))
        parallels_peaks.append(peaks)

    for y in range(len(data_smoothed)):
        for x in range(len(data_smoothed[y])):
            if x in parallels_peaks[y][0] and y in perpendiculars_peaks[x][0]:
                found_peaks[y][x] = 1
    return found_peaks






















def print_images_peaks(data):
    # decreasing image values precision
    #data = divide_image_in_two(data, 100)
    #data = decrease_image_precision(data, 20)
    #data = reduce_image_dimension(data,2)
    data_smoothed = data.copy()


    # smoothing the image
    sigma_y = 1
    sigma_x = 1
    sigma = [sigma_y, sigma_x]
    """data_smoothed = sp.ndimage.filters.gaussian_filter(data, sigma, mode='constant')"""
    plot_data = sp.ndimage.filters.gaussian_filter(data_smoothed, sigma, mode='constant')
    # print_image_given_matrix(data)
    # print_image_given_matrix(data_smoothed)

    freq_scan = scipy.fftpack.fft2(data_smoothed)
    keep_fraction = 0.15
    im_fft2 = freq_scan.copy()
    r, c = im_fft2.shape
    im_fft2[int(r * keep_fraction):int(r * (1 - keep_fraction))] = 0
    im_fft2[:, int(c * keep_fraction):int(c * (1 - keep_fraction))] = 0
    data_smoothed = scipy.fftpack.ifft2(im_fft2).real


    # PEAKS
    min_height = 0
    prominence = 20
    perpendiculars_peaks = []
    for i in range(len(data_smoothed[0])):
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
        parallels_peaks.append(peaks)

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
            if x in parallels_peaks[y][0]: data_par_peaks[y][x] = data_smoothed[y][x] = 2000
            if y in perpendiculars_peaks[x][0]: data_per_peaks[y][x] = data_smoothed[y][x] = 2000
            #if y in perpendiculars[x][0] or x in parallels[y][0]:
            if x in parallels_peaks[y][0] and y in perpendiculars_peaks[x][0]:
                data_smoothed[y][x] = 3000
                #data_par_peaks[y][x] = data_smoothed[y][x] = -1
                #data_per_peaks[y][x] = -1
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
            plot_data[y][x] = 0
            if data_smoothed[y][x] >= 2000: plot_data[y][x] = data[y][x]
            if data_smoothed[y][x] < 0:
                plot_data[y][x] = 0
    #print_image_given_matrix(plot_data)
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

def increment_image_dimension(matrix, factor):
    new_matrix = []
    for y in range(len(matrix)):
        for i in range(factor): new_matrix.append([])
        for x in range(len(matrix[y])):
            for i in range(factor):
                for j in range(factor):
                    new_matrix[y*factor+i].append(matrix[y][x])
    return new_matrix
























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
#data_set = get_json_content("../Data/NO2/Sabetta Port/images/2021/05/01/balanced/")["4"]
#data_set = get_json_content_w_name("../Data/NO2/Sabetta Port/images/2021/05/04/balanced/", "mean")#["4"]












#print_image_given_matrix(data_set)
"""new_data = print_images_peaks(data_set)
print_image_given_matrix(new_data)"""

"""def unify(m1, m2, weights):
    for y in range(len(m1)):
        for x in range(len(m1[y])):
            m1[y][x] = (m2[y][x] * weights[1] + m1[y][x]*weights[0]) / (weights[0]+weights[1])
    return m1

old_data = data_set.copy()
data = data_set
values = print_images_peaks(data)
values = rotate(values, angle=45, reshape=False, order=1)
values = rotate(values, angle=-45, reshape=False, order=1)
for i in range(32):
    angle = (i+1)*5
    data = rotate(old_data, angle=angle, reshape=False, order=1)
    values = unify(values, rotate(print_images_peaks(data), angle=-angle, reshape=False, order=1), [i+1, 1])
    #print_image_given_matrix(values)
print_image_given_matrix(values)"""

########################################################################

def get_linear_interpolated_image(points, data):
    all_points = []
    for y in range(len(data)):
        for x in range(len(data[y])):
            all_points.append([y, x])
    values = []
    for i in range(len(points)):
        values.append(data[points[i][0]][points[i][1]])
    values = numpy.array(values)
    new_values = griddata(points, values, all_points, method='linear')
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

    if what_to_return == "variance":
        for y in range(len(data_mean)):
            for x in range(len(data_mean[y])):
                values_list = []
                for i in range(len(list_of_data_set)):
                    if list_of_data_set[i][y][x] != -1: values_list.append(list_of_data_set[i][y][x])
                var = stst.variance(values_list)
                final_data[y][x] = var
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

def get_final_interpolation(data_set, side_lens, types):
    mean_types = types
    final_data = []
    #print_image_given_matrix(data_set)
    for y in range(len(data_set)):
        final_data.append([])
        for x in range(len(data_set[y])):
            final_data[y].append(0)
    for i in range(len(mean_types)):
        data_tmp = get_mean_of_interpolated_data(data_set, side_lens, mean_types[i])
        #print_image_given_matrix(data_tmp)
        for y in range(len(final_data)):
            for x in range(len(final_data[y])):
                final_data[y][x] += data_tmp[y][x]
    for y in range(len(final_data)):
        for x in range(len(final_data[y])):
            final_data[y][x] = final_data[y][x] / len(mean_types)
    #print_image_given_matrix(final_data)
    return final_data

def get_matrix_low_pass_filtered(data, keep_fraction):
    freq_scan = scipy.fftpack.fft2(data)
    im_fft2 = freq_scan.copy()
    r, c = im_fft2.shape
    im_fft2[int(r * keep_fraction):int(r * (1 - keep_fraction))] = 0
    im_fft2[:, int(c * keep_fraction):int(c * (1 - keep_fraction))] = 0
    signal_lowpass = scipy.fftpack.ifft2(im_fft2).real
    return signal_lowpass


def contour_data_set(data_set):
    data_set = get_final_interpolation(data_set)
    data_static = []
    for y in range(len(data_set)):
        data_static.append([])
        for x in range(len(data_set[y])):
            data_static[y].append(data_set[y][x])
    tmp_data_set = data_set.copy()
    print_image_given_matrix(data_set)
    np_data = numpy.array(data_set)
    max = np_data.max()
    new_data = []
    for y in range(len(np_data)):
        new_data.append([])
        for x in range(len(np_data[y])):
            new_data[y].append(float(float(np_data[y][x])/max))
    np_data = numpy.array(new_data)
    contours = measure.find_contours(np_data, 0.5)

    """x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
    r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))
    contours = measure.find_contours(r, 0.8)
    print(contours)
    for cont in contours:
        print(cont[:, 1])"""

    for contour in contours:
        for cc in contour:
            tmp_data_set[round(cc[0])][round(cc[1])] = -1

    print_image_given_matrix(tmp_data_set)



    def add_values_around_pixel(data, y, x, value):
        changed = []
        """if y>0 and x>0:
            if data[y-1][x-1] != -1:
                data[y-1][x-1] = value
                changed.append([y-1, x-1])"""
        if y>0:
            if data[y - 1][x] != -1:
                data[y - 1][x] = value
                changed.append([y-1, x])
        """if y>0 and x<len(data[0])-1:
            if data[y - 1][x + 1] != -1:
                data[y - 1][x + 1] = value
                changed.append([y - 1, x + 1])"""
        if x > 0:
            if data[y][x - 1] != -1:
                data[y][x - 1] = value
                changed.append([y, x-1])
        if x < len(data[0])-1:
            if data[y][x + 1] != -1:
                data[y][x + 1] = value
                changed.append([y, x+1])
        """if y<len(data)-1 and x>0:
            if data[y+1][x-1] != -1:
                data[y+1][x-1] = value
                changed.append([y+1, x-1])"""
        if y<len(data)-1:
            if data[y + 1][x] != -1:
                data[y + 1][x] = value
                changed.append([y+1, x])
        """if y<len(data)-1 and x<len(data[0])-1:
            if data[y + 1][x + 1] != -1:
                data[y + 1][x + 1] = value
                changed.append([y+1, x+1])"""
        return changed

    def add_zeroes_around_pixel(data, y, x, value):
        changed = []
        """if y>0 and x>0:
            if data[y-1][x-1] != -1:
                data[y-1][x-1] = value
                changed.append([y-1, x-1])"""
        if y>0:
            if data[y - 1][x] == -1:
                data[y - 1][x] = value
                changed.append([y-1, x])
        """if y>0 and x<len(data[0])-1:
            if data[y - 1][x + 1] != -1:
                data[y - 1][x + 1] = value
                changed.append([y - 1, x + 1])"""
        if x > 0:
            if data[y][x - 1] == -1:
                data[y][x - 1] = value
                changed.append([y, x-1])
        if x < len(data[0])-1:
            if data[y][x + 1] == -1:
                data[y][x + 1] = value
                changed.append([y, x+1])
        """if y<len(data)-1 and x>0:
            if data[y+1][x-1] != -1:
                data[y+1][x-1] = value
                changed.append([y+1, x-1])"""
        if y<len(data)-1:
            if data[y + 1][x] == -1:
                data[y + 1][x] = value
                changed.append([y+1, x])
        """if y<len(data)-1 and x<len(data[0])-1:
            if data[y + 1][x + 1] != -1:
                data[y + 1][x + 1] = value
                changed.append([y+1, x+1])"""
        return changed

    def create_shape(shapes_matrix, y, x, value):
        #print(shapes_matrix)
        if value == 0:
            changed = add_zeroes_around_pixel(shapes_matrix, y, x, value)
        else:
            changed = add_values_around_pixel(shapes_matrix, y, x, value)
        for i in range(len(changed)):
            create_shape(shapes_matrix, changed[i][0], changed[i][1], value)



    for y in range(len(data_set)):
        for x in range(len(data_set[y])):
            if data_set[y][x] == max:
                max_cc = [y, x]
    sys.setrecursionlimit(10000)
    create_shape(data_set, max_cc[0], max_cc[1], -1)
    create_shape(data_set, max_cc[0], max_cc[1], 0)
    for y in range(len(data_set)):
        for x in range(len(data_set[y])):
            if data_set[y][x] == -1: data_set[y][x] = data_static[y][x]
    print_image_given_matrix(data_set)

    tmp_data_set = data_set.copy()
    np_data = numpy.array(data_set)
    max = np_data.max()
    new_data = []
    for y in range(len(np_data)):
        new_data.append([])
        for x in range(len(np_data[y])):
            new_data[y].append(float(float(np_data[y][x])/max))
    np_data = numpy.array(new_data)
    contours = measure.find_contours(np_data, 0.5)
    for contour in contours:
        for cc in contour:
            tmp_data_set[round(cc[0])][round(cc[1])] = -1

    print_image_given_matrix(tmp_data_set)

    for y in range(len(data_set)):
        for x in range(len(data_set[y])):
            if data_set[y][x] == max:
                max_cc = [y, x]
    create_shape(data_set, max_cc[0], max_cc[1], -1)
    create_shape(data_set, max_cc[0], max_cc[1], 0)
    for y in range(len(data_set)):
        for x in range(len(data_set[y])):
            if data_set[y][x] == -1: data_set[y][x] = data_static[y][x]
    print_image_given_matrix(data_set)


def get_interpolation_variance(data_set, dimension):
    points = []
    new_data_set = copy.deepcopy(data_set)
    for y in range(len(data_set)):
        for x in range(len(data_set[y])):
            points.append([y, x])
    for y in range(len(data_set)-dimension):
        if (y + 1) % 10 == 0:
            print_image_given_matrix(new_data_set)
        for x in range(len(data_set[y])-dimension):
            not_to_take_points = []
            for yy in range(dimension):
                for xx in range(dimension):
                    not_to_take_points.append([y+yy, x+xx])
            to_take_points = points[:]
            for point in not_to_take_points:
                to_take_points.remove(point)
            #point_value = get_interpolated_image(to_take_points, data_set)
            #new_data_set[y+round(dimension/2)][x+round(dimension/2)] = point_value[y+round(dimension/2)][x+round(dimension/2)]
            if y%dimension == 0 and x%dimension == 0:
                point_value = get_interpolated_image(to_take_points, data_set)
                for point in not_to_take_points:
                    new_data_set[point[0]][point[1]] = point_value[point[0]][point[1]]
    return new_data_set


def get_all_interpolated_values(data_set, dimension, internal):

    new_data_set = []
    for yy in range(len(data_set)):
        new_data_set.append([])
        for xx in range(len(data_set[yy])):
            new_data_set[yy].append([])

    # SETTING POINTS FOR INTERPOLATION

    to_take_points = []
    for yy in range(dimension):
        for xx in range(dimension):
            to_take_points.append([yy, xx])
    not_to_take_points = []
    for yy in range(internal):
        for xx in range(internal):
            not_to_take_points.append([yy+round((dimension-internal)/2), xx+round((dimension-internal)/2)])
    for point in not_to_take_points:
        to_take_points.remove(point)

    # STARTING THE INTEROPLATION CYCLE

    for y in range(len(data_set)-dimension):
        for x in range(len(data_set[y])-dimension):
            tmp_data_set = []
            for yy in range(dimension):
                tmp_data_set.append([])
                for xx in range(dimension):
                    tmp_data_set[yy].append(data_set[y+yy][x+xx])
            intrp_data_set = get_interpolated_image(to_take_points, tmp_data_set)
            #new_data_set[y+round(dimension/2)][x+round(dimension/2)] = point_value[y+round(dimension/2)][x+round(dimension/2)]
            for yy in range(internal):
                for xx in range(internal):
                    tmp_y = yy+round((dimension-internal)/2)
                    tmp_x = xx+round((dimension-internal)/2)
                    new_data_set[y + tmp_y][x + tmp_x].append(intrp_data_set[tmp_y][tmp_x])
    return new_data_set

def get_single_interpolated_values(data_set, dimension, internal):

    new_data_set = []
    for yy in range(len(data_set)):
        new_data_set.append([])
        for xx in range(len(data_set[yy])):
            new_data_set[yy].append(-1)

    # SETTING POINTS FOR INTERPOLATION

    to_take_points = []
    for yy in range(dimension):
        for xx in range(dimension):
            to_take_points.append([yy, xx])
    not_to_take_points = []
    for yy in range(internal):
        for xx in range(internal):
            not_to_take_points.append([yy+round((dimension-internal)/2), xx+round((dimension-internal)/2)])
    for point in not_to_take_points:
        to_take_points.remove(point)

    # STARTING THE INTEROPLATION CYCLE

    for y in range(len(data_set)-dimension):
        for x in range(len(data_set[y])-dimension):
            tmp_data_set = []
            for yy in range(dimension):
                tmp_data_set.append([])
                for xx in range(dimension):
                    tmp_data_set[yy].append(data_set[y+yy][x+xx])
            intrp_data_set = get_interpolated_image(to_take_points, tmp_data_set)
            new_data_set[y+round(dimension/2)][x+round(dimension/2)] = intrp_data_set[round(dimension/2)][round(dimension/2)]
    return new_data_set

def unify_list_of_matrix(all_data_set, type):
    new_data_set = []
    for y in range(len(all_data_set)):
        new_data_set.append([])
        for x in range(len(all_data_set[y])):
            new_data_set[y].append(-1)
            if len(all_data_set[y][x]) > 0:
                if type == "median":
                    new_data_set[y][x] = stst.mode(all_data_set[y][x])
                if type == "mean":
                    new_data_set[y][x] = stst.mean(all_data_set[y][x])
                if type == "cleaned_mean":
                    all_data_set[y][x].sort()
                    new_list = []
                    for i in range(len(all_data_set[y][x])):
                        if (i+1)/len(all_data_set[y][x]) >= 0.1 and (i+1)/len(all_data_set[y][x]) <= 0.9:
                            new_list.append(all_data_set[y][x][i])
                    if len(new_list) > 0:
                        new_data_set[y][x] = stst.mean(new_list)
                if type == "variance":
                    if len(all_data_set[y][x]) > 1:
                        new_data_set[y][x] = stst.variance(all_data_set[y][x])
    return new_data_set

def get_approximated_interpolation_data_set(data_set, dimension, internal):
    all_data_set = get_all_interpolated_values(data_set, dimension, internal)
    #new_data_set = unify_list_of_matrix(all_data_set, "mean")
    #print_image_given_matrix(new_data_set)
    #new_data_set = unify_list_of_matrix(all_data_set, "median")
    #print_image_given_matrix(new_data_set)
    new_data_set = unify_list_of_matrix(all_data_set, "cleaned_mean")
    print_image_given_matrix(new_data_set)
    #new_data_set = unify_list_of_matrix(all_data_set, "variance")
    #print_image_given_matrix(new_data_set)

def fill_white_spaces(data_set):
    points = []
    for y in range(len(data_set)):
        for x in range(len(data_set[y])):
            if data_set[y][x] != -1 and data_set[y][x] != None:
                points.append([y, x])
    data_set = get_interpolated_image(points, data_set)
    return data_set

"""data_set = get_json_content("../Data/NO2/Sabetta Port/images/2021/05/05/balanced/")["4"]
data_set = fill_white_spaces(data_set)
print_image_given_matrix(data_set)
median_data_set = get_interpolated_data(data_set, 4, "median")
print_image_given_matrix(median_data_set)
get_approximated_interpolation_data_set(data_set, 15, 11)"""
#big_data_set = get_single_interpolated_values(median_data_set, 17, 11)
#print_image_given_matrix(big_data_set)




"""
var_data_set = get_interpolated_data(data_set, 3, "variance")
for y in range(len(var_data_set)):
    for x in range(len(var_data_set[y])):
        var_data_set[y][x] = round(var_data_set[y][x]*0.5)
print_image_given_matrix(var_data_set)

# getting var list
var_list = []
for y in range(len(var_data_set)-6):
    for x in range(len(var_data_set[y+3])-6):
        var_list.append(var_data_set[y+3][x+3])
var_list.sort()
var_mid = var_list[round(len(var_list)*0.75)]
var_up = var_list[round(len(var_list)*0.9)]
var_down = var_list[round(len(var_list)*0.1)]
#var_mid = (max(var_list) + min(var_list))/2

points = []
for y in range(len(data_set)):
    for x in range(len(data_set[y])):
        #if var_data_set[y][x] <= var_mid:
        if var_data_set[y][x] >= var_up or var_data_set[y][x] <= var_down:
        #if var_data_set[y][x] <= var_up and var_data_set[y][x] >= var_down:
            points.append([y, x])

median_data_set = get_interpolated_image(points, data_set)
print_image_given_matrix(median_data_set)"""


































"""peaks = get_all_peaks(data_set)
print_image_given_matrix(peaks)"""





########################################################################
########################################################################

# MEAN

########################################################################
########################################################################






date = datetime.datetime.now()
date_start = date.replace(year=2021, month=5, day=5, hour=0, minute=0, second=0, microsecond=0)
date_end = date.replace(year=2021, month=5, day=6, hour=0, minute=0, second=0, microsecond=0)




data_set = get_json_content_w_name("../Data/NO2/Sabetta Port/images/2021/05/04/balanced/", "mean")


#print_image_given_matrix(data_set)

def get_lowed(data_set):
    #new_data = ndimage.median_filter(data_set, 5)
    my_list = []
    for y in data_set:
        for x in y:
            my_list.append(x)
    thresh = threshold_otsu(data_set)
    fig, ax = plt.subplots()
    plt.imshow(thresh)
    plt.show()
    #print_image_given_matrix(new_data)
    return my_list




#sx = ndimage.sobel(data_set, axis=0, mode='constant')
#sy = ndimage.sobel(data_set, axis=1, mode='constant')
#sob_data = np.hypot(sx, sy)
#print_image_given_matrix(sx)
#print_image_given_matrix(sy)
#print_image_given_matrix(sob_data)


mean_set = []
values_set = []
median_set = []
mode_set = []
mean_cleaned_set = []
tot = 0
for y in range(len(data_set)):
    mean_set.append([])
    median_set.append([])
    values_set.append([])
    mode_set.append([])
    mean_cleaned_set.append([])
    for x in range(len(data_set[y])):
        mean_set[y].append(0)
        median_set[y].append(0)
        mean_cleaned_set[y].append(0)
        mode_set[y].append(0)
        values_set[y].append([])
for day_counter in range(int((date_end - date_start).days)):
    date = date_start + datetime.timedelta(days=day_counter)
    print("at day " + date.strftime("%Y-%m-%d"))
    directory_path = "../data/" + product_type + "/" + location_name + "/images/"
    directory_path = directory_path + date.strftime("%Y") + "/" + date.strftime("%m") + "/"
    directory_path = directory_path + date.strftime("%d") + "/"
    my_path = Path(directory_path)
    if my_path.is_dir():
        data_set = get_json_content_w_name(directory_path + "balanced/", "mean")  # ["3"]
        # interpolated_data_set = get_final_interpolation(data_set, ["median"])
        interpolated_data_set = get_lowed(data_set)
        tot += 1
        # print_image_given_matrix(interpolated_data_set)
        for y in range(len(mean_set)):
            for x in range(len(mean_set[y])):
                mean_set[y][x] = mean_set[y][x] + interpolated_data_set[y][x]
                values_set[y][x].append(interpolated_data_set[y][x])

def get_cleaned_mean(list):
    tot = 0
    value = 0
    for i in range(len(list)):
        if i/(len(list)-1) > 0.25 and i/(len(list)-1) <= 1.0:
            tot += 1
            value += list[i]
    return value/tot

global_set = []
for y in range(len(mean_set)):
    global_set.append([])
    for x in range(len(mean_set[y])):
        mean_set[y][x] = round(mean_set[y][x] / tot)
        values_set[y][x].sort()
        median_set[y][x] = stst.median(values_set[y][x])
        mean_cleaned_set[y][x] = get_cleaned_mean(values_set[y][x])
        global_set[y].append(mean_cleaned_set[y][x])
        #global_set[y].append((mean_cleaned_set[y][x] + median_set[y][x] + mean_set[y][x])/3)
        #mode_set[y][x] = stst.mode(values_set[y][x])*9

g_list = []
for y in range(len(global_set)):
    for x in range(len(global_set[y])):
        g_list.append(global_set[y][x])
max_val = max(g_list)


mult = 1
for y in range(len(mean_set)):
    for x in range(len(mean_set[y])):
        mean_set[y][x] = mean_set[y][x] * mult
        median_set[y][x] = median_set[y][x] * mult
        mean_cleaned_set[y][x] = mean_cleaned_set[y][x] * mult
        global_set[y][x] = pow(global_set[y][x]/max_val, 2) * max_val
        global_set[y][x] = global_set[y][x] * mult
print_image_given_matrix(mean_set)
print_image_given_matrix(median_set)
#print_image_given_matrix(global_set)
print_image_given_matrix(mean_cleaned_set)

peaks = get_all_peaks(median_set, 80, 0)
for y in range(len(peaks)):
    for x in range(len(peaks[y])):
        peaks[y][x] = peaks[y][x] * 1000
print_image_given_matrix(peaks)

#print_image_given_matrix(mode_set)

########################################################################
########################################################################
########################################################################
########################################################################


"""print(frequencies)
keys = list(frequencies.keys())
df_array = []
for i in range(200):
    if str(i) in keys:
        df_array.append({
            "x": i,
            "value": frequencies[str(i)],
            "name": "normal"
        })
    else:
        df_array.append({
            "x": i,
            "value": 0,
            "name": "normal"
        })

df = get_df_array_converted(df_array)
fig = px.histogram(df, x="x", y="value", color="name", barmode="overlay", nbins=int((date_end - date_start).days))
fig.show()"""

















#print_images_peaks(data)

"""for i in range(10):
    day_string = ""
    if i+1 <=9: day_string = "0"
    day_string += str(i+1)
    data = get_json_content("../Data/NO2/Sabetta Port/images/2021/05/" + day_string + "/balanced/")["4"]
    test(data)"""

#print_images_peaks(data)

"""def unify(m1, m2, weights):
    for y in range(len(m1)):
        for x in range(len(m1[y])):
            m1[y][x] = (m2[y][x] * weights[1] + m1[y][x]*weights[0]) / (weights[0]+weights[1])
    return m1

print_image_given_matrix(data)
data = reduce_image_dimension(data, 5)
data = increment_image_dimension(data, 5)
#data = sp.ndimage.filters.gaussian_filter(data, [1,1], mode='constant')
print_image_given_matrix(data)

old_data = data
values = print_images_peaks(data)
values = rotate(values, angle=45, reshape=False, order=1)
values = rotate(values, angle=-45, reshape=False, order=1)
for i in range(32):
    angle = (i+1)*5
    data = rotate(old_data, angle=angle, reshape=False, order=1)
    values = unify(values, rotate(print_images_peaks(data), angle=-angle, reshape=False, order=1), [i+1, 1])
    #print_image_given_matrix(values)


print_image_given_matrix(values)
for y in range(len(values)):
    for x in range(len(values[y])):
        if (values[y][x] >= 250): values[y][x] = 1000
        else: values[y][x] = 0
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