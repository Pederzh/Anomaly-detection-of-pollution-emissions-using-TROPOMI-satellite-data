import datetime
import io
import json
import math
from pathlib import Path

import copy
import numpy
import sys
import pandas as pd
import scipy.ndimage
import scipy.fftpack
import scipy.signal
import scipy.stats as st
import numpy as np
from PIL import Image
import statistics as stst
from scipy.ndimage import median_filter, median, maximum_filter, gaussian_filter
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import scipy.signal
from scipy.interpolate import griddata
from numpy import array





min_peak_height = 33.3
og_min_peaks_distance = 0.3
min_peaks_distance = 0.3
min_shape_border = 0.5
distance_level = 30



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#                               SHAPES FUNCTIONS
#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

"""min_peaks_distance = 0.25
min_shape_border = 0.5"""


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
    max_peak["prop_height_from_top"] = (level+10) / max_peak["peak"]
    for y in range(len(shapes)):
        for x in range(len(shapes[y])):
            if shapes[y][x] == shape_id:
                if unify:
                    found_shapes[y][x] = max_peak
                else:
                    if found_shapes[y][x] != None:
                        if "prop_height_from_top" not in found_shapes[y][x].keys():
                            found_shapes[y][x]["prop_height_from_top"] = (level+10) / found_shapes[y][x]["peak"]
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
                if data[y][x]["id"] == -1: matrix[y].append(data[y][x]["id"])
                else:
                    if data[y][x]["id"] == 1: matrix[y].append(1000)
                    else: matrix[y].append(200)
                    #matrix[y].append(data[y][x]["id"] * 50)
    print_image_given_matrix(matrix)

def get_matrix_shapes_from_peak(data, peak_level):
    level = peak_level
    found_shapes = get_matrix_filled_w_value(data, None)
    while level >= 0: #peak_level*0.5:
        cutted_data = cut_matrix_values_in_half(data, level)
        shapes_data = differentiate_shapes(cutted_data)
        unify_shapes(found_shapes, shapes_data, level)
        level = level - 10
    return found_shapes



































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

def save_json(directory_path, json_file, json_file_name):
    Path(directory_path).mkdir(parents=True, exist_ok=True)
    with open(directory_path + json_file_name + ".json", 'w') as outfile:
        json.dump(json_file, outfile)

def save_image(directory_path, file_name, rgbt_matrix):
    Path(directory_path).mkdir(parents=True, exist_ok=True)
    array = np.array(rgbt_matrix, dtype=np.uint8)
    image = Image.fromarray(array)
    image.save(directory_path + file_name + ".png", format="png")


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

def print_image_given_matrix(matrix):
    image = create_image_from_matrix(matrix)
    plt.imshow(image)
    plt.show()

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

def get_matrix_low_pass_filtered(data, keep_fraction):
    freq_scan = scipy.fftpack.fft2(data)
    im_fft2 = freq_scan.copy()
    r, c = im_fft2.shape
    im_fft2[int(r * keep_fraction):int(r * (1 - keep_fraction))] = 0
    im_fft2[:, int(c * keep_fraction):int(c * (1 - keep_fraction))] = 0
    signal_lowpass = scipy.fftpack.ifft2(im_fft2).real
    return signal_lowpass

















def get_lowed_image(data_set, data_range):
    values_set = []
    freq_set = []
    lowed = 0
    for row in data_set:
        for value in row:
            values_set.append(value)
    min_val = min(values_set)
    max_val = max(values_set)
    tot_freq = (max_val - min_val) / data_range
    for i in range(int(tot_freq)+1):
        freq_set.append(0)
    for value in values_set:
        freq_set[int((value - min_val) / data_range)] += 1
    max_freq = max(freq_set)
    for i in range(len(freq_set)):
        if freq_set[i] == max_freq:
            lowed = min_val + (i) * data_range
            break
    new_data_set = []
    for y in range(len(data_set)):
        new_data_set.append([])
        for x in range(len(data_set[y])):
            #new_data_set[y].append(0)
            #if data_set[y][x] > lowed: new_data_set[y][x] = data_set[y][x]
            new_data_set[y].append(data_set[y][x] - lowed)
            if new_data_set[y][x] < 0: new_data_set[y][x] = 0
    return new_data_set

def get_lowed_image_proprtionally(data_set, data_range):
    values_set = []
    freq_set = []
    lowed = 0
    for row in data_set:
        for value in row:
            values_set.append(value)
    min_val = min(values_set)
    max_val = max(values_set)
    tot_freq = (max_val - min_val) / data_range
    for i in range(int(tot_freq) + 1):
        freq_set.append(0)
    for value in values_set:
        freq_set[int((value - min_val) / data_range)] += 1
    max_freq = max(freq_set)
    for i in range(len(freq_set)):
        if freq_set[i] == max_freq:
            lowed = min_val + (i) * data_range
            break
    new_data_set = []
    print(lowed)
    for y in range(len(data_set)):
        new_data_set.append([])
        for x in range(len(data_set[y])):
            # new_data_set[y].append(0)
            # if data_set[y][x] > lowed: new_data_set[y][x] = data_set[y][x]
            new_data_set[y].append(data_set[y][x] * 50 / lowed - 50)
            if new_data_set[y][x] < 0: new_data_set[y][x] = 0
    return new_data_set


































def get_value_from_rgb( rgb):
    """
    [minVal, [0, 0, 0.5]],
    [minVal + 0.125 * diff, [0, 0, 1]],
    [minVal + 0.375 * diff, [0, 1, 1]],
    [minVal + 0.625 * diff, [1, 1, 0]],
    [minVal + 0.875 * diff, [1, 0, 0]],
    [maxVal, [0.5, 0, 0]]
    """
    if rgb[3] == 0: return -1
    precision = 1016  # given by the rgb composition
    minVal = 0
    maxVal = minVal + precision
    diff = maxVal - minVal
    # [0, 0, 1]
    if rgb[0] == 0 and rgb[1] == 0:
        return minVal + 0.125 * diff * (rgb[2]-128)/127
    # [0, 1, 1]
    if rgb[0] == 0 and rgb[2] == 255:
        min = 0.125
        max = 0.375
        rangeVal = max - min
        return (minVal + min * diff) + rangeVal * diff * rgb[1]/255
    # [1, 1, 0]
    if rgb[1] == 255:
        min = 0.375
        max = 0.625
        rangeVal = max - min
        return (minVal + min * diff) + rangeVal * diff * rgb[0] / 255
    # [1, 0, 0]
    if rgb[0] == 255:
        min = 0.625
        max = 0.875
        rangeVal = max - min
        return (minVal + min * diff) + rangeVal * diff * (255-rgb[1]) / 255
    # [0.5, 0, 0]
    min = 0.875
    max = 1
    rangeVal = max - min
    return (minVal + min * diff) + rangeVal * diff * (127-(rgb[0]-128)) / 127


def create_image_matrix(image):
    image_matrix = [[]]
    image_matrix.clear()
    for y in range(len(image)):
        image_matrix.append([])
        for x in range(len(image[y])):
            if image[y][x][3] == 0: image_matrix[y].append(-1)
            else: image_matrix[y].append(int(get_value_from_rgb(image[y][x])))
    return image_matrix






def is_peak(point, data_set):
    y = point[0]
    x = point[1]
    if data_set[y - 1][x]["value"] > data_set[y][x]["value"] or \
            data_set[y + 1][x]["value"] > data_set[y][x]["value"] or \
            data_set[y - 1][x - 1]["value"] > data_set[y][x]["value"] or \
            data_set[y][x - 1]["value"] > data_set[y][x]["value"] or \
            data_set[y + 1][x - 1]["value"] > data_set[y][x]["value"] or \
            data_set[y - 1][x + 1]["value"] > data_set[y][x]["value"] or \
            data_set[y][x + 1]["value"] > data_set[y][x]["value"] or \
            data_set[y + 1][x + 1]["value"] > data_set[y][x]["value"]:
        data_set[y][x]["is_peak"] = -1
        return
    if data_set[y - 1][x]["value"] < data_set[y][x]["value"] and \
            data_set[y + 1][x]["value"] < data_set[y][x]["value"] and \
            data_set[y - 1][x - 1]["value"] < data_set[y][x]["value"] and \
            data_set[y][x - 1]["value"] < data_set[y][x]["value"] and \
            data_set[y + 1][x - 1]["value"] < data_set[y][x]["value"] and \
            data_set[y - 1][x + 1]["value"] < data_set[y][x]["value"] and \
            data_set[y][x + 1]["value"] < data_set[y][x]["value"] and \
            data_set[y + 1][x + 1]["value"] < data_set[y][x]["value"]:
        data_set[y][x]["is_peak"] = 1
        return
    return

def set_neighbors_value(point, data_set):
    y = point[0]
    x = point[1]
    yy = y - 1
    xx = x
    if data_set[yy][xx]["value"] == data_set[y][x]["value"] and data_set[yy][xx]["is_peak"] == 0:
        data_set[yy][xx]["is_peak"] = data_set[y][x]["is_peak"]
        set_neighbors_value([yy, xx], data_set)
    yy = y + 1
    xx = x
    if data_set[yy][xx]["value"] == data_set[y][x]["value"] and data_set[yy][xx]["is_peak"] == 0:
        data_set[yy][xx]["is_peak"] = data_set[y][x]["is_peak"]
        set_neighbors_value([yy, xx], data_set)
    yy = y - 1
    xx = x - 1
    if data_set[yy][xx]["value"] == data_set[y][x]["value"] and data_set[yy][xx]["is_peak"] == 0:
        data_set[yy][xx]["is_peak"] = data_set[y][x]["is_peak"]
        set_neighbors_value([yy, xx], data_set)
    yy = y
    xx = x - 1
    if data_set[yy][xx]["value"] == data_set[y][x]["value"] and data_set[yy][xx]["is_peak"] == 0:
        data_set[yy][xx]["is_peak"] = data_set[y][x]["is_peak"]
        set_neighbors_value([yy, xx], data_set)
    yy = y + 1
    xx = x - 1
    if data_set[yy][xx]["value"] == data_set[y][x]["value"] and data_set[yy][xx]["is_peak"] == 0:
        data_set[yy][xx]["is_peak"] = data_set[y][x]["is_peak"]
        set_neighbors_value([yy, xx], data_set)
    yy = y - 1
    xx = x + 1
    if data_set[yy][xx]["value"] == data_set[y][x]["value"] and data_set[yy][xx]["is_peak"] == 0:
        data_set[yy][xx]["is_peak"] = data_set[y][x]["is_peak"]
        set_neighbors_value([yy, xx], data_set)
    yy = y
    xx = x + 1
    if data_set[yy][xx]["value"] == data_set[y][x]["value"] and data_set[yy][xx]["is_peak"] == 0:
        data_set[yy][xx]["is_peak"] = data_set[y][x]["is_peak"]
        set_neighbors_value([yy, xx], data_set)
    yy = y + 1
    xx = x + 1
    if data_set[yy][xx]["value"] == data_set[y][x]["value"] and data_set[yy][xx]["is_peak"] == 0:
        data_set[yy][xx]["is_peak"] = data_set[y][x]["is_peak"]
        set_neighbors_value([yy, xx], data_set)
    return


def two_dimensional_peak_finding(data_set):
    sys.setrecursionlimit(10000)
    data_and_peaks = []
    for y in range(len(data_set)):
        data_and_peaks.append([])
        for x in range(len(data_set[y])):
            data_and_peaks[y].append(
                {"value": data_set[y][x],
                 "is_peak": 0,
                })
    for y in range(len(data_and_peaks)):
        for x in range(len(data_and_peaks[y])):
            if y>0 and y<len(data_and_peaks)-1 and x>0 and x<len(data_and_peaks[y])-1:
                is_peak([y, x], data_and_peaks)
            else:
                data_and_peaks[y][x]["is_peak"] = -1
    for y in range(len(data_and_peaks)):
        for x in range(len(data_and_peaks[y])):
            if y>0 and y<len(data_and_peaks)-1 and x>0 and x<len(data_and_peaks[y])-1:
                if data_and_peaks[y][x]["is_peak"] == -1:
                    set_neighbors_value([y, x], data_and_peaks)
    for y in range(len(data_and_peaks)):
        for x in range(len(data_and_peaks[y])):
            if y>0 and y<len(data_and_peaks)-1 and x>0 and x<len(data_and_peaks[y])-1:
                if data_and_peaks[y][x]["is_peak"] == 0:
                    data_and_peaks[y][x]["is_peak"] = 1
                    set_neighbors_value([y, x], data_and_peaks)

    data_tmp = []
    peaks = []
    for y in range(len(data_and_peaks)):
        data_tmp.append([])
        for x in range(len(data_and_peaks[y])):
            if data_and_peaks[y][x]["is_peak"] == 0:
                data_tmp[y].append(200)
            else:
                if data_and_peaks[y][x]["is_peak"] == 1:
                    if data_and_peaks[y][x]["value"] >= min_peak_height:
                        peaks.append([y, x])
                        data_tmp[y].append(1000)
                    else:
                        data_tmp[y].append(200)
                else:
                    data_tmp[y].append(0)
    return peaks







def calculate_means(date_start, date_end):

    mean_set = []
    tot = 0
    for day_counter in range(int((date_end - date_start).days)):
        date = date_start + datetime.timedelta(days=day_counter)
        directory_path = "../data/" + product_type + "/" + location_name + "/images/"
        directory_path = directory_path + date.strftime("%Y") + "/" + date.strftime("%m") + "/"
        directory_path = directory_path + date.strftime("%d") + "/" + "lowed" + "/"
        my_path = Path(directory_path)
        if my_path.is_dir():
            data_set = get_json_content_w_name(directory_path, "mean")
            data_set = maximum_filter(data_set, 5)
            tot += 1
            for y in range(len(data_set)):
                if len(mean_set) < len(data_set): mean_set.append([])
                for x in range(len(data_set[y])):
                    if len(mean_set[y]) < len(data_set[y]): mean_set[y].append(data_set[y][x])
                    else: mean_set[y][x] = mean_set[y][x] + data_set[y][x]
    if tot > 0:
        for y in range(len(mean_set)):
            for x in range(len(mean_set[y])):
                mean_set[y][x] = round(mean_set[y][x] / tot)
        return mean_set
    else:
        return None

def save_range_json(product_type, location_name, date_global_start, date_global_end, data_range):
    months_set = {}
    for day_counter in range(int((date_global_end - date_global_start).days)-data_range):
        date_start = date_global_start + datetime.timedelta(days=day_counter)
        date_end = date_start + datetime.timedelta(days=data_range)
        print("at day " + date_start.strftime("%Y-%m-%d"))
        month_set = calculate_means(date_start, date_end)
        if month_set != None:
            months_set[date_start.strftime("%Y-%m-%d")] = month_set

    directory_path = "../data/" + product_type + "/" + location_name + "/range_data/" + str(data_range) + "/"
    save_json(directory_path, months_set, "data")

def create_images_from_json(directory_path, file_name):
    my_path = Path(directory_path)
    if my_path.is_dir():
        my_file = Path(directory_path + file_name + ".json")
        if my_file.is_file():
            months_set = get_json_content_w_name(directory_path, file_name)
            directory_path = directory_path + "images/"
            keys = list(months_set.keys())
            for kk in keys:
                print("at day " + kk)
                rgb_matrix  = create_image_from_matrix(months_set[kk])
                save_image(directory_path, kk, rgb_matrix)






def gauss_value(parameters, point):
    A = parameters[0]
    B = parameters[1]
    y = point[0]
    x = point[1]
    return (A * pow(math.e, -B * (pow(x, 2) + pow(y, 2))))

def get_gaussian_parameters(data_set, plumes, plume_id):
    volume = 0
    area = 0
    max_value = 0
    prop = 0

    for y in range(len(data_set)):
        for x in range(len(data_set[y])):
            if plumes[y][x] != None:
                if plumes[y][x]["id"] == plume_id:
                    volume += data_set[y][x]
                    area += 1
                    if data_set[y][x] > max_value:
                        prop = plumes[y][x]["prop_height_from_top"]
                        max_value = data_set[y][x]
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # FORMULA: A * e^(-Bx^2)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    A = max_value
    if prop > 0:
        Xb = math.sqrt(area) / math.sqrt(math.pi)
        B = - math.log(prop, math.e) / pow(Xb, 2)
    else:
        B = A * math.pi / volume
    """x = 5
    y = 5
    print("prop: " + str(prop))
    print("volume: " + str(volume))
    print("area: " + str(area))
    print("max: " + str(max_value))
    print("Xb: " + str(Xb))
    print("A: " + str(A))
    print("B: " + str(B))
    print(A * pow(math.e, -B * (x*x + y*y)))"""
    return [A, B, volume]


def create_gaussian_image(image, parameters, point):
    gaussian_image = []
    for y in range(len(image)):
        gaussian_image.append([])
        for x in range(len(image[y])):
            gauss_point = [
                abs(point[0] - y),
                abs(point[1] - x)
            ]
            gaussian_image[y].append(round(gauss_value(parameters, gauss_point)))
    return gaussian_image







def get_point_plume_id(plumes, point):
    y = point[0]
    x = point[1]
    if plumes[y][x] != None: return plumes[y][x]["id"]
    diameter = 10
    plume_id = 0
    plume_prop_h_from_top = 100
    plume_distance = 100
    for yy in range(diameter):
        for xx in range(diameter):
            y_tmp = y - (yy - int(diameter / 2))
            x_tmp = x - (xx - int(diameter / 2))
            if y_tmp > 0 and x_tmp > 0 and y_tmp < len(plumes) and x_tmp < len(plumes[y]):
                distance = round(pow((pow(y - y_tmp, 2) + pow(x - x_tmp, 2)), 0.5))
                if plumes[y_tmp][x_tmp] != None:
                    if distance == plume_distance:
                        if plume_prop_h_from_top < plumes[y_tmp][x_tmp]["prop_height_from_top"]:
                            plume_id = plumes[y_tmp][x_tmp]["id"]
                            plume_distance = distance
                            plume_prop_h_from_top = plumes[y_tmp][x_tmp]["prop_height_from_top"]
                    if distance < plume_distance:
                        plume_id = plumes[y_tmp][x_tmp]["id"]
                        plume_distance = distance
                        plume_prop_h_from_top = plumes[y_tmp][x_tmp]["prop_height_from_top"]
    return plume_id

def get_single_plume(data_set, plumes, point_id):
    single_plume = []
    list_interp = []
    for y in range(len(plumes)):
        single_plume.append([])
        for x in range(len(plumes[y])):
            single_plume[y].append(-1)
            if plumes[y][x] == None:
                if data_set[y][x] == 0:
                    single_plume[y][x] = 0
                    list_interp.append([y, x])
            else:
                list_interp.append([y, x])
                if plumes[y][x]["id"] == point_id:
                    single_plume[y][x] = data_set[y][x]
                else:
                    single_plume[y][x] = 0
    #print_image_given_matrix(single_plume)
    single_plume = get_interpolated_image(list_interp, single_plume)
    #print_image_given_matrix(single_plume)
    plume = []
    for y in range(len(single_plume)):
        plume.append([])
        for x in range(len(single_plume[y])):
            plume[y].append(None)
            if round(single_plume[y][x]) > 0:
                plume[y][x] = {
                    "id": point_id,
                    "prop_height_from_top": 0
                }
    return plume

def get_image_gaussian_plume_given_poi(data_set, plumes, point, format):
    point_id = get_point_plume_id(plumes, point)
    print_plume = []
    for y in range(len(plumes)):
        print_plume.append([])
        for x in range(len(plumes[y])):
            print_plume[y].append(plumes[y][x])
            if plumes[y][x] != None:
                if plumes[y][x]["id"] != point_id:
                    print_plume[y][x] = None
    #print_image_given_plumes(print_plume)
    if point_id != 0:
        plume = get_single_plume(data_set, plumes, point_id)
        params = get_gaussian_parameters(data_set, plume, point_id)
        if format == "image": gauss_plume = create_gaussian_image(data_set, params, point)
    else:
        params = [0, 0, 0]
        if format == "image":
            gauss_plume = []
            for y in range(len(data_set)):
                gauss_plume.append([])
                for x in range(len(data_set[y])):
                    gauss_plume[y].append(0)
    if format == "image": return gauss_plume
    if format == "parameters": return params

def get_image_gaussian_plumes(data_set, peaks, format):
    #print_image_given_matrix(data_set)
    gaussian_shapes = {}
    max_val = 0
    for elems in data_set:
        for val in elems:
            if val > max_val: max_val = val
    #min_peaks_distance = og_min_peaks_distance * max_val / 400
    plumes = get_matrix_shapes_from_peak(data_set, max_val)
    #print_image_given_plumes(plumes)
    for peak in peaks:
        image_set = get_image_gaussian_plume_given_poi(data_set, plumes, peak["point"], format)
        gaussian_shapes[str(peak["id"])] = image_set
        #print_image_given_matrix(image_set)
    return gaussian_shapes



def get_all_images_plumes(directory_path, additional_peaks_path, additional_images_path, date_start, date_end):

    peaks = get_json_content_w_name(directory_path + additional_peaks_path + "peaks/", "peaks")
    gaussian_shape_list = {}

    for day_counter in range(int((date_end - date_start).days)):
        date = date_start + datetime.timedelta(days=day_counter)
        print("at day " + date.strftime("%Y-%m-%d"))
        directory_img_path = "../data/" + product_type + "/" + location_name + "/images/"
        directory_img_path = directory_img_path + date.strftime("%Y") + "/" + date.strftime("%m") + "/"
        directory_img_path = directory_img_path + date.strftime("%d") + "/" + "balanced" + "/"
        my_path = Path(directory_img_path)
        if my_path.is_dir():
            my_file = Path(directory_img_path + "mean" + ".json")
            if my_file.is_file():
                data_set = get_json_content_w_name(directory_img_path, "mean")
                data_set = get_lowed_image(data_set, 10)
                gaussian_shapes = get_image_gaussian_plumes(data_set, peaks, "image")
                for peak in peaks:
                    if str(peak["id"]) not in gaussian_shape_list.keys():
                        gaussian_shape_list[str(peak["id"])] = {}
                    gaussian_shape_list[str(peak["id"])][date.strftime("%Y-%m-%d")] = gaussian_shapes[str(peak["id"])]
                    #gaussian_shape_list[str(peak["id"])].append(gaussian_shapes[str(peak["id"])])

    for peak in peaks:
        save_json(directory_path + additional_peaks_path + "gaussian_shapes/" + "peak_" + str(peak["id"]) +"/", gaussian_shape_list[str(peak["id"])], "data")
    for peak in peaks:
        create_images_from_json(directory_path + additional_peaks_path + "gaussian_shapes/" + "peak_" + str(peak["id"]) +"/", "data")













def create_peaks_id(directory_path, file_name):
    peaks_matrix = get_json_content_w_name(directory_path, file_name)
    peaks = []
    counter_id = 0
    for y in range(len(peaks_matrix)):
        for x in range(len(peaks_matrix[y])):
            if peaks_matrix[y][x] != 0:
                counter_id += 1
                peaks.append({"id": counter_id, "point": [y, x]})
    save_json(directory_path, peaks, "peaks")






















def get_all_parameters_plumes(directory_path, additional_peaks_path, additional_images_path, date_start, date_end):

    peaks = get_json_content_w_name(directory_path + additional_peaks_path + "peaks/", "peaks")
    gaussian_shape_list = {}
    f_name = ""
    if date_start.year < 2021: f_name = "old_"
    for day_counter in range(int((date_end - date_start).days)):
        date = date_start + datetime.timedelta(days=day_counter)
        print("at day " + date.strftime("%Y-%m-%d"))
        directory_img_path = "../data/" + product_type + "/" + location_name + "/images/"
        directory_img_path = directory_img_path + date.strftime("%Y") + "/" + date.strftime("%m") + "/"
        directory_img_path = directory_img_path + date.strftime("%d") + "/" + "balanced" + "/"
        my_path = Path(directory_img_path)
        if my_path.is_dir():
            my_file = Path(directory_img_path + "mean" + ".json")
            if my_file.is_file():
                data_set = get_json_content_w_name(directory_img_path, "mean")
                data_set = get_lowed_image(data_set, 10)
                gaussian_shapes = get_image_gaussian_plumes(data_set, peaks, "parameters")
                for peak in peaks:
                    if str(peak["id"]) not in gaussian_shape_list.keys():
                        gaussian_shape_list[str(peak["id"])] = {}
                    gaussian_shape_list[str(peak["id"])][date.strftime("%Y-%m-%d")] = gaussian_shapes[str(peak["id"])]
                    #gaussian_shape_list[str(peak["id"])].append(gaussian_shapes[str(peak["id"])])

    for peak in peaks:
        save_json(directory_path + additional_peaks_path + "gaussian_shapes/" + "peak_" + str(peak["id"]) +"/", gaussian_shape_list[str(peak["id"])], f_name + "parameters")














# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#                                   FOR CALCULATING MEAN JSON
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def create_dir_mean_from_json(directory_path):
    my_file = Path(directory_path + "data.json")
    mean_matrix = []
    tot_matrix = []
    if my_file.is_file():
        data_set = get_json_content(directory_path)
        keys = list(data_set.keys())
        for j in range(round(len(keys)*0.2)):
            i = j + round(len(keys)*0.8)
            for y in range(len(data_set[keys[i]])):
                if len(mean_matrix) == y:
                    mean_matrix.append([])
                    tot_matrix.append([])
                for x in range(len(data_set[keys[i]][y])):
                    if len(mean_matrix[y]) == x:
                        mean_matrix[y].append(0)
                        tot_matrix[y].append(0)
                    if data_set[keys[i]] != -1:
                        mean_matrix[y][x] += data_set[keys[i]][y][x]
                        tot_matrix[y][x] += 1
        for y in range(len(mean_matrix)):
            for x in range(len(mean_matrix[y])):
                mean_matrix[y][x] = round(mean_matrix[y][x] / tot_matrix[y][x])
        return mean_matrix
        #save_json(directory_path, mean_matrix, "mean")

def create_all_mean_json(product_type, location_name, peak):
    directory_path = "../data/" + product_type + "/" + location_name + "/range_data/30/gaussian_shapes/"
    directory_path = directory_path + "peak_" + str(peak) + "/"
    my_path = Path(directory_path)
    if my_path.is_dir():
        return create_dir_mean_from_json(directory_path)





# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#                       MAIN
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

values = {
    "product_types": ["NO2", "CO", "CH4", "SO2"],
    "locations_name": ["Bering Strait", "Sabetta Port"],
    "minQas": ["high", "all"],
    "image_types": ["unprocessed", "balanced"]
}

sys.setrecursionlimit(10000)
product_type = values["product_types"][0]
location_name = values["locations_name"][1]
minQa = values["minQas"][1]
image_type = values["image_types"][1]

date = datetime.datetime.now()
date_start = date.replace(year=2021, month=10, day=1, hour=0, minute=0, second=0, microsecond=0)
date_end = date.replace(year=2021, month=11, day=1, hour=0, minute=0, second=0, microsecond=0)
data_range = 30

directory_path = "../data/" + product_type + "/" + location_name + "/"
additional_peaks_path = "range_data/" + str(data_range) + "/"
additional_images_path = "images/"

#save_range_json(product_type, location_name, date_start, date_end, data_range)

#create_images_from_json(directory_path, "data")

#create_peaks_id(directory_path + additional_peaks_path + "peaks/", "data")

#get_all_images_plumes(directory_path, additional_peaks_path, additional_images_path, date_start, date_end)

get_all_parameters_plumes(directory_path, additional_peaks_path, additional_images_path, date_start, date_end)

#mean = create_all_mean_json(product_type, location_name, 3)
#print_image_given_matrix(mean)



#data_set = get_json_content_w_name(directory_path+"images/2021/05/07/balanced/", "data")["4"]
#print_image_given_matrix(data_set)
#data_set = get_lowed_image(data_set, 10)

#data_set = get_matrix_low_pass_filtered(data_set, 0.2)
#get_image_gaussian_plumes(data_set, [{"id": 3, "point": [65, 80]}], "image")
#get_image_gaussian_plumes(data_set, [{"id": 3, "point": [49, 49]}], "image")

"""data_set = get_json_content_w_name(directory_path+"images/2021/05/03/balanced/", "mean")
data_set = get_lowed_image(data_set, 10)
print_image_given_matrix(data_set)
end_set = get_image_gaussian_plumes(data_set, get_json_content_w_name(directory_path + additional_peaks_path + "peaks/", "peaks"))
print_image_given_matrix(end_set["2"])
print_image_given_matrix(end_set["3"])"""
"""data_set = get_matrix_low_pass_filtered(data_set, 0.075)
print_image_given_matrix(data_set)
#data_set = get_lowed_image(data_set, 10)
#print_image_given_matrix(data_set)
end_set = get_image_gaussian_plumes(data_set, get_json_content_w_name(directory_path + additional_peaks_path + "peaks/", "peaks"))
print_image_given_matrix(end_set["2"])
print_image_given_matrix(end_set["3"])"""