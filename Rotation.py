import datetime
import io
import json
import math
from pathlib import Path
import plotly.express as px
import pandas as pd


import matplotlib.pyplot as plt
from scipy.ndimage import rotate

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
    if value == -1 or value == None:
        return [0, 0, 0, 0]
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






# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#                                           MAIN
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


data_old = get_json_content("../Data/NO2/Sabetta Port/images/2021/05/05/balanced/")["4"]
print_image_given_matrix(data_old)
data = []
data = rotate(data_old, angle=45, reshape=False, order=1)
print_image_given_matrix(data)

data = rotate(data, angle=-45, reshape=False, order=1)
print_image_given_matrix(data)


print(data[6])



print(" ")
print(" ")
data = data_old

print(data[6])
