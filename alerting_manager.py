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
from numpy import polyfit
from PIL import Image
import statistics as stst
from scipy.ndimage import median_filter, median, maximum_filter, gaussian_filter
from skimage.filters import threshold_otsu
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.signal
from scipy.interpolate import griddata
from numpy import array


from statsmodels.tsa.statespace.sarimax import SARIMAX



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
        rgb[0] = 255
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

def print_image_given_matrix(matrix):
    image = create_image_from_matrix(matrix)
    plt.imshow(image)
    plt.show()

def get_json_content_w_name(directory_path, name):
    with open(directory_path + name + ".json") as json_file:
        data = json.load(json_file)
    return data













def get_parameters_mobile_mean(data_set, days_range):
    keys = list(data_set.keys())
    date = datetime.datetime.now()
    #date = date_start + datetime.timedelta(days=day_counter)
    params_list = {}
    for key in keys:
        date_str = key.split('-')
        date = date.replace(year=int(date_str[0]), month=int(date_str[1]), day=int(date_str[2]), hour=0, minute=0, second=0, microsecond=0)
        params_list[key] = {
            "date": date,
            "parameters": data_set[key]
        }

    keys.sort()
    range_parameters = []
    for key in keys:
        counter = 0
        date_start = params_list[key]["date"]
        range_parameters.append({
            "date": date_start + datetime.timedelta(days=int(days_range/2)),
            "parameters": params_list[key]["parameters"]
        })
        for day_range in range(days_range - 1):
            day_range_act = day_range + 1
            date = date_start + datetime.timedelta(days=day_range_act)
            date_str = date.strftime("%Y-%m-%d")
            if date_str in params_list.keys():
                counter += 1
                for i in range(len(range_parameters[len(range_parameters)-1]["parameters"])):
                    range_parameters[len(range_parameters) - 1]["parameters"][i] += params_list[date_str]["parameters"][i]
        if counter + 1 == days_range:
            for i in range(len(range_parameters[len(range_parameters) - 1]["parameters"])):
                range_parameters[len(range_parameters) - 1]["parameters"][i] = \
                range_parameters[len(range_parameters) - 1]["parameters"][i] / counter
        else:
            range_parameters.pop()
    return range_parameters




def get_parameters(data_set):
    keys = list(data_set.keys())
    date = datetime.datetime.now()
    keys.sort()
    params_list = []
    for key in keys:
        date_str = key.split('-')
        date = date.replace(year=int(date_str[0]), month=int(date_str[1]), day=int(date_str[2]), hour=0, minute=0,
                            second=0, microsecond=0)
        params_list.append({
            "date": date,
            "parameters": data_set[key]
        })
    return params_list



def get_parameters_mean(data_set, day_range, type):
    keys = list(data_set.keys())
    date = datetime.datetime.now()
    keys.sort()
    params_list = []
    counter = 0
    for key in keys:
        if counter == 0:
            params_list.append([])
        date_str = key.split('-')
        date = date.replace(year=int(date_str[0]), month=int(date_str[1]), day=int(date_str[2]), hour=0, minute=0,
                            second=0, microsecond=0)
        counter += 1
        params_list[len(params_list)-1].append({
            "date": date,
            "parameters": data_set[key]
        })
        if counter % day_range == 0: counter = 0
    if counter != 0: params_list.pop()
    parameters_list = []
    for par in params_list:
        params = []
        for i in range(len(par[0]["parameters"])):
            params.append([])
            for p in par: params[len(params)-1].append(p["parameters"][i])
        param = []
        for i in range(len(par[0]["parameters"])):
            param.append(0)
            if type == "mean":
                param[len(param)-1] = stst.mean(params[len(param)-1])
            if type == "cleaned_mean":
                new_params = []
                params[len(param) - 1].sort()
                for ii in range(len(params[len(param) - 1])):
                    prop = ii / (len(params[len(param) - 1])-1)
                    if prop <= 0.8 and prop >= 0.1:
                        new_params.append(params[len(param) - 1][ii])
                params[len(param) - 1] = new_params
                param[len(param)-1] = stst.mean(params[len(param)-1])
            if type == "median":
                param[len(param)-1] = stst.median(params[len(param)-1])
        obj = {
            "date": par[0]["date"],
            "parameters": param
        }
        parameters_list.append(obj)
    return parameters_list

















# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#                                  LINEAR REGRESSION
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def linear_regression_test(data_set_mean, target):
    df_array = []
    for i in range(len(data_set_mean)):
        data = {
            "target": data_set_mean[i]["parameters"][target],
            "year": data_set_mean[i]["date"].year,
            "day": data_set_mean[i]["date"].day,
        }
        for i in range(12):
            data["month_" + str(i+1)] = 0
        for i in range(7):
            data["day_of_week" + str(i+1)] = 0
        data["month_" + str(data_set_mean[i]["date"].month)] = 1
        data["day_of_week" + str(data_set_mean[i]["date"].weekday())] = 1
        df_array.append(data)
    df = pd.DataFrame(df_array)

    # splitting in target and parameters
    y = df["target"]
    x = df.drop("target", axis=1)

    # splitting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.05, random_state = 210)

    # fitting the training data
    LR = LinearRegression()
    LR.fit(x_train, y_train)
    y_prediction = LR.predict(x_test)
    print(y_prediction)
    print(list(y_test))












# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#             SARIMA Seasonal Autoregressive Integrated Moving-Average
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def SARIMA_test(data_set_mean, target, d_order, s_order):

    y = []
    x = []
    for i in range(len(data_set_mean)):
        y.append(data_set_mean[i]["parameters"][target])
        x.append(i)

    plt.scatter(x, y)
    plt.show()

    for i in range(10):
        yy = []
        pos = []
        yyy = 0
        len_train = i + 158
        for i in range(len(y) - len_train):
            if i < len(y) - len_train - 1:
                yy.append(y[i])
                pos.append(i)
            else:
                yyy = y[i]
        model = SARIMAX(yy, order=(d_order[0], d_order[1], d_order[2]), seasonal_order=(s_order[0], s_order[1], s_order[2], s_order[3]))
        model_fit = model.fit(disp=False)
        # make prediction
        yhat = model_fit.predict(len(yy), len(yy))
        print("prediction at pos " + str(len(yy)))
        print("prediction: " + str(yhat))
        print("actual value: " + str(yyy))
        print(" ")








# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#                           POLYNOMIAL REGRESSION
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def polynomial_regression_test(data_set, target, order):
    df_array = []
    for i in range(len(data_set)):
        data = {
            "target": data_set[i]["parameters"][target],
            "position": i,
        }
        df_array.append(data)
    df = pd.DataFrame(df_array)

    # splitting in target and parameters
    y = df["target"]
    x = df.drop("target", axis=1)
    x = df["position"]

    # splitting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.05, random_state = 210)

    mymodel = numpy.poly1d(numpy.polyfit(x, y, order))
    myline = numpy.linspace(0, len(data_set), 100)
    plt.scatter(x, y)
    plt.plot(myline, mymodel(myline))
    plt.show()













# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#                                       MAIN
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


data_set = get_json_content_w_name("../Data/NO2/Sabetta Port/range_data/30/gaussian_shapes/peak_2/", "parameters")
old_data_set = get_json_content_w_name("../Data/NO2/Sabetta Port/range_data/30/gaussian_shapes/peak_2/", "old_parameters")
prop_2020_to_2021 = 2.4

data_set_mean_mobile = get_parameters_mobile_mean(data_set, 11)
data_set_normal = get_parameters(data_set)
data_set_mean = get_parameters_mean(data_set, 10, "cleaned_mean")

data = data_set_normal
for i in range(len(data)):
    data[i]["parameters"][0] = data[i]["parameters"][2] / math.pi


new_data = []
date = datetime.datetime.now()
date_start = date.replace(year=2020, month=3, day=1, hour=0, minute=0, second=0, microsecond=0)
date_end = date.replace(year=2021, month=10, day=1, hour=0, minute=0, second=0, microsecond=0)
for day_counter in range(int((date_end - date_start).days)):
    date = date_start + datetime.timedelta(days=day_counter)
    date_str = date.strftime("%Y-%m-%d")
    if date_str in data_set:
        new_data.append({
            "parameters": data_set[date_str]
        })
    if date_str in old_data_set:
        old_data_set[date_str][0] = old_data_set[date_str][0] * prop_2020_to_2021
        old_data_set[date_str][2] = old_data_set[date_str][2] * prop_2020_to_2021
        new_data.append({
            "parameters": old_data_set[date_str]
        })
    if date_str not in data_set and date_str not in old_data_set:
        new_data.append({
            "parameters": [np.nan, np.nan, np.nan]
        })

old_data_set_mean = get_parameters_mean(old_data_set, 10, "cleaned_mean")

tot_data

#linear_regression_test(data, 2)
SARIMA_test(new_data, 2, [1, 1, 0], [0, 0, 0, 0])
#polynomial_regression_test(old_data_set_mean, 2, 7)

# SARIMAX
# LSTM






