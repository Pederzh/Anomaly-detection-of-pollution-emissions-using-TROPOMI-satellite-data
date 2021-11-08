import datetime
import io
import json
import math
from pathlib import Path

import copy
import numpy
import sys
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from tqdm import tqdm_notebook
from itertools import product
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




def get_json_content_w_name(directory_path, name):
    with open(directory_path + name + ".json") as json_file:
        data = json.load(json_file)
    return data









def gauss_value(parameters, point):
    A = parameters[0]
    B = parameters[1]
    y = point[0]
    x = point[1]
    return (A * pow(math.e, -B * (pow(x, 2) + pow(y, 2))))

def get_gaussian_parameters(volume):
    A = pow(volume, 2 / 3) / pow(math.pi, 2 / 3)
    B = 1 / (pow(A, 1 / 2))
    return [A, B, volume]

def create_gaussian_image_w_parameters(image, parameters, point):
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

def create_gaussian_image(image, volume, point):
    parameters = get_gaussian_parameters(volume)
    return create_gaussian_image_w_parameters(image, parameters, point)


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
#             SARIMA Seasonal Autoregressive Integrated Moving-Average
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def SARIMA_test(data_set_mean, target, d_order, s_order, day_range_prediction):

    y = []
    x = []
    for i in range(len(data_set_mean)):
        y.append(data_set_mean[i]["parameters"][target])
        x.append(i)


    yy = []
    pos = []
    yyy = []
    len_train = day_range_prediction
    for j in range(len(y)):
        if j < len(y) - len_train - 1:
            yy.append(y[j])
            pos.append(j)
        else:
            if len(yyy) < len_train:
                yyy.append(y[j])
    model = SARIMAX(yy, order=(d_order[0], d_order[1], d_order[2]),
                    seasonal_order=(s_order[0], s_order[1], s_order[2], s_order[3]))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.predict(len(yy), len(yy) + day_range_prediction - 1)
    return {
        "prediction": yhat,
        "actual_value": yyy
    }


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#                                    OTHER FUNCTION
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def get_error(prediction, actual):
    if actual != np.nan:
        return abs(prediction - actual)
    else:
        return 0

def get_RMSE(pred, act, max_error_position):
    mse = 0
    tot = 0
    for i in range(len(pred)):
        if act[i] != np.nan and i != max_error_position:
            mse += pow(pred[i] - act[i], 2)
            tot += 1
    mse = mse / tot
    rmse = math.sqrt(mse)
    return rmse












def main_alerter(product_type, location_name, date_start, date_end, data_range, peak_id, parameter, day_range_prediction):

    directory_path = "./Data/" + product_type + "/" + location_name +  " /range_data/ "
    directory_path = directory_path + str(data_range) + "/gaussian_shapes/peak_" + str(peak_id) + "/"
    data_set = get_json_content_w_name(directory_path , "parameters")

    # DATA PREPARATION
    new_data = []
    for day_counter in range(int((date_end - date_start).days)):
        date = date_start + datetime.timedelta(days=day_counter)
        date_str = date.strftime("%Y-%m-%d")
        if date_str in data_set: new_data.append({"parameters": data_set[date_str]})
        else: new_data.append({"parameters": [np.nan, np.nan, np.nan]})

    # PREDICTION
    pred = SARIMA_test(new_data, parameter, [0, 1, 3], [1, 1, 1, 4], day_range_prediction)

    # ALARMING
    max_difference = 20000
    max_error = 0
    max_error_position = 0
    for i in range(len(pred["prediction"])):
        error = get_error(pred["prediction"][i], pred["actual_value"][i])
        if error > max_error:
            max_error = error
            max_error_position = i
    rmse = get_RMSE(pred["prediction"], pred["actual_value"], max_error_position)

    flag = "GREEN"
    is_ok = True

    if rmse > max_difference:
        is_ok = False
        if rmse > max_difference * 2:
            flag = "RED"
        else:
            flag = "ORANGE"
    else:
        if rmse > max_difference / 2:
            flag = "YELLOW"

    print("RMSE: " + str(rmse))
    print(is_ok)
    print(flag)



def main_alerter_sabetta():

    date = datetime.datetime.now()
    date_start = date.replace(year=2021, month=3, day=1, hour=0, minute=0, second=0, microsecond=0)
    date_end = date.replace(year=2021, month=10, day=1, hour=0, minute=0, second=0, microsecond=0)

    main_alerter("NO2", "Sabetta Port", date_start, date_end, 30, 2, 2, 10)


def main_alerter_default(location_name, date_start, date_end, peak_id):

    main_alerter("NO2", location_name, date_start, date_end, 30, peak_id, 2, 10)


main_alerter_sabetta()
