import datetime
import io

import math
import json
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
import xgboost as xgb
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

def get_json_content_w_name_checking_exist(directory_path, name):
    my_file = Path(directory_path + name + ".json")
    if my_file.is_file():
        with open(directory_path + name + ".json") as json_file:
            data = json.load(json_file)
        return data
    else: return None

def get_standard_rgba_values(value):
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
            image[y].append(get_standard_rgba_values(data[y][x]))
    return image







def gauss_value(parameters, point):
    A = parameters[0]
    B = parameters[1]
    y = point[0]
    x = point[1]
    return (A * pow(math.e, -B * (pow(x, 2) + pow(y, 2))))

def get_gaussian_parameters(volume):
    A = pow(volume, 2 / 3) / pow(math.pi, 2 / 3)
    if A == 0:
        B = 1
    else:
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

def get_image_png(rgbt_matrix):
    array = np.array(rgbt_matrix, dtype=np.uint8)
    image = Image.fromarray(array)
    return image


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

def get_new_coordinates(lat, lon, distance_lat, distance_lon):
    lat_new = lat + (180 / math.pi) * (distance_lat / 6378137)
    lon_new = lon + (180 / math.pi) * (distance_lon / 6378137) / math.cos(math.pi/180*lat)
    return [lat_new, lon_new]

def get_new_longitude(lat, lon, distance_lon):
    return get_new_coordinates(lat, lon, 0, distance_lon)[1]

def get_new_latitude(lat, lon, distance_lat):
    return get_new_coordinates(lat, lon, distance_lat, 0)[0]

def get_bbox_coordinates_from_center(coordinates, distance):
    new_coordinates = []
    lat = coordinates[0]
    lon = coordinates[1]
    new_coordinates.append(get_new_latitude(lat, lon, distance[0]))
    new_coordinates.append(get_new_longitude(lat, lon, distance[1]))
    return new_coordinates

























# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#             SARIMA Seasonal Autoregressive Integrated Moving-Average
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def SARIMA_test(data_set_mean, target, d_order, s_order, day_range_prediction, tot_range_days):

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
    if (tot_range_days+1-day_range_prediction) > 365:
        model = SARIMAX(yy, order=(d_order[0], d_order[1], d_order[2]),
                        seasonal_order=(s_order[0], s_order[1], s_order[2], 365))
    else:
        model = SARIMAX(yy, order=(d_order[0], d_order[1], d_order[2]))
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

def get_mean_from_RMSE(pred, act, max_error_position):
    new_pred = 0
    new_act = 0
    tot = 0

    for i in range(len(pred)):
        if act[i] != np.nan and i != max_error_position:
            new_pred += pred[i]
            new_act += act[i]
            tot += 1
    new_pred = new_pred / tot
    new_act = new_act / tot
    return [new_pred, new_act]












def main_alerter(product_type, location_name, date_start, date_end, data_range, peak_id, parameter, day_range_prediction):

    directory_path = "../Data/" + product_type + "/" + location_name + "/range_data/"
    directory_path = directory_path + str(data_range) + "/peaks/"
    peaks = get_json_content_w_name(directory_path, "peaks")
    if peaks == None: return {"error": "peaks file not found, please reprocess"}
    if len(peaks) == 0: return {"error": "no peaks found"}
    image_ccs = None
    for peak in peaks:
        if peak["id"] == peak_id:
            image_ccs = peak["point"]
    if image_ccs == None: return None

    map_ccs = get_json_content_w_name("../Data/" + product_type + "/" + location_name + "/", "coordinates")

    directory_path = "../Data/" + product_type + "/" + location_name +  "/range_data/"
    directory_path = directory_path + str(data_range) + "/gaussian_shapes/peak_" + str(peak_id) + "/"
    data_set = get_json_content_w_name(directory_path , "parameters")

    # DATA PREPARATION
    new_data = []
    tot_range_days = 0
    for day_counter in range(int((date_end - date_start).days)):
        date = date_start + datetime.timedelta(days=day_counter)
        date_str = date.strftime("%Y-%m-%d")
        if date_str in data_set:
            new_data.append({
                "parameters": data_set[date_str],
                "date": date_str
            })
            tot_range_days = day_counter
        else: new_data.append({
            "parameters": [np.nan, np.nan, np.nan],
            "date": date_str
        })

    # PREDICTION
    pred = SARIMA_test(new_data, parameter, [0, 1, 3], [1, 1, 1, 4], day_range_prediction, tot_range_days)

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
    new_pred = get_mean_from_RMSE(pred["prediction"], pred["actual_value"], max_error_position)

    pred_parameters = get_gaussian_parameters(new_pred[0])
    actl_parameters = get_gaussian_parameters(new_pred[1])
    distance = [
        image_ccs[0] - 49,
        image_ccs[1] - 49
    ]
    final_ccs = get_bbox_coordinates_from_center(map_ccs["coordinates"], distance)

    # SETTING THE RESPONSE
    rmse_final = abs(pred_parameters[2] - actl_parameters[2])
    flag = "GREEN"
    if rmse_final > max_difference:
        if rmse_final > max_difference * 2:
            flag = "RED"
        else:
            flag = "ORANGE"
    else:
        if rmse_final > max_difference / 2:
            flag = "YELLOW"

    response = {
        "status": flag,
        "forecasted_value": {
            "peak": pred_parameters[0],
            "attenuation": pred_parameters[1],
            "volume": pred_parameters[2],
            #"GROTE_image": GROTE_img_forecst,
        },
        "actual_value": {
            "peak": actl_parameters[0],
            "attenuation": actl_parameters[1],
            "volume": actl_parameters[2],
            #"GROTE_image": GROTE_img_act,
            #"original_image": original_im,
            #"processed_image": processed_im,
        },
        "other_information": {
            "coordinates": final_ccs,
            "days_range": day_range_prediction,
            "date": date_end,
        }
    }
    return response



def main_alerter_sabetta():

    date = datetime.datetime.now()
    date_start = date.replace(year=2021, month=3, day=1, hour=0, minute=0, second=0, microsecond=0)
    date_end = date.replace(year=2021, month=10, day=1, hour=0, minute=0, second=0, microsecond=0)
    return main_alerter("NO2", "Sabetta Port", date_start, date_end, 30, 2, 2, 10)


def main_alerter_default(location_name, date_start, date_end, peak_id):

    return main_alerter("NO2", location_name, date_start, date_end, 30, peak_id, 2, 30)










# MAIN TMP



date = datetime.datetime.now()
month_start = 4
date_start = date.replace(year=2021, month=3, day=1, hour=0, minute=0, second=0, microsecond=0)
date_end = date.replace(year=2021, month=month_start, day=25, hour=0, minute=0, second=0, microsecond=0)
date_pred = date.replace(year=2021, month=month_start, day=1, hour=0, minute=0, second=0, microsecond=0)
preds = []
acts = []
errs = []
for i in range(180):
    if i % 1 == 0:
        date_pred = date_end + datetime.timedelta(days=i)
        res = main_alerter_default("Sabetta Port", date_start, date_pred, 2)
        preds.append(res["forecasted_value"]["volume"])
        acts.append(res["actual_value"]["volume"])
        errs.append(abs(res["forecasted_value"]["volume"] - res["actual_value"]["volume"]))

print(preds)
print(acts)

#plt.scatter(x, y)
plt.plot(preds)
plt.show()
plt.plot(acts)
plt.show()
plt.plot(errs)
plt.show()














