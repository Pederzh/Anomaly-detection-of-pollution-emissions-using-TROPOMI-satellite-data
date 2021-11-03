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




# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#                                       MAIN
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

day_range = 11
data_set = get_json_content_w_name("../Data/NO2/Sabetta Port/range_data/30/gaussian_shapes/peak_2/", "parameters")
data_set_mean = get_parameters_mobile_mean(data_set, day_range)

training_set = []
for i in range(len(data_set_mean)):
    training_set.append({
        "year": data_set_mean[i]["date"].year,
        "month": data_set_mean[i]["date"].month,
        "day": data_set_mean[i]["date"].day,
        "day_of_week": data_set_mean[i]["date"].weekday(),
        "target": data_set_mean[i]["parameters"][2]
    })





df_array = []
df = pd.DataFrame()
for i in range(len(data_set_mean)):
    data = {
        "target": data_set_mean[i]["parameters"][2],
        "year": data_set_mean[i]["date"].year,
        "day": data_set_mean[i]["date"].day,
    }
    for i in range(12):
        data["month_" + str(i+1)] = 0
    for i in range(7):
        data["day_of_week" + str(i+1)] = 0
    data["month_" + str(data_set_mean[i]["date"].month)] = 1
    data["day_of_week" + str(data_set_mean[i]["date"].weekday())] = 1
    """data = {
        "target": data_set_mean[i]["parameters"][2],
        "year": data_set_mean[i]["date"].year,
        "month": data_set_mean[i]["date"].month,
        "day": data_set_mean[i]["date"].day,
        "day_of_week": data_set_mean[i]["date"].weekday(),
    }"""
    df_array.append(data)
df = pd.DataFrame(df_array)

# splitting in target and parameters
y = df["target"]
x = df.drop("target", axis=1)

"""months = pd.get_dummies(x, drop_first=True)
x = x.drop('month', axis=1)
x = pd.concat([x, months], axis=1)
print(x)"""

# splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.05, random_state = 210)

# LINEAR REGRESSION
"""
# fitting the training data
LR = LinearRegression()
LR.fit(x_train,y_train)
y_prediction =  LR.predict(x_test)
print(y_prediction)
print(list(y_test))
"""

# SARIMA Seasonal Autoregressive Integrated Moving-Average
for i in range(10):
    yy = []
    yyy = 0
    len_train = i + 0
    for i in range(len(y) - len_train):
        if i < len(y) - len_train -1:
            yy.append(y[i])
        else:
            yyy = y[i]
    model = SARIMAX(yy, order=(2, 2, 2), seasonal_order=(0, 0, 0, 0))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.predict(len(yy), len(yy))
    print("prediction: " + str(yhat))
    print("actual value: " + str(yyy))
    print( " ")





"""lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Visualizing the Linear Regression results
def viz_linear():
    plt.scatter(x, y, color='red')
    plt.plot(x, lin_reg.predict(x), color='blue')
    plt.title('Truth or Bluff (Linear Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
    return
viz_linear()"""








