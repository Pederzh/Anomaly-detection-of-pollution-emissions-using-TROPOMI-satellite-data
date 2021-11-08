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
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE


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
#                                  RANDOM FOREST
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def random_forest_test(data_set_mean, target, day_range_prediction):
    df_array_train = []
    df_array_test = []
    for i in range(len(data_set_mean)):

        datestr = data_set_mean[i]["date"]
        datestr = datestr.split("-")
        date = datetime.datetime.now()
        date_final = date.replace(year=int(datestr[0]), month=int(datestr[1]), day=int(datestr[2]))
        day_initial = date.replace(year=int(datestr[0]), month=1, day=1)
        data = {
            "target": data_set_mean[i]["parameters"][target],
            "day_of_year": (date_final - day_initial).days
        }
        for j in range(10):
            data["year_" + str(2020 + j)] = 0
        for j in range(12):
            data["month_" + str(j+1)] = 0
        for j in range(31):
            data["day_" + str(j+1)] = 0
        for j in range(7):
            data["day_of_week" + str(j)] = 0
        data["year_" + str(date_final.year)] = 1
        data["month_" + str(date_final.month)] = 1
        data["day_" + str(date_final.day)] = 1
        #print(data_set_mean[i]["date"].weekday())
        data["day_of_week" + str(date_final.weekday())] = 1
        if i < (len(data_set_mean) - day_range_prediction):
            df_array_train.append(data)
        else:
            df_array_test.append(data)
    df_train = pd.DataFrame(df_array_train)
    print(df_train)
    df_train.dropna(subset = ["target"], inplace=True)

    df_test = pd.DataFrame(df_array_test)
    print(df_test)
    df_test.dropna(subset=["target"], inplace=True)

    # splitting in target and parameters
    y_train = df_train["target"]
    x_train = df_train.drop("target", axis=1)

    y_test = df_test["target"]
    x_test = df_test.drop("target", axis=1)

    # splitting the data
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.05, random_state = 2)

    # fitting the training data
    rfe = RFE(RandomForestRegressor(n_estimators=100, random_state=1))
    rfe_fit = rfe.fit(x_train, y_train)
    y_prediction = rfe.predict(x_test)
    MSE = 0
    for i in range(len(list(y_prediction))):
        MSE += pow(list(y_prediction)[i] - list(y_test)[i], 2)
    return MSE



























# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#                                  LINEAR REGRESSION
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def linear_regression_test(data_set_mean, target, day_range_prediction):
    df_array_train = []
    df_array_test = []
    for i in range(len(data_set_mean)):

        datestr = data_set_mean[i]["date"]
        datestr = datestr.split("-")
        date = datetime.datetime.now()
        date_final = date.replace(year=int(datestr[0]), month=int(datestr[1]), day=int(datestr[2]))
        day_initial = date.replace(year=int(datestr[0]), month=1, day=1)
        data = {
            "target": data_set_mean[i]["parameters"][target],
            "day_of_year": (date_final - day_initial).days
        }
        for j in range(10):
            data["year_" + str(2020 + j)] = 0
        for j in range(12):
            data["month_" + str(j + 1)] = 0
        for j in range(31):
            data["day_" + str(j + 1)] = 0
        for j in range(7):
            data["day_of_week" + str(j)] = 0
        data["year_" + str(date_final.year)] = 1
        data["month_" + str(date_final.month)] = 1
        data["day_" + str(date_final.day)] = 1
        # print(data_set_mean[i]["date"].weekday())
        data["day_of_week" + str(date_final.weekday())] = 1
        if i < (len(data_set_mean) - day_range_prediction):
            df_array_train.append(data)
        else:
            df_array_test.append(data)
    df_train = pd.DataFrame(df_array_train)
    df_train.dropna(subset=["target"], inplace=True)

    df_test = pd.DataFrame(df_array_test)
    df_test.dropna(subset=["target"], inplace=True)

    # splitting in target and parameters
    y_train = df_train["target"]
    x_train = df_train.drop("target", axis=1)

    y_test = df_test["target"]
    x_test = df_test.drop("target", axis=1)

    # fitting the training data
    LR = LinearRegression()
    LR.fit(x_train, y_train)
    y_prediction = LR.predict(x_test)
    MSE = 0
    for i in range(len(list(y_prediction))):
        MSE += pow(list(y_prediction)[i] - list(y_test)[i], 2)
    return MSE












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


    """y_tmp = []
    new_y = []
    new_x = []
    for i in range(len(y)):
        if y[i] != np.nan: y_tmp.append(y[i])
        if (i+1) % 10 == 0:
            if len(y_tmp) == 0: new_y.append(np.nan)
            else:
                if len(y_tmp) == 1: new_y.append(y_tmp[0])
                else:
                    y_tmp.sort()
                    tot = 0
                    count = 0
                    for j in range(len(y_tmp)):
                        if j/(len(y_tmp)-1) <= 0.8:
                            count += 1
                            tot += y_tmp[j]
                    new_y.append(tot/count)
            y_tmp = []
            new_x.append(len(new_x))
    y = new_y
    x = new_x"""

    yy = []
    pos = []
    yyy = []
    len_train = 10
    for j in range(len(y)):
        if j < len(y) - len_train - 1:
            yy.append(y[j])
            pos.append(j)
        else:
            if len(yyy) < 10:
                yyy.append(y[j])
    model = SARIMAX(yy, order=(d_order[0], d_order[1], d_order[2]),
                    seasonal_order=(s_order[0], s_order[1], s_order[2], s_order[3]))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.predict(len(yy), len(yy) + 9)
    print("prediction at pos " + str(len(yy)))
    print("prediction: ")
    print(yhat)
    print("actual value: ")
    print(yyy)
    print(" ")

    #plt.show()

    """plt.plot(x, y)
    plt.show()"""







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
#                                DECOMPOSITION
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def decomposition_test(data_set, target, d_order, s_order):
    y = []
    x = []
    df_array = []
    date = datetime.datetime.now()
    #date_start = date.replace(year=2020, month=3, day=1, hour=0, minute=0, second=0, microsecond=0)
    date_start = date.replace(year=2021, month=3, day=1, hour=0, minute=0, second=0, microsecond=0)
    for i in range(len(data_set)):
        i_date = date_start + datetime.timedelta(days=i)
        y.append(data_set[i]["parameters"][target])
        x.append(i)
        df_array.append({
            "Date": i_date,
            "Target": data_set[i]["parameters"][target]
        })

    df = pd.DataFrame(df_array)
    #df = df.dropna()

    sc_in = MinMaxScaler(feature_range=(0, 1))
    scaled_input = sc_in.fit_transform(df[["Target"]])
    scaled_input = pd.DataFrame(scaled_input)
    X = scaled_input
    X = X.dropna()

    df.index = df["Date"]
    df = df.drop("Date", axis=1)

    print(X)

    seas_d = sm.tsa.seasonal_decompose(X[0], model ="add", period = 7);
    fig = seas_d.plot()
    fig.set_figheight(4)
    plt.show()

    def optimize_SARIMA(parameters_list, d, D, s, exog):
        """
            Return dataframe with parameters, corresponding AIC and SSE
            parameters_list - list with (p, q, P, Q) tuples
            d - integration order
            D - seasonal integration order
            s - length of season
            exog - the exogenous variable
        """

        results = []
        for param in parameters_list:
            try:
                model = SARIMAX(exog, order=(param[0], d, param[1]), seasonal_order=(param[2], D, param[3], s)).fit(
                    disp=-1)
            except:
                continue
            aic = model.aic
            results.append([param, aic])
        result_df = pd.DataFrame(results)
        result_df.columns = ['(p,q)x(P,Q)', 'AIC']
        # Sort in ascending order, lower AIC is better
        result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)

        return result_df

    p = range(0, 4, 1)
    d = 1
    q = range(0, 4, 1)
    P = range(0, 4, 1)
    D = 1
    Q = range(0, 4, 1)
    s = 4
    parameters = product(p, q, P, Q)
    parameters_list = list(parameters)
    print(len(parameters_list))
    result_df = optimize_SARIMA(parameters_list, 1, 1, 4, df)
    print(result_df)








# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#                                       MAIN
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


data_set = get_json_content_w_name("../Data/NO2/Sabetta Port/range_data/30/gaussian_shapes/peak_2/", "parameters")
old_data_set = get_json_content_w_name("../Data/NO2/Sabetta Port/range_data/30/gaussian_shapes/peak_2/", "old_parameters")
test_set = get_json_content_w_name("../Data/NO2/Sabetta Port/range_data/30/gaussian_shapes/peak_2/", "test_set")
prop_2020_to_2021 = 2.4

data_set_mean_mobile = get_parameters_mobile_mean(data_set, 11)
data_set_normal = get_parameters(data_set)
data_set_mean = get_parameters_mean(data_set, 10, "cleaned_mean")

data = data_set_normal
for i in range(len(data)):
    data[i]["parameters"][0] = data[i]["parameters"][2] / math.pi


new_data = []
date = datetime.datetime.now()
date_start = date.replace(year=2021, month=3, day=1, hour=0, minute=0, second=0, microsecond=0)
date_end = date.replace(year=2021, month=10, day=15, hour=0, minute=0, second=0, microsecond=0)
for day_counter in range(int((date_end - date_start).days)):
    date = date_start + datetime.timedelta(days=day_counter)
    date_str = date.strftime("%Y-%m-%d")
    if date_str in data_set:
        new_data.append({
            "parameters": data_set[date_str],
            "date": date_str
        })
    """if date_str in old_data_set:
        old_data_set[date_str][0] = old_data_set[date_str][0] * prop_2020_to_2021
        old_data_set[date_str][2] = old_data_set[date_str][2] * prop_2020_to_2021
        new_data.append({
            "parameters": old_data_set[date_str],
            "date": date_str
        })"""
    """if date_str in test_set:
        new_data.append({
            "parameters": test_set[date_str]
        })"""
    if date_str not in data_set: # and date_str not in old_data_set: #and date_str not in test_set: #and date_str not in old_data_set:
        new_data.append({
            "parameters": [np.nan, np.nan, np.nan],
            "date": date_str
        })

old_data_set_mean = get_parameters_mean(old_data_set, 10, "cleaned_mean")


#linear_regression_test(data, 2)
# [0, 1, 3], [1, 1, 1, 4] new
# [3, 1, 3], [0, 1, 3, 4] old & new
#SARIMA_test(new_data, 2, [0, 1, 3], [1, 1, 1, 4])
#decomposition_test(new_data, 2, [1, 1, 0], [0, 0, 0, 0])
#polynomial_regression_test(data_set_mean, 2, 7)
MSE = random_forest_test(new_data, 2, 10)

#MSE = linear_regression_test(new_data, 2, 5)
print(MSE)

# random forest
# 22503333 + 1900224098 + 13585669 + 4437918 + 50123599
random_f = pow((22503333 + 1900224098 + 13585669 + 4437918 + 50123599 + 47444089752)/35, 1/2)
# random forest
# 11598878 + 1660596509 + 453114053 + 1280070904 + 816598364
linear_r = pow((11598878 + 1660596509 + 453114053 + 1280070904 + 816598364)/25, 1/2)
print("random forest " + str(random_f))
print("linear regr " + str(linear_r))

# SARIMAX
# LSTM

"""predicted_on = [14416.00501533, 13381.46654801,  9929.42646877, 15620.59473791, 11022.62273669,
                11333.98580037,  9760.64618931, 11937.11369269, 20552.57430616, 18451.35259097,
                20192.26793016, 22211.42801011, 16975.65180541, 17842.65930563, 18285.33206189]
actual_on = [9917, 3099, 4842, 6046, 45339, 15242, 1132,
             18705, 15400, 944, 8468, 3076, 8756, 2848, 12031]

predicted_n = [12078.76886436, 10474.69347418, 9435.82755632, 10712.58372369, 9135.41076978,
               10421.17844541, 9872.87050471, 4021.57830159, 2577.86673349, 5825.61405374,
               2182.07542376, 5094.74660349, 4380.64204229, 3945.76367494, 1548.24728261]
actual_n = [9917, 3099, 4842, 6046, 45339, 15242, 1132,
            18705, 15400, 944, 8468, 3076, 8756, 2848, 12031]"""

# NEW & OLD

"""p1 = [8133.22001065, 10512.09547713, 10720.0390543, 5707.36774615, 9572.30696778]
a1 = [92747, 70355, 23753, 39351, 6984]

p2 = [14416.00501533, 13381.46654801,  9929.42646877, 15620.59473791, 11022.62273669, 11333.98580037,  9760.64618931]
a2 = [9917, 3099, 4842, 6046, 45339, 15242, 1132]

p3 = [11937.11369269, 20552.57430616, 18451.35259097, 20192.26793016, 22211.42801011, 16975.65180541,
 17842.65930563, 18285.33206189]
a3 = [18705, 15400, 944, 8468, 3076, 8756, 2848, 12031]"""

# NEW

"""p1 = [7006.87161057, 10236.64531677, 10733.60130703, 10145.73010346, 10637.79668064]
a1 = [92747, 70355, 23753, 39351, 6984]

p2 = [12078.76886436, 10474.69347418, 9435.82755632, 10712.58372369, 9135.41076978, 10421.17844541, 9872.87050471]
a2 = [9917, 3099, 4842, 6046, 45339, 15242, 1132]

p3 = [4021.57830159, 2577.86673349, 5825.61405374, 2182.07542376, 5094.74660349, 4380.64204229, 3945.76367494, 1548.24728261]
a3 = [18705, 15400, 944, 8468, 3076, 8756, 2848, 12031]

def cleaned_mean(data):
    data.sort()

    count = 0
    tot = 0
    for i in range(len(data)):
        if i/len(data)-1 <= 0.8:
            count += 1
            tot += data[i]
    return tot/count

p1 = stst.mean(p1)
p2 = stst.mean(p2)
p3 = stst.mean(p3)
a1 = cleaned_mean(a1)
a2 = cleaned_mean(a2)
a3 = cleaned_mean(a3)
#print(str(p1) + " --> " + str(a1))
#print(str(p2) + " --> " + str(a2))
#print(str(p3) + " --> " + str(a3))


def get_MSEP(pred, act):
    mse = 0
    for i in range(len(pred)):
        mse += pow((pred[i] - act[i])/act[i], 2)
    mse = mse / len(pred)
    print(mse)

#mean square error
def get_MSE(pred, act):
    mse = 0
    for i in range(len(pred)):
        mse += pow(pred[i] - act[i], 2)
    mse = mse / len(pred)
    print(mse)

#root mean square error
def get_RMSE(pred, act):
    mse = 0
    for i in range(len(pred)):
        mse += pow(pred[i] - act[i], 2)
    mse = mse / len(pred)
    rmse = math.sqrt(mse)
    print(rmse)

print("using 2020 & 2021")
get_RMSE(predicted_on, actual_on)
print(" ")
print("using 2021")
get_RMSE(predicted_n, actual_n)"""


"""
old + new
8929.005851202 --> 46638.0
12209.249642341429 --> 12231.0
18306.047462877497 --> 8778.5
[8929.005851202, 12209.249642341429, 18306.047462877497]
[46638.0, 12231.0, 8778.5]
"""

"""
new
9752.129003693999 --> 46638.0
10304.476191207143 --> 12231.0
3697.0667644887494 --> 8778.5
[9752.129003693999, 10304.476191207143, 3697.0667644887494]
[46638.0, 12231.0, 8778.5]
"""