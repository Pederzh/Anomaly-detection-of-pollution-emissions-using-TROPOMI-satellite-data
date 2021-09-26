import datetime
import io
import json
import math
import statistics as stst

import matplotlib.pyplot as plt

from PIL import Image
from numpy import array
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

from tkinter import ttk, Tk, Button, Frame, Canvas, BOTH, LEFT, VERTICAL, RIGHT, X, Y, Listbox, END, Label

# Your client credentials
client_id = '982de4f4-dade-4f98-9b49-4374cd896bb6'
client_secret = '%p/,0Yrd&/mO%cdudUsby[>@]MB|2<rf1<NnXkZr'

# Create a sessionv
client = BackendApplicationClient(client_id=client_id)
oauth = OAuth2Session(client=client)

# Get token for the session
token = oauth.fetch_token(token_url='https://services.sentinel-hub.com/oauth/token',
                          client_id=client_id, client_secret=client_secret)


def image_mean_value(image):
    tot = -1
    n = 0
    for y in range(len(image)):
        for x in range(len(image[y])):
            if image[y][x] != -1:
                tot += image[y][x]
                n += 1
    if tot != -1:
        return tot / n
    return -1


def image_variance_value(image, mean):
    tot = -1
    n = 0
    for y in range(len(image)):
        for x in range(len(image[y])):
            if image[y][x] != -1:
                tot += pow((image[y][x] - mean), 2)
                n += 1
    if tot != -1:
        return tot / n
    return -1


def image_to_sorted_list(image):
    list = []
    for i in range(len(image)): list += image[i]
    while -1 in list: list.remove(-1)
    list.sort()
    return list

def image_mode_value(frequencies):
    keys = list(frequencies.keys())
    max_freq = 0
    mode = -1
    for i in range(len(keys)):
        if frequencies[keys[i]] > max_freq:
            max_freq = frequencies[keys[i]]
            mode = int(keys[i])
    return mode

def list_max_value(sorted_list):
    if len(sorted_list) == 0: return -1
    return sorted_list[len(sorted_list) - 1]


def list_min_value(sorted_list):
    if len(sorted_list) == 0: return -1
    return sorted_list[0]

def list_quartile_value(sorted_list, position):
    if len(sorted_list) == 0: return -1
    return sorted_list[int(round(((len(sorted_list) - 1) * position), 0))]

def list_mean_value(sorted_list):
    if len(sorted_list) == 0: return -1
    return stst.mean(sorted_list)

def list_mode_value(sorted_list):
    if len(sorted_list) == 0: return -1
    return stst.mode(sorted_list)

def list_median_value(sorted_list):
    if len(sorted_list) == 0: return -1
    return stst.median(sorted_list)

def list_variance_value(sorted_list, mean):
    if len(sorted_list) == 0: return -1
    if len(sorted_list) == 1: return 0
    return stst.variance(sorted_list, mean)

def image_to_frequencies(sorted_list):
    values = {}
    for i in range(len(sorted_list)):
        if sorted_list[i] != -1:
            if str(sorted_list[i]) not in values.keys():
                values[str(sorted_list[i])] = 0
            values[str(sorted_list[i])] += 1
    return values


def image_tot_non_zero_values(image):
    tot = 0
    for y in range(len(image)):
        for x in range(len(image[y])):
            if image[y][x] != -1:
                tot += 1
    return tot


def image_tot_values(image):
    return len(image) * len(image[0])


def image_non_zero_values_ratio(image):
    return image_tot_non_zero_values(image) / image_tot_values(image)


def images_variation_mean(image, next):
    tot = 0
    n = 0
    for y in range(len(image)):
        for x in range(len(image[y])):
            if image[y][x] != -1 and next[y][x] != -1:
                tot += abs(next[y][x] - image[y][x])
                n += 1
    if n != 0:
        return tot / n
    return -1


def images_variation_variance(image, next, mean):
    tot = 0
    n = 0
    for y in range(len(image)):
        for x in range(len(image[y])):
            if image[y][x] != -1 and next[y][x] != -1:
                tmp = abs(next[y][x] - image[y][x])
                tot += pow((tmp - mean), 2)
                n += 1
    if n != 0:
        return tot / n
    return -1



def get_image_stats(image):
    sorted_list = image_to_sorted_list(image)
    frequencies = image_to_frequencies(sorted_list)
    median = list_median_value(sorted_list)
    mode = image_mode_value(frequencies)
    mean = image_mean_value(image)
    stats = {
        "frequencies": frequencies,
        "image_statistics": {
            "mean": mean,
            "mode": mode,
            "median": median,
            "variance": image_variance_value(image, mean),
            "n_tot": image_tot_values(image),
            "non_zeroes": image_tot_non_zero_values(image),
            "zeroes_frequency": image_non_zero_values_ratio(image),
        },
        "box_plot": {
            "min": list_min_value(sorted_list),
            "quartile_010": list_quartile_value(list, 0.10),
            "quartile_025": list_quartile_value(list, 0.25),
            "median": median,
            "quartile_075": list_quartile_value(list, 0.75),
            "quartile_090": list_quartile_value(list, 0.90),
            "max": list_max_value(sorted_list),
        },
    }
    return stats

def get_image_stats_and_variation(image, image_next):
    stats = get_image_stats(image)
    var_mean = images_variation_mean(image, image_next)
    stats["next_image_variation"] = {
        "variation_mean": var_mean,
        "variation_variance": images_variation_variance(image, image_next, var_mean)
    }
    return stats

def date_to_str(date):
    date_str = str(date.year) + "-"
    if (date.month < 10): date_str += "0"
    date_str += str(date.month) + "-"
    if (date.day < 10): date_str += "0"
    date_str += str(date.day)
    return date_str

def str_to_date(str_date):
    str_date = str_date.split("-")
    date = datetime.datetime.now()
    date = date.replace(year=int(str_date[0]), month=int(str_date[1]), day=int(str_date[2]))
    return date




# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#                       DAYS STATISTICS GIVEN A DATA SET OF IMAGES
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def save_days_stats(data, directory_path):
    stats = {}
    keys = list(data.keys())
    for i in range(len(keys)):
        if i != 0:
            stat = get_image_stats_and_variation(data[keys[i]], data[keys[i - 1]])
        else:
            stat = get_image_stats(data[keys[i]])
        stats[keys[i]] = stat
        if i % 100 == 0: print(i)
    with open(directory_path + "Statistics/" + "days.json", 'w') as outfile:
        json.dump(stats, outfile)





# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#                               PIXELS STATISTICS OVER PERIOD
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def save_period_pixels_stats(data_set, date_start, date_end, location_name, product_type):
    pixels = []
    for y in range(len(data_set[date_to_str(date_start)])):
        pixels.append([])
        for x in range(len(data_set[date_to_str(date_start)][y])):
            list = []
            # generating pixel list of different days
            for i in range(int((date_end - date_start).days)):
                date = date_start + datetime.timedelta(days=i)
                if data_set[date_to_str(date)][y][x] != -1:
                    list.append(data_set[date_to_str(date)][y][x])
            list.sort()
            mean = list_mean_value(list)
            pixel = {
                "mean": mean,
                "median": list_median_value(list),
                "mode": list_mode_value(list),
                "variance": list_variance_value(list, mean),
                "min": list_min_value(list),
                "quartile_010": list_quartile_value(list, 0.10),
                "quartile_025": list_quartile_value(list, 0.25),
                "quartile_075": list_quartile_value(list, 0.75),
                "quartile_090": list_quartile_value(list, 0.90),
                "max": list_max_value(list),
                "zeroes": int((date_end - date_start).days) - len(list),
                "non_zeroes": len(list)
            }
            pixels[y].append(pixel)

    stats = {
        "info": {
            "location_name": location_name,
            "product_type": product_type,
            "date_start": date_to_str(date_start),
            "date_end": date_to_str(date_end),
        },
        "data": pixels
    }
    # saving obj pixels as json file
    info_name = location_name.strip() + "_" + product_type + "_"
    file_name = info_name + date_to_str(date_start) + "_" + date_to_str(date_end) + "_STATS"
    directory_path = "./Data/" + location_name + "/" + product_type + "/Statistics/"
    with open(directory_path + file_name + ".json", 'w') as outfile:
        json.dump(stats, outfile)





# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#                           PERIOD STATISTICS GIVEN DAYS STATISTICS
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def frequencies_to_list(frequencies):
    keys = list(frequencies.keys())
    list_out = []
    for i in range(len(keys)):
        for f in range(frequencies[keys[i]]):
            list_out.append(int(keys[i]))
    return list_out

def get_frequencies_stats(fqs):
    list = frequencies_to_list(fqs)
    list.sort()
    mean = list_mean_value(list)
    median = list_median_value(list)
    stat = {
        "statistics": {
            "mean": mean,
            "mode": list_mode_value(list),
            "median": median,
            "variance": list_variance_value(list, mean),
            "n_tot": 0,
            "non_zeroes": 0,
            "zeroes_frequency": 0
        },
        "box_plot": {
            "min": list_min_value(list),
            "quartile_010": list_quartile_value(list, 0.10),
            "quartile_025": list_quartile_value(list, 0.25),
            "median": median,
            "quartile_075": list_quartile_value(list, 0.75),
            "quartile_090": list_quartile_value(list, 0.90),
            "max": list_max_value(list),
        },
    }
    return stat

def save_periodicity_stats(days_stats, periodicity):
    # periodicity --> WEEKLY, MONTHLY, ANNUALLY
    stats = {}
    frequencies = {}
    n_tot = 0
    non_zeroes = 0
    keys = list(days_stats.keys())
    keys.sort()
    day_start = keys[0]
    for i in range(len(keys)):
        # setting next date
        current_day = str_to_date(keys[i])
        if i < len(keys)-1: next_day = str_to_date(keys[i+1])
        # aggregating frequencies
        f_list = days_stats[keys[i]]["frequencies"]
        f_keys = list(f_list.keys())
        for f in range(len(f_keys)):
            if f_keys[f] not in frequencies.keys():
                frequencies[f_keys[f]] = f_list[f_keys[f]]
            else:
                frequencies[f_keys[f]] += f_list[f_keys[f]]
        # calculating elements presence
        n_tot += days_stats[keys[i]]["image_statistics"]["n_tot"]
        non_zeroes += days_stats[keys[i]]["image_statistics"]["non_zeroes"]
        refresh = False
        if periodicity == "WEEKLY":
            if int((next_day - current_day).days) >= 7 or next_day.weekday() < current_day.weekday():
                refresh = True
        if periodicity == "MONTHLY":
            if current_day.year != next_day.year or current_day.month != next_day.month:
                refresh = True
        if periodicity == "ANNUALLY":
            if current_day.year != next_day.year:
                refresh = True
        if refresh or i == len(keys)-1:
            if periodicity == "WEEKLY": key = day_start
            if periodicity == "MONTHLY":
                key = date_to_str(current_day)
                key = key.split("-")[0] + "-" + key.split("-")[1]
            if periodicity == "ANNUALLY": key = str(current_day.year)
            stat = get_frequencies_stats(frequencies)
            stat["statistics"]["n_tot"] = n_tot
            stat["statistics"]["non_zeroes"] = non_zeroes
            stat["statistics"]["zeroes_frequency"] = (n_tot-non_zeroes)/n_tot
            n_tot = 0
            non_zeroes = 0
            frequencies = {}
            stats[key] = stat
            if i < len(keys)-1: day_start = keys[i+1]
    stats["info"] = {"periodicity": periodicity}
    return stats





# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#                                   PARAMETERS DEFINITION
#                                      FUNCTION CALLER
#                                           MAIN
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

location_names = ["Bering Strait", "Sabetta Port"]
product_types = ["CO", "NO2", "CH4"]

location_name = location_names[0]
product_type = product_types[0]
directory_path = "./Data/" + location_name + "/" + product_type + "/"

"""with open(directory_path + "2019.json") as json_file:
    data_2019 = json.load(json_file)
with open(directory_path + "2020.json") as json_file:
    data_2020 = json.load(json_file)
with open(directory_path + "2021.json") as json_file:
    data_2021 = json.load(json_file)

data_set = dict(data_2019["data"])
data_set.update(data_2020["data"])
data_set.update(data_2021["data"])"""

date = datetime.datetime.now()
date_start = date.replace(year=2021, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
date_end = date.replace(year=2021, month=9, day=1, hour=0, minute=0, second=0, microsecond=0)

# !!!!!!!!!!!! CALLING !!!!!!!!!!!!!!!
#save_days_stats(data_set, directory_path)
#save_period_pixels_stats(data_set, date_start, date_end, location_name, product_type)

with open(directory_path + "Statistics/days.json") as json_file:
    days_stats = json.load(json_file)

tmp = save_periodicity_stats(days_stats, "MONTHLY")
print(tmp["2021-08"]["statistics"])