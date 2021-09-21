import datetime
import io
import json
import math

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
        return tot/n
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
    list.sort()
    return list

def image_median_value(sorted_list):
    median_position = int((len(sorted_list)-1)/2)
    return sorted_list[median_position]

def image_mode_value(frequencies):
    keys = list(frequencies.keys())
    max_freq = 0
    mode = -1
    for i in range(len(keys)):
        if frequencies[keys[i]] > max_freq:
            max_freq = frequencies[keys[i]]
            mode = int(keys[i])
    return mode

def image_max_value(sorted_list):
    return sorted_list[len(sorted_list)-1]

def image_min_value(sorted_list):
    return sorted_list[0]

def image_min_quartile_value(sorted_list):
    return sorted_list[int((len(sorted_list)-1)/4)]

def image_max_quartile_value(sorted_list):
    return sorted_list[int((len(sorted_list)-1)/4)]

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
    return len(image)*len(image[0])

def image_non_zero_values_ratio(image):
    return image_tot_non_zero_values(image)/image_tot_values(image)

def get_image_stats(image):
    sorted_list = image_to_sorted_list(image)
    frequencies = image_to_frequencies(sorted_list)
    median = image_median_value(sorted_list)
    mean = image_mean_value(image)
    stats = {
        "frequencies": frequencies,
        "mean": mean,
        "mode": image_mode_value(frequencies),
        "median": median,
        "variance": image_variance_value(image, mean),
        "non_zeroes": image_tot_non_zero_values(image),
        "zeroes": image_non_zero_values_ratio(image),
        "n_tot": image_tot_values(image),
        "box_plot": {
            "min": image_min_value(sorted_list),
            "quartile 0.25": image_min_quartile_value(sorted_list),
            "median": median,
            "quartile 0.75": image_max_quartile_value(sorted_list),
            "max": image_max_value(sorted_list)
        }
    }
    return stats


location_name = "Bering Strait"
product_type = "NO2"
directory_path = "./Data/" + location_name + "/" + product_type + "/"

with open(directory_path + "2019.txt") as json_file:
    data_2019 = json.load(json_file)
with open(directory_path + "2020.txt") as json_file:
    data_2020 = json.load(json_file)
with open(directory_path + "2021.txt") as json_file:
    data_2021 = json.load(json_file)

#

stats = get_image_stats(data_2019["2019-01-19"])

print(stats)
