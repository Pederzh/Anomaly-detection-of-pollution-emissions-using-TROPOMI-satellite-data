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



def date_to_str(date):
    date_str = str(date.year) + "-"
    if (date.month < 10): date_str += "0"
    date_str += str(date.month) + "-"
    if (date.day < 10): date_str += "0"
    date_str += str(date.day)
    return date_str



def stat_to_image(data, key, precision):
    rgb_matrix = []
    for y in range(len(data)):
        rgb_matrix.append([])
        for x in range(len(data[y])):
            value = 255 - int(data[y][x][key] * 255 / precision)
            if value != -1: rgb_matrix[y].append([value, value, value, 255])
            else: rgb_matrix[y].append([255, 255, 255, 0])
    return rgb_matrix





# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#              PARAMETERS DEFINITION
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

location_names = ["Bering Strait", "Sabetta Port"]
product_types = ["CO", "NO2", "CH4"]

location_name = location_names[0]
product_type = product_types[0]
directory_path = "./Data/" + location_name + "/" + product_type + "/"

date = datetime.datetime.now()
date_start = date.replace(year=2021, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
date_end = date.replace(year=2021, month=9, day=1, hour=0, minute=0, second=0, microsecond=0)

precision = 10000
stats_keys = ["mean", "mode", "min", "min_quartile", "median", "max_quartile", "max", ]

info_name = location_name.strip() + "_" + product_type + "_"
file_name = info_name + date_to_str(date_start) + "_" + date_to_str(date_end) + "_STATS"
directory_path = "./Data/" + location_name + "/" + product_type + "/Statistics/"

# READING JSON FILE
with open(directory_path + file_name + ".json") as json_file:
    data_set = json.load(json_file)

for i in range(len(stats_keys)):
    image = stat_to_image(data_set["data"], stats_keys[i], precision)
    print(stats_keys[i])
    fig=plt.figure()
    plt.imshow(image)
    plt.show()


