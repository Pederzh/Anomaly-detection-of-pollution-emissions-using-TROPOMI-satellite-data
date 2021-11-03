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
from osgeo import gdal
from osgeo import osr
import numpy as np
import pathlib
import os, sys

# Your client credentials
client_id = '982de4f4-dade-4f98-9b49-4374cd896bb6'
client_secret = '%p/,0Yrd&/mO%cdudUsby[>@]MB|2<rf1<NnXkZr'

# Create a sessionv
client = BackendApplicationClient(client_id=client_id)
oauth = OAuth2Session(client=client)

# Get token for the session
token = oauth.fetch_token(token_url='https://services.sentinel-hub.com/oauth/token',
                          client_id=client_id, client_secret=client_secret)


def stat_to_image(data, key, precision):
    rgb_matrix = []
    for y in range(len(data)):
        rgb_matrix.append([])
        for x in range(len(data[y])):
            value = 255 - int(data[y][x][key] * 255 / precision)
            if value != -1: rgb_matrix[y].append([value, value, value, 255])
            else: rgb_matrix[y].append([255, 255, 255, 0])
    return rgb_matrix

def get_standard_rgb_values(value):
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
    rgb[3] = round(prop * 1016) + 50
    if rgb[3] > 255: rgb[3] = 255
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

def get_red_rgba_values(value):
    if value == -1 or value == None: return [0, 0, 0, 0]
    limit_1 = 255
    if value <= limit_1:
        rgb = [255, 0, 0, 0]
        prop = value / limit_1
        rgb_value = round(255 * prop)
        rgb[3] = rgb_value
        return rgb
    limit_1 = 255
    limit_2 = 510
    if value > limit_1 and value <= limit_2:
        rgb = [0, 0, 0, 255]
        prop = (value - limit_1) / (limit_2 - limit_1)
        rgb_value = round(255 - 255 * prop)
        rgb[0] = rgb_value
        return rgb
    limit_1 = 510
    limit_2 = 765
    if value > limit_1 and value <= limit_2:
        rgb = [0, 0, 0, 255]
        prop = (value - limit_1) / (limit_2 - limit_1)
        rgb_value = round(150 - 150 * prop)
        rgb[0] = rgb_value
        return rgb
    return [0, 0, 0, 250]

def get_rgba_values(value):
    if value == -1 or value == None: return [0, 255, 0, 255]
    limit_1 = 255
    if value <= limit_1:
        rgb = [255, 0, 0, 0]
        prop = value / limit_1
        rgb_value = round(255 * prop)
        #rgb[1] = 255 - rgb_value
        rgb[1] = 255 - rgb_value
        rgb[2] = 255 - rgb_value
        rgb[3] = rgb_value*2
        if rgb[3] > 255: rgb[3] = 255
        return rgb
    limit_1 = 255
    limit_2 = 510
    if value > limit_1 and value <= limit_2:
        rgb = [0, 0, 0, 255]
        prop = (value - limit_1) / (limit_2 - limit_1)
        rgb_value = round(255 * prop)
        rgb[0] = 255 - rgb_value
        return rgb
    limit_1 = 510
    limit_2 = 1016
    prop = (value - limit_1) / (limit_2 - limit_1)
    return [prop * 100, 0, prop * 150, 250]

def create_image_from_matrix(data):
    image = []
    for y in range(len(data)):
        image.append([])
        for x in range(len(data)):
            image[y].append(get_rgba_values(data[y][x]))
    return image

def print_image_given_matrix(matrix):
    image = create_image_from_matrix(matrix)
    plt.imshow(image)
    plt.show()

def get_json_content_w_name(directory_path, name):
    with open(directory_path + name + ".json") as json_file:
        data = json.load(json_file)
    return data

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#              PARAMETERS DEFINITION
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

location_names = ["Bering Strait", "Sabetta Port"]
locations_coords = [
    {
        "lat": [65.396, 66.230],
        "lon": [-170.844, -167.622]
    },
    {
        "lat": [70.36645, 72.16308],
        "lon": [69.26337, 74.85694]
    }
]
product_types = ["CO", "NO2", "CH4", "SO2"]

date = datetime.datetime.now()
date_start = date.replace(year=2021, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
date_end = date.replace(year=2021, month=9, day=1, hour=0, minute=0, second=0, microsecond=0)

locationIndex = 1
location_name = location_names[locationIndex]
location_coord = locations_coords[locationIndex]
product_type = product_types[1]

precision = 10000
stats_keys = ["mean", "mode", "min", "min_quartile", "median", "max_quartile", "max", ]

info_name = location_name.strip() + "_" + product_type + "_"
file_name = info_name + date_start.strftime("%Y-%m-%d") + "_" + date_end.strftime("%Y-%m-%d") + "_STATS"
directory_path = "./data/" + location_name + "/" + product_type + "/Statistics/"
print("from file: " + directory_path + file_name)

directory_path = "../Data/NO2/Sabetta Port/range_data/30/gaussian_shapes/peak_2/"
data_set = get_json_content_w_name(directory_path, "data")
directory_path = "../Data/NO2/Sabetta Port/images/2021/05/03/lowed/"
file_name = "mean"
directory_path = "../Data/NO2/Sabetta Port/range_data/30/"
file_name = "data"
data_set = get_json_content_w_name(directory_path, file_name)["2021-04-01"]

directory_write = "../Thesis images/"
file_write = "2021-04-01"


data = data_set
key = stats_keys[0]

#  Initialize the Image Size
image_size = (len(data[0]), len(data)) #x,y

#  Choose some Geographic Transform (Around Lake Tahoe)

lat = location_coord["lat"]
lon = location_coord["lon"]

r_pixels = np.zeros((image_size), dtype=np.uint8)
g_pixels = np.zeros((image_size), dtype=np.uint8)
b_pixels = np.zeros((image_size), dtype=np.uint8)
a_pixels = np.zeros((image_size), dtype=np.uint8)

for y in range(len(data)):
    for x in range(len(data[y])):
        rgba = get_rgba_values(data[y][x])
        r_pixels[y, x] = rgba[0]
        g_pixels[y, x] = rgba[1]
        b_pixels[y, x] = rgba[2]
        a_pixels[y, x] = rgba[3]

# set geotransform
nx = image_size[0]
ny = image_size[1]
xmin, ymin, xmax, ymax = [min(lon), min(lat), max(lon), max(lat)]
xres = (xmax - xmin) / float(nx)
yres = (ymax - ymin) / float(ny)
geotransform = (xmin, xres, 0, ymax, 0, -yres)

info_name_write = location_name.strip() + "_" + product_type + "_"
file_name_write = info_name_write + date_start.strftime("%Y-%m-%d") + "_" + date_end.strftime("%Y-%m-%d") + "_" + key + "_GEOTIF"
directory_path_write = "./data/" + location_name + "/" + product_type + "/geotiff/"
directory_path_write = directory_write
file_name_write = file_write

# create the 3-band raster file
dst_ds = gdal.GetDriverByName('GTiff').Create(directory_path_write + file_name_write + '.tif', ny, nx, 4, gdal.GDT_Byte)

dst_ds.SetGeoTransform(geotransform)    # specify coords
srs = osr.SpatialReference()            # establish encoding
srs.ImportFromEPSG(4326)                # lat/long (correct: 4326)
dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
dst_ds.GetRasterBand(1).WriteArray(r_pixels)   # write r-band to the raster
dst_ds.GetRasterBand(2).WriteArray(g_pixels)   # write g-band to the raster
dst_ds.GetRasterBand(3).WriteArray(b_pixels)   # write b-band to the raster
dst_ds.GetRasterBand(4).WriteArray(a_pixels)   # write b-band to the raster
dst_ds.FlushCache()                     # write to disk
dst_ds = None