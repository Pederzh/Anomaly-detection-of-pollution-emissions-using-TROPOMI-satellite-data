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
        "lat": [71.138, 71.374],
        "lon": [71.779, 72.683]
    }
]
product_types = ["CO", "NO2", "CH4", "SO2"]

date = datetime.datetime.now()
date_start = date.replace(year=2021, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
date_end = date.replace(year=2021, month=9, day=1, hour=0, minute=0, second=0, microsecond=0)

locationIndex = 0
location_name = location_names[locationIndex]
location_coord = locations_coords[locationIndex]
product_type = product_types[1]

precision = 10000
stats_keys = ["mean", "mode", "min", "min_quartile", "median", "max_quartile", "max", ]

info_name = location_name.strip() + "_" + product_type + "_"
file_name = info_name + date_start.strftime("%Y-%m-%d") + "_" + date_end.strftime("%Y-%m-%d") + "_STATS"
directory_path = "./data/" + location_name + "/" + product_type + "/Statistics/"
print("from file: " + directory_path + file_name)

# READING JSON FILE
with open(directory_path + file_name + ".json") as json_file:
    data_set = json.load(json_file)


data = data_set["data"]
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
        value = 255 - int(data[y][x]["statistics"][key][0] * 255 / precision)
        if value != -1:
            r_pixels[y, x] = value
            g_pixels[y, x] = value
            b_pixels[y, x] = value
            a_pixels[y, x] = 255

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