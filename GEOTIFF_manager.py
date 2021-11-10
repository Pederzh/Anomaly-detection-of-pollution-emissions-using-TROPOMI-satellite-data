import datetime
import io
import json
import math
import statistics as stst
import numpy

import matplotlib.pyplot as plt

from PIL import Image
from numpy import array
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from osgeo import gdal
from osgeo import osr
import tifffile as tiff
import numpy as np
import rasterio
from rasterio.plot import show
import pathlib
import os, sys
from pathlib import Path
from scipy.ndimage import median_filter, median, maximum_filter, gaussian_filter

def get_json_content_w_name(directory_path, name):
    with open(directory_path + name + ".json") as json_file:
        data = json.load(json_file)
    return data

def get_new_coordinates(lat, lon, distance_lat, distance_lon):
    lat_new = lat + (180 / math.pi) * (distance_lat / 6378137)
    lon_new = lon + (180 / math.pi) * (distance_lon / 6378137) / math.cos(math.pi / 180 * lat)
    return [lat_new, lon_new]


def get_new_longitude(lat, lon, distance_lon):
    return get_new_coordinates(lat, lon, 0, distance_lon)[1]


def get_new_latitude(lat, lon, distance_lat):
    return get_new_coordinates(lat, lon, distance_lat, 0)[0]


def get_bbox_coordinates_from_center(coordinates, distance):
    bbox_coordinates = []
    lat = coordinates[0]
    lon = coordinates[1]
    bbox_coordinates.append(get_new_longitude(lat, lon, -distance))
    bbox_coordinates.append(get_new_latitude(lat, lon, -distance))
    bbox_coordinates.append(get_new_longitude(lat, lon, distance))
    bbox_coordinates.append(get_new_latitude(lat, lon, distance))
    return bbox_coordinates

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




def convert_json_to_GEOTIFF(data, coordinates):
    bbox = get_bbox_coordinates_from_center(coordinates, 100000)
    location_coord = {
            "lat": [bbox[1], bbox[3]],
            "lon": [bbox[0], bbox[2]]
        }

    #  Initialize the Image Size
    image_size = (len(data[0]), len(data))  # x,y

    #  Choose some Geographic Transform (Around Lake Tahoe)
    location_coord = {
        "lat": [70.36645, 72.16308],
        "lon": [69.26337, 74.85694]
    }
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

    directory_path = "../data/"
    file_name = "tmp"

    dst_ds = gdal.GetDriverByName('GTiff').Create(directory_path + file_name + '.tiff', ny, nx, 4,gdal.GDT_Byte)
    dst_ds.SetGeoTransform(geotransform)  # specify coords
    srs = osr.SpatialReference()  # establish encoding
    srs.ImportFromEPSG(4326)  # lat/long (correct: 4326)
    dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(r_pixels)  # write r-band to the raster
    dst_ds.GetRasterBand(2).WriteArray(g_pixels)  # write g-band to the raster
    dst_ds.GetRasterBand(3).WriteArray(b_pixels)  # write b-band to the raster
    dst_ds.GetRasterBand(4).WriteArray(a_pixels)  # write b-band to the raster
    #dst_ds.FlushCache()  # write to disk
    #dst_ds = None
    print(dst_ds)
    #im = tiff.imread(directory_path + file_name + '.tif')
    #fp = r'../data/tmp.tif'
    #im = rasterio.open(fp)
    #show(im)
    #print(im)
    #im = gdal.Open(directory_path + file_name + '.tif')
    im = rasterio.open(directory_path + file_name + ".tiff")
    return im




data_set = get_json_content_w_name("../Data/NO2/Sabetta Port/images/2021/05/05/lowed/", "mean")
coordinates = get_json_content_w_name("../Data/NO2/Sabetta Port/", "coordinates")["coordinates"]
image = convert_json_to_GEOTIFF(data_set, coordinates)
rasterio.plot.show(image, title = "TMP")
with rasterio.open("../Data/masked.tif", "w") as dest:
    dest.write(image)
"""band = image.GetRasterBand(1)
arr = band.ReadAsArray()
[rows, cols] = arr.shape
arr_min = arr.min()
arr_max = arr.max()
arr_mean = int(arr.mean())
arr_out = arr
driver = gdal.GetDriverByName("GTiff")
outdata = driver.Create("../Data/tmp.tif", cols, rows, 1, gdal.GDT_UInt16)"""
"""outdata.SetGeoTransform(image.GetGeoTransform())##sets same geotransform as input
outdata.SetProjection(image.GetProjection())##sets same projection as input
outdata.GetRasterBand(1).WriteArray(arr_out)
outdata.GetRasterBand(1).SetNoDataValue(10000)##if you want these values transparent
outdata.FlushCache() ##saves to disk!!
outdata = None
band=None
ds=None"""
#tiff.imsave("../Data/tmp.tif", image)
#image.show()