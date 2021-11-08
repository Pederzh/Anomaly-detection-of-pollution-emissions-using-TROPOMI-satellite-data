import datetime
import io
import json
import math
import os
from pathlib import Path

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



# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#             POST REQUEST FUNCTION
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def get_response (bbox, date_from_str, date_to_str, s_product, dimension):

    if s_product == "NO2":
        product_evalscritp = """
                    //VERSION=3
                    function setup() {
                        return {
                            input: ["NO2", "dataMask"],
                                output: { bands:  4},
                            }
                        }                                    
                    const minVal = 0.0
                    const maxVal = 0.0001
                    const diff = maxVal - minVal
                    const rainbowColors = [
                        [minVal, [0, 0, 0.5]],
                        [minVal + 0.125 * diff, [0, 0, 1]],
                        [minVal + 0.375 * diff, [0, 1, 1]],
                        [minVal + 0.625 * diff, [1, 1, 0]],
                        [minVal + 0.875 * diff, [1, 0, 0]],
                        [maxVal, [0.5, 0, 0]]
                    ]
                    const viz = new ColorRampVisualizer(rainbowColors)
                    function evaluatePixel(sample) {
                        var rgba= viz.process(sample.NO2)
                        rgba.push(sample.dataMask)
                        return rgba
                    }
                    """
    if s_product == "CH4":
        product_evalscritp = """
                    //VERSION=3
                    function setup() {
                        return {
                            input: ["CH4", "dataMask"],
                                output: { bands:  4}
                            }
                        }                                    
                    const minVal = 1600.0
                    const maxVal = 2000.0
                    const diff = maxVal - minVal
                    const rainbowColors = [
                        [minVal, [0, 0, 0.5]],
                        [minVal + 0.125 * diff, [0, 0, 1]],
                        [minVal + 0.375 * diff, [0, 1, 1]],
                        [minVal + 0.625 * diff, [1, 1, 0]],
                        [minVal + 0.875 * diff, [1, 0, 0]],
                        [maxVal, [0.5, 0, 0]]
                    ]
                    const viz = new ColorRampVisualizer(rainbowColors)
                    function evaluatePixel(sample) {
                        var rgba= viz.process(sample.CH4)
                        rgba.push(sample.dataMask)
                        return rgba
                    }
                    """
    if s_product == "CO":
        product_evalscritp = """
                    //VERSION=3
                    function setup() {
                        return {
                            input: ["CO", "dataMask"],
                                output: { bands:  4}
                            }
                        }                                    
                    const minVal = 0.0
                    const maxVal = 0.1
                    const diff = maxVal - minVal
                    const rainbowColors = [
                        [minVal, [0, 0, 0.5]],
                        [minVal + 0.125 * diff, [0, 0, 1]],
                        [minVal + 0.375 * diff, [0, 1, 1]],
                        [minVal + 0.625 * diff, [1, 1, 0]],
                        [minVal + 0.875 * diff, [1, 0, 0]],
                        [maxVal, [0.5, 0, 0]]
                    ]
                    const viz = new ColorRampVisualizer(rainbowColors)
                    function evaluatePixel(sample) {
                        var rgba= viz.process(sample.CO)
                        rgba.push(sample.dataMask)
                        return rgba
                    }
                    """

    response = oauth.post('https://creodias.sentinel-hub.com/api/v1/process',
        json={
          "input": {
              "bounds": {
                  "properties": {
                      "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                  },
                  "bbox": bbox
              },
              "data": [
                  {
                      "type": "sentinel-5p-l2",
                      "dataFilter": {
                          "timeRange": {
                              "from": date_from_str[0] + "T" + date_from_str[1] + "Z",
                              "to": date_to_str[0] + "T" + date_to_str[1] + "Z",
                          }
                      },
                      "processing": {"minQa": 0}
                  }
              ]
          },
          "output": dimension,
          "evalscript": product_evalscritp
        })
    return response



def save_json_w_name(directory_path, json_file, json_file_name):
    Path(directory_path).mkdir(parents=True, exist_ok=True)
    with open(directory_path + json_file_name + ".json", 'w') as outfile:
        json.dump(json_file, outfile)

def get_new_coordinates(lat, lon, distance_lat, distance_lon):
    lat_new = lat + (180 / math.pi) * (distance_lat / 6378137)
    lon_new = lon + (180 / math.pi) * (distance_lon / 6378137) / math.cos(math.pi/180*lat)
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












def get_value_from_rgb( rgb):
    """
    [minVal, [0, 0, 0.5]],
    [minVal + 0.125 * diff, [0, 0, 1]],
    [minVal + 0.375 * diff, [0, 1, 1]],
    [minVal + 0.625 * diff, [1, 1, 0]],
    [minVal + 0.875 * diff, [1, 0, 0]],
    [maxVal, [0.5, 0, 0]]
    """
    if rgb[3] == 0: return -1
    precision = 1016  # given by the rgb composition
    minVal = 0
    maxVal = minVal + precision
    diff = maxVal - minVal
    # [0, 0, 1]
    if rgb[0] == 0 and rgb[1] == 0:
        return minVal + 0.125 * diff * (rgb[2]-128)/127
    # [0, 1, 1]
    if rgb[0] == 0 and rgb[2] == 255:
        min = 0.125
        max = 0.375
        rangeVal = max - min
        return (minVal + min * diff) + rangeVal * diff * rgb[1]/255
    # [1, 1, 0]
    if rgb[1] == 255:
        min = 0.375
        max = 0.625
        rangeVal = max - min
        return (minVal + min * diff) + rangeVal * diff * rgb[0] / 255
    # [1, 0, 0]
    if rgb[0] == 255:
        min = 0.625
        max = 0.875
        rangeVal = max - min
        return (minVal + min * diff) + rangeVal * diff * (255-rgb[1]) / 255
    # [0.5, 0, 0]
    min = 0.875
    max = 1
    rangeVal = max - min
    return (minVal + min * diff) + rangeVal * diff * (127-(rgb[0]-128)) / 127


def create_image_matrix(image):
    image_matrix = [[]]
    image_matrix.clear()
    for y in range(len(image)):
        image_matrix.append([])
        for x in range(len(image[y])):
            if image[y][x][3] == 0: image_matrix[y].append(-1)
            else: image_matrix[y].append(int(get_value_from_rgb(image[y][x])))
    return image_matrix

def save_json(image_list, product_type, location_name):

    image = image_list[0]
    json_file = {}
    for i in range(len(image_list)):
        image = image_list[i]
        json_file[str(image["time"])] = create_image_matrix(image["image"])
    directory_path = "./data/" + product_type + "/" + location_name + "/images/" \
                     + image["date"].strftime("%Y") + "/" + image["date"].strftime("%m") \
                     + "/" + image["date"].strftime("%d") + "/" + "unprocessed/"
    Path(directory_path).mkdir(parents=True, exist_ok=True)
    with open(directory_path + "data.json", 'w') as outfile:
        json.dump(json_file, outfile)

















def save_image(image, product_type, location_name):
    directory_path = "./data/" + product_type + "/" + location_name + "/images/" \
                         + image["date"].strftime("%Y") + "/" + image["date"].strftime("%m") \
                           + "/" + image["date"].strftime("%d") + "/" + "unprocessed/"
    file_name = str(image["time"])
    Path(directory_path).mkdir(parents=True, exist_ok=True)
    image["image"].save(directory_path + file_name + ".png", format="png")


def get_hourly_images(date_start, start_h, range_h, coordinates, location_name, product_type):
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #                   PARAMETERS
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    images = []
    # setting coordinates (diameter of 200 kilometers)
    distance_radius = 100000
    bbox_coordinates = get_bbox_coordinates_from_center(coordinates, distance_radius)
    # setting image dimension
    precision = 2000 # meters per pixel
    img_length = int(distance_radius*2/precision)
    dimension = {"width": img_length, "height": img_length}
    # gettin an image every 1.5 hours
    date_from_h = date_start + datetime.timedelta(hours=start_h)
    date_to_h = date_from_h
    for hours_counter in range(int(range_h / 1.5)):
        date_from_h = date_to_h
        date_to_h = date_to_h + datetime.timedelta(hours=1.5)
        date_from_str = [date_from_h.strftime("%Y-%m-%d"), date_from_h.strftime("%H:%M:%S")]
        date_to_str = [date_to_h.strftime("%Y-%m-%d"), date_to_h.strftime("%H:%M:%S")]
        print(date_from_str)
        # getting post responce
        response = get_response(bbox_coordinates, date_from_str, date_to_str, product_type, dimension)
        in_memory_file = io.BytesIO(response.content)
        img_png = Image.open(in_memory_file)
        img_png = array(img_png)
        # pushin the image in the list
        image = {
            "date": date_start,
            "time": hours_counter,
            "image": img_png
        }
        """fig = plt.figure()
        plt.imshow(img_png)
        plt.show()"""
        #save_image(image, product_type, location_name)
        images.append(image)
    save_json(images, product_type, location_name)













def main_downloader(date_start, date_end, start_h, range_h, coordinates, location_name, product_type):

    directory_path = "./Data/" + product_type + "/" + location_name + "/"
    save_json_w_name(directory_path, {"coordinates": coordinates}, "coordinates")

    for day_counter in range(int((date_end - date_start).days)):
        date = date_start + datetime.timedelta(days=day_counter)
        get_hourly_images(date, start_h, range_h, coordinates, location_name, product_type)

def main_downloader_default(date_start, date_end, coordinates):
    start_h = 0
    range_h = 24
    product_type = "NO2"
    location_name = "[" + str(coordinates[0]) + "," + str(coordinates[1]) + "]"
    main_downloader(date_start, date_end, start_h, range_h, coordinates, location_name, product_type)

def main_downloader_sabetta():

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #                       MAIN
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    values = {
        "product_types": ["NO2", "CO", "CH4", "SO2"],
        "locations_name": ["Bering Strait", "Sabetta Port"],
    }
    date = datetime.datetime.now()
    date_start = date.replace(year=2021, month=10, day=1, hour=0, minute=0, second=0, microsecond=0)
    date_end = date.replace(year=2021, month=11, day=1, hour=0, minute=0, second=0, microsecond=0)
    start_h = 0
    range_h = 24

    product_type = values["product_types"][0]
    location_name = values["locations_name"][1]

    coordinates = [71.264765, 72.060155]

    if location_name == "Sabetta Port":
        coordinates = [71.264765, 72.060155]
        start_h = 23
        range_h = 10.5

    main_downloader(date_start, date_end, start_h, range_h, coordinates, location_name, product_type)

    # FOR IMAGES DOWNLOAD
    # download_images(product_type, location_name, minQa, date_start, date_end)



date = datetime.datetime.now()
date_start = date.replace(year=2021, month=5, day=1, hour=0, minute=0, second=0, microsecond=0)
date_end = date.replace(year=2021, month=5, day=3, hour=0, minute=0, second=0, microsecond=0)
coordinates = [64.17296424691946, -51.68451411171911]
main_downloader_default(date_start, date_end, coordinates)